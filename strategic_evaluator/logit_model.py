from pathlib import Path
from biogeme import models
from biogeme.expressions import Beta
import numpy as np
import biogeme.database as db
import biogeme.results as res
import biogeme.biogeme as bio
import pandas as pd
import re
import logging
from script.strategic.launch_parameters import n_alternatives_max, n_archetypes
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def utility_function(database, n_alternatives: int, fixed_params: dict = None, logit_type: str ="spanish"):
    """This function defines the parameters to be estimated and the utility function of each alternative

    Args:
        database (biogeme.database): biogeme database with the information of alternatives
        n_alternatives (int): number of alternatives between OD pairs
        fixed_params (dict): dictionary with variables that have already been calibrated in a previous iteration
        logit_type (str): variable that determines the type of logit model. "spanish" for the national case and "international" for the international one
    Returns:
        dict: dictionary with the utility function of each alternative
    """
    if logit_type=="spanish": # base case for international logit
        #it would be good to add a configuration file or something like that 
        # Parameters to be estimated
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0) # 1 means you do not estimate it
        ASC_PLANE = Beta('ASC_PLANE', 0, None, None, 0) # 0 means you estimate it
        ASC_MULTIMODAL = Beta('ASC_MULTIMODAL', 0, None, None, 1)

        # Retrieve fixed values for B_COST and B_TIME, or estimate them if not provided
        B_TIME = Beta('B_TIME', fixed_params['B_TIME'] if fixed_params and 'B_TIME' in fixed_params else 0, None, None, 
                    1 if fixed_params and 'B_TIME' in fixed_params else 0)
        B_COST = Beta('B_COST', fixed_params['B_COST'] if fixed_params and 'B_COST' in fixed_params else 0, None, None, 
                    1 if fixed_params and 'B_COST' in fixed_params else 0)

        # B_CO2 is estimated only if B_TIME and B_COST are fixed
        B_CO2 = Beta('B_CO2', 0, None, None, 0) if (fixed_params and 'B_TIME' in fixed_params and 'B_COST' in fixed_params) else Beta('B_CO2', 0, None, None, 1)

        # Define the utility function for each alternative
        V = {}
        for i in range(1, n_alternatives + 1):
            V[i] = (ASC_TRAIN * database.variables[f'train_{i}'] + 
                    ASC_PLANE * database.variables[f'plane_{i}'] + 
                    ASC_MULTIMODAL * database.variables[f'multimodal_{i}'] +
                    B_TIME * database.variables[f'travel_time_{i}'] +
                    B_COST * database.variables[f'cost_{i}'] +
                    B_CO2 * database.variables[f'emissions_{i}'])
    
    if logit_type == "international":
        #the following lines are only relevant for calibration purposes
        ASC_PLANE = Beta('ASC_PLANE', 0, None, None, 0) # 0 means you estimate it
        ASC_MULTIMODAL = Beta('ASC_MULTIMODAL', 0, None, None, 1) #1 means you do not estimate it
        B_COST= Beta("B_COST",0,None, None, 0)
        B_TIME=Beta("B_TIME",0,None, None, 0)
        # Define the utility function
        V={}
        for i in range(1,n_alternatives+1):
            V[i]=   (ASC_PLANE * database.variables[f'plane_{i}'] + 
                    ASC_MULTIMODAL * database.variables[f'multimodal_{i}'] +
                    B_TIME * database.variables[f'travel_time_{i}'] +
                    B_COST * database.variables[f'cost_{i}'])
        
    return V


def alternative_availability(database, n_alternatives: int):
    """This function defines the variable in the database that defines the availability of each alternative

    Args:
        database (biogeme.database): biogeme database with the information of alternatives
        n_alternatives (int): number of alternatives between OD pairs

    Returns:
        dict: dictionary with the availability information of each alternative
    """
    # Availabilities of alternatives
    av = {i: database.variables[f'av_{i}']
          for i in range(1, n_alternatives + 1)}
    return av


def logit_model(database, V: dict, av: dict, weight_column: str):
    """Define and calibrate the logit model for an archetype defined by the "weight_column"

    Args:
        database (biogeme.database): biogeme database with the information of alternatives
        V (dict): utility functions
        av (dict): availability functions
        weight_column (str): name of the variable indicating the number of trips

    Returns:
        the_biogeme (biogeme object): the object defining the biogeme model
        results (biogeme object): the object containing the results of the calibration
    """

    number_of_observations = database.getNumberOfObservations()
    number_of_trips = database.valuesFromDatabase(
        database.variables[weight_column]).sum()

    # We define de logit model based on the utility function (V), availability (av) and observed choice
    logprob = models.loglogit(V, av, database.variables['observed_choice'])
    formulas = {'loglike': logprob,
                'weight': database.variables[weight_column] / number_of_trips * number_of_observations}

    # We create the biogeme object and calibrate the model
    the_biogeme = bio.BIOGEME(database, formulas)
    the_biogeme.modelName = weight_column
    the_biogeme.generate_html = False
    the_biogeme.generate_pickle = True
    the_biogeme.save_iterations = False

    results = the_biogeme.estimate()

    return the_biogeme, results


def test_data_analysis(database, V: dict, av: dict, n_alternatives: int, weight_column: str, beta_values: dict):
    # TO DO
    simulate = {}
    for i in range(1, n_alternatives + 1):
        prob_i = models.logit(V, av, i)
        # trips_i = prob_i * database.variables[weight_column]

        simulate[f'prob_{i}'] = prob_i
        # simulate[f'trips_{i}'] = trips_i

    biosim = bio.BIOGEME(database, simulate)
    biosim.modelName = weight_column + '_test'
    test_results = biosim.simulate(theBetaValues=beta_values)
    return test_results

    # df = pd.DataFrame({'observed_choice': database_test.valuesFromDatabase(database_test.variables['observed_choice']),
    #           'trips': database_test.valuesFromDatabase(database_test.variables['trips'])})

    # df.groupby(['observed_choice'])['trips'].sum() / df['trips'].sum()


def calibrate_main(database_path: str, n_archetypes: int, n_alternatives: int, fixed_params: dict=None,):
    """Main function to calibrate the logit model

    Args:
        database_path (str): path to the csv file with the information to calibrate the model
        n_archetypes (int): number of archetypes
        n_alternatives (int): number of alternatives between OD pairs
        o_d_info (pd.Dataframe): information about the origin and destination of the trips
        fixed_params (dict): dictionary with the fixed parameters coming from a previous calibration
    """
    # TO DO: create a configuration file. The input will only be the configuration file. For now we include other inputs

    if fixed_params is None:
        fixed_params = {} #set default value in the case there are no fixed parameters

    od_matrix = pd.read_csv(database_path)
    
    od_matrix_train, od_matrix_test = train_test_split(
        od_matrix, test_size=0.2, random_state=43)

    database_train = db.Database("train", od_matrix_train)
    database_test = db.Database("test", od_matrix_test)

    test_results_archetypes={}
    if n_archetypes>1:
        for k in range(n_archetypes):
            weight_column = f'archetype_{k}'
            archetype_fixed_params = fixed_params.get(f"archetype_{k}", {})

            V = utility_function(database_train, n_alternatives,archetype_fixed_params)
            av = alternative_availability(database_train, n_alternatives)

            the_biogeme, results = logit_model(
                database_train, V, av, weight_column)
            beta_values=results.get_beta_values()
            # beta_values.update(archetype_fixed_params)
            test_results=test_data_analysis(database_test, V, av, n_alternatives, weight_column, beta_values)
            # test_results=pd.merge(test_results,o_d_info_archetype,left_index=True,right_index=True,how="inner") #gets the origin and destination information
            # for i in range(1,n_alternatives+1):
            #     trips_i=f"trips_{i}"
            #     prob_i=f"prob_{i}"
            #     test_results[trips_i]=test_results[f"trips_per_od_pair_arch_{k}"]*test_results[prob_i]
            test_results_archetypes[f"test_results_archetype_{k}"]=test_results
    elif n_archetypes==1:
        weight_column="trips"
        # no need to specify fixed parameter since the default is none
        V = utility_function(database=database_train, n_alternatives=n_alternatives)
        av = alternative_availability(database_train, n_alternatives)
        the_biogeme, results = logit_model(
            database_train, V, av, weight_column)
        beta_values=results.get_beta_values()
        test_results=test_data_analysis(database_test, V, av, n_alternatives, weight_column, beta_values)
        test_results_archetypes[f"test_results"]=test_results

        print("Training results:")
        print(results.short_summary())
        print(results.getEstimatedParameters())
        print(results.getBetaValues())
        print("Test results:")
        print(beta_values)
        print(test_results)
    return test_results_archetypes
        
 


def predict_probabilities(database, V: dict, av: dict, n_alternatives: int, weight_column: str, beta_values: dict):
    """Estimate the probabilities of each alternative for an archetype ("weight_column")

    Args:
        database (biogeme.database): biogeme database with the information of alternatives
        V (dict): utility functions
        av (dict): availability functions
        n_alternatives (int): number of alternatives between OD pairs
        weight_column (str): archetype
        beta_values (dict): value of the parameters of the logit model

    Returns:
        pd.DataFrame: df with the probabilities of each alternative
    """
    simulate = {}
    for i in range(1, n_alternatives + 1):
        prob_i = models.logit(V, av, i)

        simulate[f'prob_{i}'] = prob_i

    biosim = bio.BIOGEME(database, simulate)
    biosim.modelName = weight_column + '_test'
    probabilities = biosim.simulate(theBetaValues=beta_values)

    return probabilities


def predict_main(paths: pd.DataFrame, n_archetypes: int, n_alternatives: int, sensitivities: str, fixed_params: dict = None, logit_type: str="spanish"):
    """_summary_

    Args:
        paths_file (str): path to the csv file with the information of alternatives between each OD pair
        n_archetypes (int): number of archetypes
        n_alternatives (int): number of alternatives between OD pairs

    Returns:
        pd.DataFrame: probability of each alternative for each archetype and OD pair
    """
    # TO DO: create a configuration file. The input will only be the configuration file. For now we include other inputs

    # paths = pd.read_csv(paths_file)
    # the paths file should have the following columns: [origin, destination, train_i, plane_i, multimodal_i, travel_time_i, cost_i, emissions_i, av_i]
    # i=1:k where k is the number of alternatives

    cols = [
        x for x in paths.columns if not x in (
            ['origin', 'destination', 'alternative_id', 'observed_choice'] +
            [f"alternative_id_{i}" for i in range(1, n_alternatives+1)]
        )
    ]

    database = db.Database("prediction", paths[cols])

    df_results = paths[['origin', 'destination']].copy()

    if fixed_params is None:
        fixed_params = {} #set default value in the case there are no fixed parameters

    for k in range(n_archetypes):
        # logger.important_info(
            # f"Predicting number of passenger on each path for archetype {k}."
        # )
        weight_column = f'archetype_{k}'
        archetype_fixed_params = fixed_params.get(f"archetype_{k}", {})
        beta_values = res.bioResults(
            pickleFile=(
                Path(sensitivities["sensitivities"]) / f"{weight_column}.pickle")
        ).getBetaValues()
        # beta_values.update(archetype_fixed_params)

        V = utility_function(database=database, 
                             n_alternatives=n_alternatives, 
                             fixed_params=archetype_fixed_params,
                             logit_type=logit_type
                             )

        av = alternative_availability(database, n_alternatives)

        probabilities = predict_probabilities(
            database, V, av, n_alternatives, weight_column, beta_values
        )

        df_results = pd.concat(
            [df_results, probabilities], axis=1, join="inner"
        ).rename({i: f'{weight_column}_{i}' for i in probabilities.columns}, axis=1)

    return df_results.drop_duplicates()


def assign_passengers2path(row, paths_prob_dict: dict, alternative_i: int):
    """Assign how many passengers belonging to an archetype choose the alternative i for the OD pair

    Args:
        row (_type_): row of the pax_demand csv
        paths_prob_dict (dict): probability information
        alternative_i (int): alternative id

    Returns:
        float: volume of trips for an archetype and alternative
    """
    O = row['origin']
    D = row['destination']
    arc = row['archetype']
    trips = row['trips']

    name = f'{arc}_prob_{alternative_i}'
    if (O, D) in paths_prob_dict.keys():
        return trips * paths_prob_dict[(O, D)][name]
    else:
        pass
        # logger.important_info(f"No paths on OD pair {O, D}")

def get_probability_path_archetype(row, paths_prob_dict: dict, alternative_i: int):
    O = row['origin']
    D = row['destination']
    arc = row['archetype']
    trips = row['trips']

    name = f'{arc}_prob_{alternative_i}'
    if (O, D) in paths_prob_dict.keys():
        return paths_prob_dict[(O, D)][name]
    else:
        pass
        # logger.important_info(f"No paths on OD pair {O, D}")



def assign_passengers_main(paths_prob: pd.DataFrame, n_alternatives: int, pax_demand_path: str):
    """Main function to assign trips to paths

    Args:
        paths_prob (pd.DataFrame): probability of each alternative for each archetype and OD pair
        n_alternatives (int): number of alternatives between OD pairs
        pax_demand_path (str): path to the pax_demand file

    Returns:
        pd.DataFrame: df with the trips assigned to each alternative
    """

    pax_demand = pd.read_csv(pax_demand_path)
    paths_prob_dict = paths_prob.set_index(
        ['origin', 'destination']).to_dict("index")

    for i in range(1, n_alternatives + 1):
        pax_demand[f'alternative_{i}'] = pax_demand.apply(
            lambda row: assign_passengers2path(row, paths_prob_dict, i), axis=1)
        pax_demand[f'alternative_prob_{i}'] = pax_demand.apply(
            lambda row: get_probability_path_archetype(row, paths_prob_dict, i), axis=1)

    return pax_demand


def select_paths(paths: pd.DataFrame, n_alternatives_max: int):
    paths_filtered = paths.sort_values("total_travel_time").groupby(
        ["origin", "destination"], as_index=False
    ).apply(lambda x: x.iloc[:n_alternatives_max]).reset_index(level=0, drop=True)
    return paths_filtered


def format_paths_for_predict(
    paths: pd.DataFrame, n_alternatives: int,
    max_connections: int, network_paths_config: str, paths_clusters: bool = True
):
    if paths_clusters:
        paths["train"] = paths["journey_type"] == "rail"
        paths["plane"] = paths["journey_type"] == "air"
        paths["multimodal"] = paths["journey_type"] == "multimodal"
    else:
        paths["mode_tp"] = paths.apply(lambda row: [
            row[f"mode_{i}"] for i in range(max_connections)if str(row[f"mode_{i}"]) != "nan"
        ], axis=1)
        paths["alternative_id"] = (
            paths["origin"] + "_" + paths["destination"] + "_" + paths["option"].astype(str)
        )

        paths["train"] = paths.apply(
            lambda row: "rail" in row["mode_tp"], axis=1).astype(int)
        paths["plane"] = paths.apply(
            lambda row: "air" in row["mode_tp"], axis=1).astype(int)
        paths["multimodal"] = (paths["nmodes"] > 1).astype(int)

    paths["option_number"] = paths.groupby(["origin", "destination"])["alternative_id"].transform(
            lambda df: range(1, len(df)+1)
        ).astype(str)
    paths_base = paths[[
        "origin", "destination", "option_number", "alternative_id",
        "multimodal", "train", "plane",
        'total_travel_time', 'total_cost', 'total_emissions'
    ]].sort_values(["origin", "destination"])

    paths_pivoted = pivot_paths(paths_base)
    #paths_pivoted_final = paths_pivoted.groupby(["origin", "destination"]).apply(
    #    lambda df: df.bfill().ffill(), include_groups=False
    #).reset_index().drop_duplicates(
    #    ["origin", "destination", "path_id"]
    #).drop(columns=["level_2"]).fillna(0).reset_index(drop=True)

    paths_pivoted_final = paths_pivoted.groupby(["origin", "destination"]).apply(
        lambda df: df.bfill().ffill()
    ).reset_index(drop=True).drop_duplicates(
        ["origin", "destination", "alternative_id"]
    ).drop(columns=["level_2"], errors='ignore').fillna(0).reset_index(drop=True)

    paths_pivoted_final.columns = [
        i.replace("total_", "") for i in paths_pivoted_final.columns
    ]

    for i in range(1, n_alternatives+1):
        paths_pivoted_final[f"av_{i}"] = (paths_pivoted_final[
            f"travel_time_{i}"
        ] != 0).astype(int)

    paths_pivoted_final.to_csv(
        Path(
            network_paths_config['output']['output_folder']
        ) / "paths_pivoted_final.csv", index=False
    )

    return paths_pivoted_final


def pivot_paths(paths: pd.DataFrame):
    paths_pivoted = paths.pivot_table(
        index=["origin", "destination", "alternative_id"],
        columns='option_number',
        values=[
            "multimodal", "train", "plane",
            'total_travel_time', 'total_cost', 'total_emissions'
        ],
        aggfunc='mean'
    ).reset_index()
    paths_pivoted.columns = [
        ("_".join(pair)).rstrip("_") for pair in paths_pivoted.columns
    ]

    # Keep order of alternatives
    paths_indexed = paths.set_index('alternative_id')
    paths_pivoted_indexed = paths_pivoted.set_index('alternative_id')
    path_pivoted_ordered = paths_pivoted_indexed.loc[paths_indexed.index]

    paths_pivoted = path_pivoted_ordered.reset_index()

    #paths_pivoted_updated = paths_pivoted.groupby(["origin", "destination"]).apply(
    #    assign_path_id_columns, include_groups=False).reset_index().drop(columns="level_2")

    paths_pivoted_updated = paths_pivoted.groupby(["origin", "destination"]).apply(
        assign_path_id_columns
    ).reset_index(drop=True).drop(columns="level_2", errors='ignore')

    #paths_pivoted_updated = paths_pivoted.groupby(["origin", "destination"]).apply(
    #    assign_path_id_columns
    #).reset_index(drop=True).drop(columns="level_2", errors='ignore')

    paths_pivoted_updated["observed_choice"] = paths_pivoted_updated.groupby(["origin", "destination"])["alternative_id"].transform(
        lambda df: range(1, len(df)+1)
    ).astype(str)

    return paths_pivoted_updated


def assign_path_id_columns(df: pd.DataFrame):
    paths_ids = df["alternative_id"].values
    for i in range(1, len(paths_ids)+1):
        df[f"alternative_id_{i}"] = paths_ids[i-1]
    return df


def assign_demand_to_paths(
    paths: pd.DataFrame, 
    n_alternatives: int,
    n_archetypes: int,
    max_connections: int, 
    network_paths_config: str,
    logit_type: str="spanish"
):
    logger.important_info("Predict demand on paths")

    # Selecting paths if necessary (too many options)
    # paths = paths.pipe(select_paths, n_alternatives_max)
    paths_final = paths.pipe(
        format_paths_for_predict, n_alternatives, max_connections, network_paths_config
    )

    paths_probabilities = predict_main(
        paths_final, n_archetypes=n_archetypes,
        n_alternatives=n_alternatives, 
        logit_type=logit_type,
        sensitivities=network_paths_config['other_param']['sensitivities_logit']
    )

    pax_demand_paths = assign_passengers_main(
        paths_probabilities, n_alternatives, Path(
            network_paths_config['demand']['demand'])
    )

    # Get a list of all columns that end with '_prob'
    prob_columns = [col for col in pax_demand_paths.columns if '_prob' in col]

    # Get the list of remaining columns
    remaining_columns = [col for col in pax_demand_paths.columns if col not in prob_columns]

    # Create the new order of columns
    new_column_order = remaining_columns + prob_columns

    # Reorder the DataFrame
    pax_demand_paths = pax_demand_paths[new_column_order]

    # Round alternatives

    alternative_columns = [col for col in pax_demand_paths.columns if re.match(r'alternative_\d+$', col)]

    # Round probabilites
    pax_demand_paths[prob_columns] = pax_demand_paths[prob_columns].round(3)

    # Round pax
    pax_demand_paths[alternative_columns] = pax_demand_paths[alternative_columns].round(2)



    return pax_demand_paths, paths_final

def calibration_results_summary(archetypes: dict):
    data=[]
    for i in range(6):
        
        archetype = archetypes[f"archetype_{i}"] # accesses the values  
        values=archetype.get_estimated_parameters() # dataframe of the betavalues 
        

        asc_plane = values.loc['ASC_PLANE']["Value"]  
        asc_plane_significance=(values.loc["ASC_PLANE"]["Rob. p-value"]<0.05)

        asc_train = values.loc['ASC_TRAIN']["Value"]
        asc_train_significance=(values.loc["ASC_TRAIN"]["Rob. p-value"]<0.05)

        b_co2 = values.loc['B_CO2']["Value"]      
        b_co2_significance=(values.loc["B_CO2"]["Rob. p-value"]<0.05)

        b_time = values.loc['B_TIME']["Value"]
        b_time_significance=(values.loc["B_TIME"]["Rob. p-value"]<0.05)

        b_cost=values.loc["B_COST"]["Value"]
        b_cost_significance=(values.loc["B_COST"]["Rob. p-value"]<0.05)

        # Create a row dictionary
        row = {
            "Archetype": f"Archetype {i}",
            "ASC_PLANE": asc_plane,
            "ASC_PLANE_SIGNIFICANCE": asc_plane_significance,
            "ASC_TRAIN": asc_train,
            "ASC_TRAIN_SIGNIFICANCE": asc_train_significance,
            "B_CO2": b_co2,
            "B_CO2_SIGNIFICANCE": b_co2_significance,
            "B_TIME": b_time,
            "B_TIME_SIGNIFICANCE": b_time_significance,
            "B_COST": b_cost,
            "B_COST_SIGNIFICANCE": b_cost_significance
        }
        
        # Append the row to the list
        data.append(row)

    # Create a DataFrame from the data
    calibration_results = pd.DataFrame(data)
    return calibration_results

def test_logit(test: dict, trips_logit: pd.DataFrame, n_alternatives=5):
    """generates the test of the logit model. Note that the trips_logit dataframe has to to
    match whether we considered rows with only one alternative or not
    
    Args:
        test: dictionary obtained in the calibration
        trips_logit: dataframe that contains the information for logit calibration
        n_alternatives: the max number of alternatives. Default set to 5"""
    final_test={}
    if n_alternatives>1:
        for i in range(n_alternatives+1):
            #initialise strings
            archetype=f"archetype_{i}"
            test_results=f"test_results_archetype_{i}"
            test_archetype=test[test_results]

            #define the dataframe that will contain the final comparison
            final_test[archetype]=pd.DataFrame(columns=["location","prob_predicted","prob_observed"])
            final_test[archetype]["location"]=test_archetype.index

            #iterate over the index and row to fetch the probabilites
            for idx, row in final_test[archetype].iterrows():
                num=trips_logit.iloc[row["location"]]["noption"]
                final_test[archetype].loc[idx,"prob_predicted"]=test_archetype.loc[row["location"],f"prob_{num}"]
                final_test[archetype].loc[idx,"prob_observed"]=trips_logit.iloc[row["location"]][f"prob_per_od_pair_arch_{archetype[-1]}"]
    elif n_alternatives==1:
        #initialise strings
        archetype="archetype_0"
        test_archetype=test["test_results"]

        #define the dataframe that will contain the final comparison
        final_test[archetype]=pd.DataFrame(columns=["location","prob_predicted","prob_observed"])
        final_test[archetype]["location"]=test_archetype.index

        for idx, row in final_test[archetype].iterrows():
            num=trips_logit.iloc[row["location"]]["noption"]
            final_test[archetype].loc[idx,"prob_predicted"]=test_archetype.loc[row["location"],f"prob_{num}"]
            final_test[archetype].loc[idx,"prob_observed"]=trips_logit.iloc[row["location"]][f"prob_per_od_pair_arch_0"]
            #print("in the making")
    return final_test

def evaluate_model(test_data, model_name):
    """
    Evaluate the model's performance by calculating MSE, MAE, Pearson correlation, and Spearman correlation.

    Args:
    - test_data (pd.DataFrame): A DataFrame containing 'prob_observed' and 'prob_predicted' columns.
    - model_name (str): The name of the model being evaluated.

    Returns:
    - mse (float): Mean Squared Error.
    - mae (float): Mean Absolute Error.
    - pearson_corr (float): Pearson correlation coefficient.
    - spearman_corr (float): Spearman correlation coefficient.
    """

    # Convert columns to numeric (in case they are not already)    test_data['prob_observed'] = pd.to_numeric(test_data['prob_observed'], errors='coerce')
    test_data['prob_predicted'] = pd.to_numeric(test_data['prob_predicted'], errors='coerce')
    test_data['prob_observed'] = pd.to_numeric(test_data['prob_observed'], errors='coerce')
    
    # Remove rows with NaN values in 'prob_observed' or 'prob_predicted'
    test_data = test_data.dropna(subset=['prob_observed', 'prob_predicted'])

    # Extract true and predicted values
    y_true = test_data['prob_observed']
    y_pred = test_data['prob_predicted']

    # calculate residuals
    residuals = y_true - y_pred

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    # Calculate dispersion of residuals (standard deviation of residuals)
    residual_dispersion = np.std(residuals)

    # Print evaluation results
    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print(f"Standard deviation: {residual_dispersion:.4f}")
    return mse, mae, pearson_corr, spearman_corr, residual_dispersion


def drop_nan(test_data):
    """Function to clean the data before using the dispersion plot
    
    Args:
        test data: dataframe with columns prob_predicted
        
    Returns:
        data cleaned"""
    initial_rows=test_data.shape[0]

    #we drop rows with nan values
    data_cleaned=test_data.dropna()

    #we remove the row that have prob=1
    data_cleaned=data_cleaned[data_cleaned["prob_predicted"]!=1]
    final_rows=data_cleaned.shape[0]

    #we print how many rows we have removed
    rows_removed=initial_rows - final_rows
    print(f"{rows_removed} rows were removed")
    return data_cleaned

def calibration_plot(test_data, model_name):
    """Function that does a dispersion plot. It makes 10 linear bins and 
    groups probabilities predicted vs observed in these bins
    
    Args: 
        test_data: dataframe with columns prob_observed and prob_predicted"""
    #clean the data if needed
    data_cleaned=drop_nan(test_data)

    #convert to numeric data if needed
    data_cleaned.loc[:,'prob_observed'] = pd.to_numeric(data_cleaned['prob_observed'], errors='coerce')
    data_cleaned.loc[:,'prob_predicted'] = pd.to_numeric(data_cleaned['prob_predicted'], errors='coerce')

    #creates bins
    bins = np.linspace(0, 1, 11)
    data_cleaned.loc[:,'bin'] = pd.cut(data_cleaned['prob_predicted'], bins)
    grouped = data_cleaned.groupby('bin',observed=False).mean()
    
    #plot options
    plt.plot(grouped['prob_predicted'], grouped['prob_observed'], "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Mean Observed Probability")
    plt.legend()
    plt.title("Calibration Plot")
    plt.show()