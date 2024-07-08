from biogeme import models
from biogeme.expressions import Beta
import biogeme.database as db
import biogeme.results as res
import biogeme.biogeme as bio
import pandas as pd
from sklearn.model_selection import train_test_split


def utility_function(database, n_alternatives: int):
    """This function define the parameters to be estimated and the utility dunction of each alternative

    Args:
        database (biogeme.database): biogeme database with the information of alternatives
        n_alternatives (int): number of alternatives between OD pairs

    Returns:
        dict: dictionary with the utiity function of each alternative
    """

    # Parameters to be estimated
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    ASC_PLANE = Beta('ASC_PLANE', 0, None, None, 0)
    ASC_MULTIMODAL = Beta('ASC_MULTIMODAL', 0, None, None, 1)

    B_TIME = Beta('B_TIME', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)
    B_CO2 = Beta('B_CO2', 0, None, None, 0)


    # We define the utility function for each alternative
    i = 1
    V = {}
    while i<=n_alternatives:
        V[i] = (ASC_TRAIN * database.variables[f'train_{i}'] + ASC_PLANE * database.variables[f'plane_{i}'] + ASC_MULTIMODAL * database.variables[f'multimodal_{i}'] +
                B_TIME * database.variables[f'travel_time_{i}'] +
                B_COST * database.variables[f'travel_cost_{i}'] +
                B_CO2 * database.variables[f'co2_{i}'])

        i+=1
    
    return V

def alternative_availability(database, n_alternatives: int):
    """This function define the variable in the database that defines the availability of each alternative

    Args:
        database (biogeme.database): biogeme database with the information of alternatives
        n_alternatives (int): number of alternatives between OD pairs

    Returns:
        dict: dictionary with the availability information of each alternative
    """
    # Availabilities of alternatives
    av = {i: database.variables[f'av_{i}'] for i in range(1, n_alternatives + 1)}
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
        reuslts (biogeme object): the object containing the results of the calibration
    """

    number_of_observations = database.getNumberOfObservations()
    number_of_trips = database.valuesFromDatabase(database.variables[weight_column]).sum()

    # We define de logit model based on the utility function (V), availability (av) and observed choice
    logprob = models.loglogit(V, av, database.variables['observed_choice'])
    formulas = {'loglike': logprob,
                'weight': database.variables[weight_column] / number_of_trips * number_of_observations }
    
    # We create the biogeme object and calibrate the model
    the_biogeme = bio.BIOGEME(database, formulas)
    the_biogeme.modelName = weight_column
    the_biogeme.generate_html  = False
    the_biogeme.generate_pickle = True
    the_biogeme.save_iterations = False

    results = the_biogeme.estimate()

    return the_biogeme, results

def test_data_analysis(database, V: dict, av: dict, n_alternatives: int, weight_column: str, beta_values: dict):
    # TO DO
    simulate = {}
    for i in range(1, n_alternatives + 1):
        prob_i = models.logit(V, av, i)
        trips_i = prob_i * database.variables[weight_column]

        simulate[f'prob_{i}'] = prob_i
        simulate[f'trips_{i}'] = trips_i

    biosim = bio.BIOGEME(database, simulate)
    biosim.modelName = weight_column + '_test'
    test_results = biosim.simulate(theBetaValues=beta_values)

    # df = pd.DataFrame({'observed_choice': database_test.valuesFromDatabase(database_test.variables['observed_choice']),
    #           'trips': database_test.valuesFromDatabase(database_test.variables['trips'])})

    # df.groupby(['observed_choice'])['trips'].sum() / df['trips'].sum()

def calibrate_main(database_path: str, n_archetypes: int, n_alternatives: int):
    """Main function to calibrate the logit model

    Args:
        database_path (str): path to the csv file with the information to calibrate the model
        n_archetypes (int): number of archetypes
        n_alternatives (int): number of alternatives between OD pairs
    """
    # TO DO: create a configuration file. The input will only be the configuration file. For now we include other inputs

    od_matrix = pd.read_csv(database_path)

    od_matrix_train, od_matrix_test = train_test_split(od_matrix, test_size=0.2, random_state = 42 )

    database_train = db.Database("train", od_matrix_train )
    database_test = db.Database("train", od_matrix_test )

    for k in range(n_archetypes):
        weight_column = f'archetype_{k}'

        V = utility_function(database_train, n_alternatives)
        av = alternative_availability(database_train, n_alternatives)

        the_biogeme, results = logit_model(database_train, V, av, weight_column)

        print(results.short_summary())
        print(results.getEstimatedParameters())
        print(results.getBetaValues())

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

def predict_main(paths_file: str, n_archetypes: int, n_alternatives: int):
    """_summary_

    Args:
        paths_file (str): path to the csv file with the information of alternatives between each OD pair
        n_archetypes (int): number of archetypes
        n_alternatives (int): number of alternatives between OD pairs

    Returns:
        pd.DataFrame: probability of each alternative for each arquetype and OD pair
    """
    # TO DO: create a configuration file. The input will only be the configuration file. For now we include other inputs

    paths = pd.read_csv(paths_file)
    # the paths file should have the following columns: [origin, destination, train_i, plane_i, multimodal_i, travel_time_i, travel_cost_i, co2_i, av_i]
    # i=1:k where k is the number of alternatives
    
    cols = [x for x in paths.columns if (x!= 'origin') & (x!= 'destination')]

    database = db.Database("prediction", paths[cols] )

    df_results = paths[['origin', 'destination']].copy()

    for k in range(n_archetypes):
        weight_column = f'archetype_{k}'
        beta_values = res.bioResults(pickleFile=f'/opt/dev/data/{weight_column}.pickle').getBetaValues()

        V = utility_function(database, n_alternatives)
        av = alternative_availability(database, n_alternatives)

        probabilities = predict_probabilities(database, V, av, n_alternatives, weight_column, beta_values)

        df_results = pd.concat([df_results, probabilities], axis=1, join="inner").rename({i: f'{weight_column}_{i}' for i in probabilities.columns }, axis = 1)

    return df_results

def assign_passengers2path(row, paths_prob_dict: dict, alternative_i: int):
    """Assign how many passengers belonging to an acrhetype choose the alternative i for the OD pair

    Args:
        row (_type_): row of the pax_demand csv
        paths_prob_dict (dict): probability information
        alternative_i (int): alternative id

    Returns:
        float: volumne of trips for an archetype and alternative
    """
    O = row['origin']
    D = row['destination']
    arc = row['archetype']
    trips = row['trips']

    name = f'{arc}_prob_{alternative_i}'

    return trips * paths_prob_dict[(O, D)][name]

def assign_passengers_main(paths_prob: pd.DataFrame, n_alternatives: int, pax_demand_path: str):
    """Main function to assign trips to paths

    Args:
        paths_prob (pd.DataFrame): probability of each alternative for each arquetype and OD pair
        n_alternatives (int): number of alternatives between OD pairs
        pax_demand_path (str): path to the pax_demand file

    Returns:
        pd.DataFrame: df with the trips assigned to each alternative
    """
    
    pax_demand = pd.read_csv(pax_demand_path)
    paths_prob_dict = paths_prob.set_index(['origin', 'destination']).to_dict('index')

    for i in range(1, n_alternatives + 1):
        pax_demand[f'alternative_{i}'] = pax_demand.apply(lambda row: assign_passengers2path(row, paths_prob_dict, i), axis = 1)

    return pax_demand