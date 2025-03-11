import pandas as pd
import re


def remove_useless_columns(trips: pd.DataFrame):
    """This function removes the columns that do not provide any useful
    information for the MultiModX project

    Args:
        trips dataframe

    Returns:
        final trips with the removed columns.
    """
    columns_to_drop=["trip_period",
    "distance",
    "route_distance",
    "duration",
    "mode",
    "service",
    "trip_vehicle_type",
    "home_census",
    "home_zone",
    "overnight_census",
    "income",
    "age",
    "sex",
    "vehicle_type",
    "short_professional_driver",
    "trips_km",
    "sample_trips"
    ]
    starting_columns=trips.columns
    # remove the unwanted columns
    trips_final=trips.drop(columns=[col for col in columns_to_drop if col in trips.columns])
    ending_columns=trips_final.columns
    num_removed_columns=len(set(starting_columns)-set(ending_columns))
    print(f"{num_removed_columns} columns were removed")
    return trips_final


def format_trips(trips: pd.DataFrame, airports_to_NUTS: dict):
    '''
    This is the main function to format trips. It takes as inputs the initial dataframe and a dictionary with
    the airports and their NUTS to actualise the trips from the canary island

    Args:
        pandas dataframe with trips from MITMA
        dictionary with canary island airports

    Returns:
        the modified dataframe
    '''
    # Modify 'mode_tp' column: replacing modes with specific terminology
    trips.loc[:, "mode_tp"] = (
        trips["mode_sequence"]
        .str.replace("bus", "road")  # replace bus to road (some people can reach infrastructure by bus)
        .str.replace("plane", "air")  # use nomenclature of the offer data
        .str.replace("train", "rail")
    )

    # Remove "road" from the 'mode_tp' column
    trips.loc[:, "mode_tp"] = trips["mode_tp"].apply(
        lambda row: [mode for mode in row.split("-") if mode != "road"]
    )  # remove "road" (it will be considered like access time)

    # Only consider trips that do not contain "ship"
    trips = trips[~trips["mode_tp"].apply(lambda x: "ship" in x)]

    # Change aggregated island NUTS to dis-aggregated NUTS
    for key in airports_to_NUTS.keys():
        trips.loc[trips["start_node"] == key, ["origin", "origin_name"]] = [
            airports_to_NUTS[key][0],
            airports_to_NUTS[key][1]
        ]  # change start node
        trips.loc[trips["end_node"] == key, ["destination", "destination_name"]] = [
            airports_to_NUTS[key][0],
            airports_to_NUTS[key][1]
        ]  # change destination node
    trips.loc[:,"mode_tp"]=trips["mode_tp"].apply(str) #change column type to string 

    # remove columns that are not useful in the context of MultiModX
    trips_final=remove_useless_columns(trips)

    # groupby by all columns except trips which gets summed up
    cols=list(trips_final.columns)
    cols.remove("trips")
    trips_final = trips_final.groupby(cols, as_index=False, dropna=False)['trips'].sum()

    return trips_final


def find_weird_stations(node_sequence: pd.DataFrame,stops_loc: pd.DataFrame):
    '''Function to identify the stations that are potentially un-localisable from the node sequence

    Args: 
        column of trips dataframe
        dataFrame that contains information about station names and coordinates

    Returns:
        a list of all the weird stations for that specific trip
    '''
    weird_stations=[]
    nodes = node_sequence.split("-")
    for node in nodes:
        if node.startswith("train_"):
            station_id=node.split("_")[1]
            if not station_id.isdigit():
                weird_stations.append(station_id)
            else:
                station_id_modified_1 = f"0071{int(station_id):05d}"
                station_id_modified_2 = f"0087{int(station_id):05d}"
                station_id_modified_3 = f"0094{int(station_id):05d}"
                if any(station_id in stops_loc["stop_id"].values for station_id in [station_id_modified_1, station_id_modified_2, station_id_modified_3]):
                    pass
                else:
                    final_id="train_"+ station_id
                    weird_stations.append(final_id)

    return weird_stations


def get_weird_stations(weird_stations: pd.DataFrame):
    '''Function to extract a list of all the unique weird station of the trips dataframe

    Args: 
        column of the dataframe containing the list of weird stations per trip
    
    Returns:
        a list of the unique weird stations
    '''
    all_weird_stations = weird_stations.explode().dropna()

    # Extract unique values
    unique_weird_stations = all_weird_stations.unique()

    # Convert back to a list if needed
    unique_weird_stations = list(unique_weird_stations)
    return unique_weird_stations


def remove_abroad_train_trips(trips: pd.DataFrame):
    """Function to remove the trips leaving or entering spain via ground 
    and maintains the trips that do not go abroad
    
    Args:
        trips dataframe
    
    Returns:
        trips dataframe
    """
    #slipt trips into trips that go abroad and trips that do not go abroad
    trips_abroad=trips[(trips["origin"]=="abroad")|(trips["destination"]=="abroad")]
    trips_no_abroad=trips[~((trips["origin"]=="abroad")|(trips["destination"]=="abroad"))]
    
    #filter trips that go abroad to remove the ground trips
    trips_abroad = trips_abroad[
        (trips_abroad["entry_point"].str.contains("airport", na=False)) |
        (trips_abroad["exit_point"].str.contains("airport", na=False))
    ]
    #concatenate the new abroad trips 
    trips_final=pd.concat([trips_no_abroad,trips_abroad],axis=0)
    return trips_final


def remove_numbers_fr_pt(column: pd.Series):
    """This function replaces the zones of france and portugal by FR or PT and 
    the abroad_mcc code by only the code
    
    Args:
        columns of the trips dataframe
        
    Returns
        columns of the trips dataframe"""
    # Remove numbers for FR and PT
    column = column.str.replace(r"^(FR|PT)[a-zA-Z0-9]{3}$", r"\1", regex=True)
    # Remove 'abroad_' prefix
    column = column.astype(str).str.replace(r"^abroad_", "", regex=True)
    return column


def map_mcc_to_country(zone, mcc_to_country: dict):
    """This function changes the origin or destination zone of a trips 
    dataframe to the country of origin if the trip was taken abroad. 
    It is meant to be applied column-wise
    
    Args: 
        the origin or the destination zone
        a dictionary containing the equivalence between mcc codes (3-digit codes)
        to country
    
    Returns: 
        the country if there is a correspondence between the mcc code
    """
    if zone in ["FR","PT"]:
        return zone
    elif re.match(r'^\d{3}$', zone):
        return mcc_to_country.get(int(zone),zone)
    else:
        return zone
    

def precise_origin_destination(trips_abroad: pd.DataFrame):
    """This function changes the name of the origin and the destination
    from 'abroad' to the exact country and drops the origin_zone and 
    destination_zone columns.
    
    Args:
        trips_abroad (pd.DataFrame): Input DataFrame with trip data.

    Returns:
        pd.DataFrame: Modified DataFrame with updated origin/destination and dropped columns.
    """
    # Update 'origin' and 'origin_name' where origin is 'abroad'
    trips_abroad.loc[trips_abroad["origin"] == "abroad", "origin"] = trips_abroad["origin_zone"]
    trips_abroad.loc[trips_abroad["origin_name"] == "abroad", "origin_name"] = trips_abroad["origin_zone"]

    # Update 'destination' and 'destination_name' where destination is 'abroad'
    trips_abroad.loc[trips_abroad["destination"] == "abroad", "destination"] = trips_abroad["destination_zone"]
    trips_abroad.loc[trips_abroad["destination_name"] == "abroad", "destination_name"] = trips_abroad["destination_zone"]

    # Drop the 'origin_zone' and 'destination_zone' columns
    trips_abroad = trips_abroad.drop(columns=["origin_zone", "destination_zone"])

    return trips_abroad


def format_trips_abroad(trips_abroad: pd.DataFrame, mcc_to_country: dict) -> pd.DataFrame:
    """This is the main function for formatting the trips from or to abroad.
    
    Args:
        trips_abroad: The original dataframe.
        mcc_to_country: A dictionary containing the equivalence between the
                        MCC code and the country.
    
    Returns:
        The modified trips_abroad dataframe (a copy of the original).
    """
    # Create a copy of the DataFrame to avoid modifying the original
    trips_abroad_copy = trips_abroad.copy()

    # Remove information about the zones in France and Portugal since this is not needed
    trips_abroad_copy.loc[:, "destination_zone"] = remove_numbers_fr_pt(trips_abroad_copy["destination_zone"])
    trips_abroad_copy.loc[:, "origin_zone"] = remove_numbers_fr_pt(trips_abroad_copy["origin_zone"])

    # Change the three-digit number to the name of the country
    trips_abroad_copy.loc[:, "destination_zone"] = trips_abroad_copy["destination_zone"].apply(
        lambda x: map_mcc_to_country(x, mcc_to_country)
    )
    trips_abroad_copy.loc[:, "origin_zone"] = trips_abroad_copy["origin_zone"].apply(
        lambda x: map_mcc_to_country(x, mcc_to_country)
    )

    # Change "abroad" to the exact origin and destination
    trips_abroad_copy = precise_origin_destination(trips_abroad_copy)

    # Drop the column containing the weird stations
    trips_abroad_copy = trips_abroad_copy.drop(["weird_stations"], axis=1)

    # Sum trips that have the same characteristics
    cols = list(trips_abroad_copy.columns)
    cols.remove("trips")

    # Remove ground entries
    trips_abroad_copy = trips_abroad_copy[
        (trips_abroad_copy["entry_point"].str.startswith("airport", na=False)) |
        (trips_abroad_copy["exit_point"].str.startswith("airport", na=False))
    ]
    trips_abroad_final = trips_abroad_copy.groupby(cols, as_index=False, dropna=False)['trips'].sum()

    return trips_abroad_final


def format_trips_national(trips_national: pd.DataFrame): 
    """This function formats the national trips to match them to the 
    international ones
    
    Args:
        trips_national dataframe
    
    Returns:
        trips_national dataframe
        """
    # drops the column containing the weird stations and origin/destination zone
    trips_national=trips_national.drop(["weird_stations"],axis=1)
    trips_national=trips_national.drop(["origin_zone"],axis=1)
    trips_national=trips_national.drop(["destination_zone"], axis=1)

    # sums trips that have the same characteristics
    cols=list(trips_national.columns)
    cols.remove("trips")
    trips_national=trips_national.groupby(cols,as_index=False, dropna=False)['trips'].sum()    

    return trips_national


def rescale_trips(trips_abroad:pd.DataFrame,coeffs_incoming: pd.DataFrame,coeffs_outgoing: pd.DataFrame):
    """This function rescales the international trips with the coefficients
    provided.
    
    Args:
        trips_abroad dataframe
        incoming_coefficients dataframe
        outgoing coefficients dataframe

    Returns:
        trips_abroad dataframe
    """
    #creata a copy of the original dataframe
    trips_abroad_final=trips_abroad.copy()
    
    #initialisation columns
    cols=[f"archetype_{n}" for n in range(6)]
    cols.append("trips")

    #set the origin and destination rows as indices in the coefficients
    coeffs_incoming_right_index=coeffs_incoming.set_index("origin")
    coeffs_outgoing_right_index=coeffs_outgoing.set_index("destination")

    #main iteration
    for idx,row in trips_abroad.iterrows():
        origin=row["origin"]
        destination=row["destination"]

        #selects origin that match the origin in the coefficients
        if origin in set(coeffs_incoming["origin"]):
            for col in cols:
                trips_abroad_final.at[idx,col]=row[col]*coeffs_incoming_right_index["real_vs_predicted_coeff"].loc[origin]

        #does the same with destination
        if destination in set(coeffs_outgoing["destination"]):
            for col in cols:
                trips_abroad_final.at[idx,col]=row[col]*coeffs_outgoing_right_index["real_vs_predicted_coeff"].loc[destination]

    return trips_abroad_final


def format_airports(airport:str,iata_to_icao:dict):
    """This function changes the entry of exit_point or entry_point of a dataframe
    from airport_XXX where XX is the IATA code, to the ICAO code
        
    Inputs:
        airport string 
        iata_to_icao dictionary
    
    Outputs:
        modified string
    """
    if pd.notna(airport):
        components=airport.split("_")
        airport=iata_to_icao[components[1]]
    return airport


def map_and_aggregate_flight_capacity(flight_schedules: pd.DataFrame,
                                      airport_to_country: dict,
                                      selected_countries=None
                                      ):
    """Aggregates the flight capacity per origin and destination airports and filters
    by selected countries (if provided)
    
    Args:
        flight_schedules: dataframe that contains information about the flight schedules on a given day
        airport_to_country: dictionary with airports and country acronym
        selected_countries: set or list of selected countries

    Returns: 
        flight_capacity: dataframe with the information
    """
    # sums the seats per airports in flight_schedules
    flights_capacity=flight_schedules.groupby(["origin","destination"])["seats"].sum().reset_index()

    # adds country of origin and destination
    flights_capacity["country_origin"]=flights_capacity["origin"].map(airport_to_country)
    flights_capacity["country_destination"]=flights_capacity["destination"].map(airport_to_country)

    # selects only the selected countries
    if selected_countries is not None:
        flights_capacity=flights_capacity[
            (flights_capacity["country_origin"].isin(selected_countries))|
            (flights_capacity["country_destination"].isin(selected_countries))]
        
    return flights_capacity


def aggregate_seats_by_airport_and_country(flights_capacity):
    """Aggregates seat counts by airport and country, categorising flights based on whether the destination is Spain ("ES").
    
    Args: 
        flights_capacity: dataframe with information about the airports of origin and destination 
        and their country

    Returns:
        a tuple containing two dictionaries, seats_from Aggregated seats for flights with destination country "ES"
        and seat_to Aggregated seats for flights with origin country not "ES"
    """
    # initialise
    seats_from={}
    seats_to={}

    for _,row in flights_capacity.iterrows():
        # check whether it is an incoming or an outgoing trip
        if row["country_destination"]=="ES":

            # select the airport and country
            airport=row["destination"]
            country=row["country_origin"]

            # assign seats per country
            if airport not in seats_from.keys():
                seats_from[airport]={country:row["seats"]}
            else:
                dict=seats_from[airport]
                if country not in dict.keys():
                    dict[country]=row["seats"]
                else:
                    dict[country]+=row["seats"]
        else:
            
            # select the airport and the country
            airport=row["origin"]
            country=row["country_destination"]

            # assign seats per country
            if airport not in seats_to.keys():
                seats_to[airport]={country:row["seats"]}
            else:
                dict=seats_to[airport]
                if country not in dict.keys():
                    dict[country]=row["seats"]
                else:
                    dict[country]+=row["seats"]

    return seats_from, seats_to


def assign_airport_incoming(trips_incoming: pd.DataFrame, 
                            flights_capacity: pd.DataFrame, 
                            seats_from: dict):
    """Recallibrates the number of trips according to the information of the airports
    
    Args:
        trips_incoming dataframe: dataframe with information about the trips coming from abroad to spain
        flight_capacity: dataframe with information about the seat capacity from spanish airport to airport in origin
        seats_from: obtained from flight_capacity using the aggregate_seats_by_airport_and_country function

    Returns: 
        trips_final: the modified dataframe
    """
    
    #merge and format dataframe
    trips_final=pd.merge(left=trips_incoming, left_on=["origin","entry_point"],right=flights_capacity,right_on=["country_origin","destination"],how="left")
    trips_final=trips_final.rename({"origin_x":"origin","destination_x":"destination"},axis=1)
    trips_final["origin"]=trips_final["origin_y"].fillna(trips_final["origin"])
    trips_final=trips_final.drop(["origin_y"],axis=1)
    trips_final["proportionality_coeff"]=float(1)

    # find the proportionality_coeff
    for idx,row in trips_final.iterrows():
        if row["entry_point"] in seats_from:
            if row["origin_name"] in seats_from[row["entry_point"]]:
                coeff=row["seats"]/seats_from[row["entry_point"]][row["origin_name"]]

                trips_final.at[idx, "proportionality_coeff"]=coeff

    # apply the proportionality_coeff to the relevant columns
    cols=[f"archetype_{n}" for n in range(6)]
    cols.append("trips")
    for col in cols:
        trips_final[col]=trips_final[col]*trips_final["proportionality_coeff"]

    #remove columns that we don't need
    trips_final=trips_final.drop(["country_origin","country_destination","destination_y"],axis=1)

    #drop trips that were not assigned (note that then the sum of the trips will not match)
    trips_final=trips_final[trips_final["seats"].notna()]

    return trips_final

def assign_airport_outgoing(trips_outgoing: pd.DataFrame, 
                            flights_capacity: pd.DataFrame, 
                            seats_to: dict):
    """Recallibrates the number of trips according to the information of the airports
    
    Args:
        trips_outgoing dataframe: dataframe with information about the trips going abroad from spain
        flight_capacity: dataframe with information about the seat capacity from spanish airport to airport in destination
        seats_to: obtained from flight_capacity using the aggregate_seats_by_airport_and_country function

    Returns: 
        trips_final: the modified dataframe
    """
    
    #merge and format dataframe
    trips_final=pd.merge(left=trips_outgoing, left_on=["destination","exit_point"],right=flights_capacity,right_on=["country_destination","origin"],how="left")
    trips_final=trips_final.rename({"origin_x":"origin","destination_x":"destination"},axis=1)
    trips_final["destination"]=trips_final["destination_y"].fillna(trips_final["origin"])
    trips_final=trips_final.drop(["destination_y"],axis=1)
    trips_final["proportionality_coeff"]=float(1)

    # find the proportionality_coeff
    for idx,row in trips_final.iterrows():
        if row["exit_point"] in seats_to:
            if row["destination_name"] in seats_to[row["exit_point"]]:
                coeff=row["seats"]/seats_to[row["exit_point"]][row["destination_name"]]

                trips_final.at[idx, "proportionality_coeff"]=coeff

    # apply the proportionality_coeff to the relevant columns
    cols=[f"archetype_{n}" for n in range(6)]
    cols.append("trips")
    for col in cols:
        trips_final[col]=trips_final[col]*trips_final["proportionality_coeff"]

    #remove columns that we don't need
    trips_final=trips_final.drop(["country_origin","country_destination","origin_y"],axis=1)

    #drop trips that were not assigned (note that then the sum of the trips will not match)
    trips_final=trips_final[trips_final["seats"].notna()]

    return trips_final  

def trips_format_to_pipeline(trips):
    """
    Processes the trips DataFrame to create a combined DataFrame with archetype-specific trip counts.

    Args:
        trips (pd.DataFrame): Input DataFrame containing trip data with archetype columns.

    Returns:
        pd.DataFrame: Combined DataFrame with columns: date, origin, destination, archetype, trips.
    """
    # Create a dictionary to store the DataFrames
    dataframes = {}

    # Generate the DataFrames dynamically
    for i in range(6):
        df_name = f"df_{i}"
        archetype_col = f"archetype_{i}"
        dataframes[df_name] = trips.groupby(["date", "origin", "destination"], as_index=False).agg({archetype_col: "sum"})

    # Create a list of tuples containing the DataFrames and their corresponding archetype columns
    dataframe_list = [
        (dataframes[f"df_{i}"], f"archetype_{i}")
        for i in range(6)
    ]

    # Combine the DataFrames with a new "archetype" column
    combined_df = pd.concat(
        [
            df.assign(archetype=archetype_col)  # Add new archetype column
            .rename(columns={archetype_col: 'trips'})  # Rename archetype column value to "trips"
            for df, archetype_col in dataframe_list
        ],
        ignore_index=True
    )

    # Reorder columns
    columns = ["date", "origin", "destination", "archetype", "trips"]
    combined_df = combined_df[columns]

    # Sort the combined DataFrame
    combined_df = combined_df.sort_values(by=["origin", "destination", "archetype"])

    return combined_df