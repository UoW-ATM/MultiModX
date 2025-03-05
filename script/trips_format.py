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