# Input Files Format

This document describes the **expected input files** for the MultiModX Pipelines.  
It is split into five sections:

1. [Introduction on files](#1-introduction-on-files)  
2. [Strategic Pipeline Inputs](#2-strategic-pipeline-inputs)  
3. [Heuristics Computation Inputs](#3-heuristics-computation-inputs)
4. [Pre-Tactical (Replanning) Pipeline Inputs](#4-pretactical-replanning-pipeline-inputs)
5. [Tactical Pipeline Inputs](#5-tactical-pipeline-inputs)
6. [Performance Indicators](#6-performance-indicators)

A **sample dataset** containing minimal valid inputs is available on Zenodo: [Download Sample Inputs](https://zenodo.org/your-dataset-link)  

---

## 1. Introduction on files

### Configuration files
All **configuration files** are stored in TOML format. See the [TOML](toml_examples.md) section of the documentation for
details on these.

### Input and output files

All **input and output data** used by MultiModX is in **CSV** format, with few exceptions (see [Special cases](#special-cases) below). 

The information is by default stored in the data folder and organised by case studies (e.g. CS10) with versions
inside (e.g. v=0.1). This is just a convention and can be changed by modifying the path of the experiment (`experiment_path`) in the relevant
TOML files (see example of [TOML file](toml_examples.md#1-strategic-pipeline)).

The name of the individual files required (and by changing their name one could change their path within the `exmperiment_path`) is also
defined within the TOML configuration files.


### Special cases

The **logit model sensitivity parameters** are stored as pickle files from the biogeme library. Calibration for three different mobility settings
are provided (see [TOML](toml_examples.md#1-strategic-pipeline) for an example and more information on this).

The Tactical Evaluator requires additional input files that are not provided by the MultiModX pipeline (e.g. ATFM delays,
minimum turnaround times, flight plans). See [Tactical Evaluator](../tactical/index.md#4-multimodx-scripts) for more information.


---

## 2. Strategic Pipeline Inputs


The strategic pipeline is configured primarily via TOML files:

- `strategic_pipeline.toml`: See example in [strategic_pipeline.toml](toml_examples.md#1-strategic-pipeline)  
- `policy_package.toml`: See example in [policy_package.toml](toml_examples.md# )

The input files are then processed to build:

- Harmonised air and rail networks  
- Multimodal layers  
- Demand assignment and itinerary computation  
- Tactical inputs  

The table below summarises the main groups of input files:

| Input Group | Purpose                                                                                                                                                              |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Demand & Logit | Passenger demand, archetypes, sensitivities                                                                                                                          |
| Network Data | Flight schedules, rail GTFS, MCTs, node locations                                                                                                                    |
| Infrastructure | Processing times, airport/rail stations, transitions, regions access                                                                                                 |
| Aircraft & Airlines | Tactical info for flights (types, capacities, codes). Needed to generate input for Tactical Evaluator (Mercury), if desired.                                         |
| Heuristics | Optional precomputed travel-time heuristics for path finding. See [Heuristics Computation Inputs](#heuristics-computation-inputs) for information on the heuristics. |


??? info "Strategic pipeline data files description"
    
    **Note:** The structure below follows the order defined in `strategic_pipeline.toml`.  
    For reference, see the [TOML examples](toml_examples.md).

    ### A. Demand Data
    
    **File:** `demand.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | date | Date of demand | 20220923 |
    | origin | Origin region or station | ES111 |
    | destination | Destination region or station | ES112 |
    | archetype | Demand archetype | archetype_0 |
    | trips | Number of trips | 521 |
    
    **Notes:** Demand is mapped to logit models via sensitivities pickle files.
    
    ---
    
    ### B. Sensitivities Logit
    
    **File(s):** `archetype_x.pickle` (from Biogeme library)
    
    - Each archetype corresponds to a separate `.pickle` file.  
      - Used for logit-based choice modelling.  
      - Provided for intra-Spain and international-Spain in [libs/logit_model](https://github.com/UoW-ATM/MultiModX/blob/main/libs/logit_model).
    
    ---
    
    ### C. Flight Schedules
    
    **File:** `flight_schedules.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | service_id | Flight ID | VY_2473 |
    | origin | Departure airport ICAO | GCRR |
    | destination | Arrival airport ICAO | LEBL |
    | dep_terminal | Departure terminal | 1 |
    | arr_terminal | Arrival terminal | 1 |
    | sobt | Scheduled off-block time | 2019-09-06 12:15:00 |
    | sibt | Scheduled in-block time | 2019-09-06 15:10:00 |
    | provider | Airline code | VY |
    | act_type | Aircraft type | 321 |
    | seats | Number of seats | 220 |
    | gcdistance | Great-circle distance (km) | 1971 |
    
    ---
    
    ### D. Alliances
    
    **File:** `alliances.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | provider | Airline code | IB |
    | alliance | Alliance name | OneWorld |
    
    ---
    
    ### E. Airports Coordinates
    
    **File:** `airports_coordinates.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | icao_id | ICAO airport code | AGGA |
    | lat | Latitude | -8.6983333333 |
    | lon | Longitude | 160.6783333333 |
    
    ---
    
    ### F. Minimum Connecting Times (Air)
    
    **File:** `mct_air.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | icao_id | ICAO airport code | BIKF |
    | standard | Standard MCT | 39 |
    | domestic | Domestic MCT | 20 |
    | international | International MCT | 60 |
    
    ---
    
    ### G. GTFS (Rail Timetables)
    
    **Folder:** `GTFS/`  
    
    - Standard GTFS format  
      - Used to generate rail network
    
    ---
    
    ### H. Rail Stations Considered
    
    **File:** `rail_stations_considered.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | stop_id | GTFS stop ID | 007102002 |
    
    ---
    
    ### I. Minimum Connecting Times (Rail)
    
    **File:** `mct_rail.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | stop_id | Rail station stop ID | 007105000 |
    | default_transfer_time | Minimum transfer time (minutes) | 6 |
    
    ---
    
    ### J. Airport & Rail Processing Times
    
    **Files:** `airport_processes.csv`, `rail_stations_processes.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | airport / station | Node ID | ACE / 007102002 |
    | pax_type | Passenger type | all |
    | k2g / k2p | Check-in to gate / station process | 90 / 15 |
    | g2k / p2k | Gate to check-in / station process | 30 / 10 |
    | k2g_multimodal / k2p_multimodal | Multimodal adjustment | 90 / 15 |
    | g2k_multimodal / p2k_multimodal | Multimodal adjustment | 30 / 10 |
    
    ---
    
    ### K IATA-ICAO Mapping
    
    **File:** `iata_icao_static.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | IATA | IATA code | AAC |
    | ICAO | ICAO code | HEAR |
    
    ---
    
    ### L. Air-Rail Transitions
    
    **File:** `air_rail_transitions.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | origin_station | Node origin | LEAL |
    | destination_station | Node destination | 7160911 |
    | layer_origin | Mode | air |
    | layer_destination | Mode | rail |
    | avg_travel_a_b | Travel time | 40 |
    | avg_travel_b_a | Travel time | 40 |
    
    ---
    
    ### M. Regions Access
    
    **File:** `regions_access.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | region | Demand region | ES111 |
    | station | Node ID | LEST |
    | layer | Mode | air |
    | pax_type | Passenger type | all |
    | avg_d2i | Avg access time (min) | 53 |
    | avg_i2d | Avg egress time (min) | 50 |
    
    ---
    
    ### N Heuristics (Optional)
    
    **Files:** `air_time_heuristics.csv`, `rail_time_heuristics.csv`
    
    | Column | Description | Example |
    |--------|------------|---------|
    | min_dist | Min distance | 0 |
    | max_dist | Max distance | 150 |
    | time | Heuristic time (minutes) | 25 |
    
    See [Heuristics Computation Inputs](#heuristics-computation-inputs) for more details.
    
    ---
    
    ### O. Aircraft & Airlines
    
    **Files:** `ac_type_icao_iata_conversion.csv`, `ac_mtow.csv`, `ac_wtc.csv`, `airline_ao_type.csv`, `airline_iata_icao.csv`  
    
    - Aircraft types, maximum take-off weights, wake turbulence categories, airline types, and code conversions.
    - Used for tactical inputs generation.
    

---

## 3. Heuristics Computation Inputs

The Heuristics are computed to support the path finding algorithm (A*). These can be provided or not, if not then the algorithm will be
performed as a uniform cost search (UCS).

This subset is sufficient to **compute travel-time heuristics** for the pathfinder. See 
[heuristics_computation.toml](toml_examples.md) for example of TOML
file configuring the [compute_air_rail_heuristics.py](https://github.com/UoW-ATM/MultiModX/blob/main/script/strategic/compute_air_rail_heuristics.py) script that is the one used to generate the heuristics.


??? info "Heuristics computation data files description"

        **Note:** For heuristics computation, only the files below are necessary. All other files are optional.
        
        
        ### A. Flight Schedules
        
        **File:** `flight_schedules.csv` (same structure as above)
        
        ### B. Airports Coordinates
        
        **File:** `airports_coordinates.csv` (same structure as above)
        
        ### C. GTFS (Rail Timetables)
        
        **Folder:** `GTFS/`  
        
        ### D. Rail Stations Considered
        
        **File:** `rail_stations_considered.csv`
        
        | Column | Description | Example |
        |--------|------------|---------|
        | stop_id | GTFS stop ID | 0077821 |
        
        ### E. Rail Stations Considered + NUTS
        
        **File:** `rail_stations_considered_nuts.csv`
        
        | Column | Description | Example |
        |--------|------------|---------|
        | NUTS_ID | NUTS region ID | ES61 |
        | LEVL_CODE | Level code | 2 |
        | NAME_LATN | Name | Andalucía |
        | num_rail_stations | Number of rail stations | 2 |
        | num_airports | Number of airports | 5 |
        | airports | List of airports in region | ['LEMG', 'LEJR', 'LEGR', 'LEAM', 'LEZL'] |
        | rail_stations | List of rail stations in region | [('ES50500', 'Estación de tren Cordoba'), ('ES51003', 'Estación de tren Sevilla-Santa Justa')] |


---

## 4. PreTactical Replanning Pipeline Inputs

**Input data formats** required by the **pre-tactical passenger replanning pipeline**.

Inputs consist of:
- Planned network outputs
- Replanned operational modifications
- Infrastructure and transfer constraints

All paths are provided via the TOML configuration file.


??? info "Pretactical replanning pipeline data files description"

    Notes
    
    - All inputs must be temporally consistent (time zones, date formats).
      - Identifiers (service IDs, stop IDs) must match across datasets.
      - Inputs are assumed to be **pre-validated** for structural correctness.
    
    
    ### A. Planned Passenger Assignments
    
    **Description:**  
    Passenger itineraries from the strategic or planned scenario.
    
    **Typical content:**
    - Passenger or demand identifier
      - Assigned itinerary
      - Service sequence
      - Travel times and costs
    
    **Source:**  
    Output of the strategic pipeline.
    
    
    ### B. Planned Flight Schedules
    
    **Description:**  
    Baseline flight schedules prior to disruption.
    
    **Format:** CSV
    
    Key fields include:
    - `service_id`
      - `origin`, `destination`
      - `sobt`, `sibt`
      - `provider`
      - `seats`
    
    
    ### C. Planned Rail Schedules (GTFS)
    
    **Description:**  
    Rail services in GTFS format.
    
    **Required files:**
    - `stops.txt`
      - `trips.txt`
      - `stop_times.txt`
      - `calendar.txt`
    
    Used to construct the planned rail network.
    
    
    ### D. Rail Stations Considered
    
    **File:** `rail_stations_considered.csv`
    
    | Column | Description |
    |------|-------------|
    | stop_id | GTFS stop identifier |
    
    Only the listed stations are retained in the replanning network.
    
    
    ### E. Replanned Actions – Flights
    
    **Description:**  
    Operational changes applied after disruption.
    
    Possible inputs:
    - Cancelled flights
      - Modified flight schedules
      - Additional flights
    
    These override or extend the planned flight network.
    
    ### F. Replanned Actions – Rail
    
    **Description:**  
    Rail timetable changes due to disruption.
    
    Possible inputs:
    - Cancelled rail services
      - Modified rail trips
      - Additional rail trips
    
    Applied on top of the planned GTFS network.
    
    
    ### G. Minimum Connecting Times (MCT)
    
    **Description:**  
    Minimum transfer times between services.
    
    **Includes:**
    - Air-to-air
      - Rail-to-rail
      - Air-to-rail transitions
    
    Used to validate itinerary feasibility during reassignment.
    
    
    ### H. Capacity Information
    
    **Description:**  
    Available capacity per service after replanning.
    
    Computed internally from:
    - Planned seat counts
      - Already assigned passengers
      - Cancelled services



---

## 5. Tactical Pipeline Inputs



---

## 6. Performance Indicators