# Strategic Pipeline Input File Formats

This document describes the **expected input files** for the MultiModX Strategic Pipeline.  
It is split into three sections:

1. [Introduction](#introduction)  
2. [Strategic Pipeline Inputs](#strategic-pipeline-inputs)  
3. [Heuristics Computation Inputs](#heuristics-computation-inputs)

A **sample dataset** containing minimal valid inputs is available on Zenodo: [Download Sample Inputs](https://zenodo.org/your-dataset-link)  

---

## Introduction

The strategic pipeline is configured primarily via TOML files:

- `strategic_pipeline.toml`: See example in [strategic_pipeline.toml](toml_examples/strategic_pipeline.toml)  
- `policy_package.toml`: See example in [policy_package.toml](toml_examples/policy_package.toml)

The input files are then processed to build:

- Harmonised air and rail networks  
- Multimodal layers  
- Demand assignment and itinerary computation  
- Tactical inputs  

The table below summarises the main groups of input files:

| Input Group | Purpose                                                                                                                      |
|------------|------------------------------------------------------------------------------------------------------------------------------|
| Demand & Logit | Passenger demand, archetypes, sensitivities                                                                                  |
| Network Data | Flight schedules, rail GTFS, MCTs, node locations                                                                            |
| Infrastructure | Processing times, airport/rail stations, transitions, regions access                                                         |
| Aircraft & Airlines | Tactical info for flights (types, capacities, codes). Needed to generate input for Tactical Evaluator (Mercury), if desired. |
| Heuristics | Optional precomputed travel-time heuristics for path finding                                                                 |


The Heuristics are computed to support the path finding algorithm (A*). These can be provided or not, if not then the algorithm will be
performed as a uniform cost search (UCS). The scrip [compute_air_rail_heuristics.py](/script/strategic/compute_air_rail_heuristics.py) 
can be used to generate these heuristcis. See [Heuristics Computation Inputs](#heuristics-computation-inputs) for info
on inputs needed.

---

## Strategic Pipeline Inputs

**Note:** The structure below follows the order defined in `strategic_pipeline.toml`.  
For reference, see the [TOML example](toml_examples/strategic_pipeline.toml).

### 1. Demand Data

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

### 2. Sensitivities Logit

**File(s):** `archetype_x.pickle` (from Biogeme library)

- Each archetype corresponds to a separate `.pickle` file.  
- Used for logit-based choice modelling.  
- Provided for intra-Spain and international-Spain in [libs/logit_model](/libs/logit_model).

---

### 3. Flight Schedules

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

### 4. Alliances

**File:** `alliances.csv`

| Column | Description | Example |
|--------|------------|---------|
| provider | Airline code | IB |
| alliance | Alliance name | OneWorld |

---

### 5. Airports Coordinates

**File:** `airports_coordinates.csv`

| Column | Description | Example |
|--------|------------|---------|
| icao_id | ICAO airport code | AGGA |
| lat | Latitude | -8.6983333333 |
| lon | Longitude | 160.6783333333 |

---

### 6. Minimum Connecting Times (Air)

**File:** `mct_air.csv`

| Column | Description | Example |
|--------|------------|---------|
| icao_id | ICAO airport code | BIKF |
| standard | Standard MCT | 39 |
| domestic | Domestic MCT | 20 |
| international | International MCT | 60 |

---

### 7. GTFS (Rail Timetables)

**Folder:** `GTFS/`  

- Standard GTFS format  
- Used to generate rail network

---

### 8. Rail Stations Considered

**File:** `rail_stations_considered.csv`

| Column | Description | Example |
|--------|------------|---------|
| stop_id | GTFS stop ID | 007102002 |

---

### 9. Minimum Connecting Times (Rail)

**File:** `mct_rail.csv`

| Column | Description | Example |
|--------|------------|---------|
| stop_id | Rail station stop ID | 007105000 |
| default_transfer_time | Minimum transfer time (minutes) | 6 |

---

### 10. Airport & Rail Processing Times

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

### 11. IATA-ICAO Mapping

**File:** `iata_icao_static.csv`

| Column | Description | Example |
|--------|------------|---------|
| IATA | IATA code | AAC |
| ICAO | ICAO code | HEAR |

---

### 12. Air-Rail Transitions

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

### 13. Regions Access

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

### 14. Heuristics (Optional)

**Files:** `air_time_heuristics.csv`, `rail_time_heuristics.csv`

| Column | Description | Example |
|--------|------------|---------|
| min_dist | Min distance | 0 |
| max_dist | Max distance | 150 |
| time | Heuristic time (minutes) | 25 |

---

### 15. Aircraft & Airlines

**Files:** `ac_type_icao_iata_conversion.csv`, `ac_mtow.csv`, `ac_wtc.csv`, `airline_ao_type.csv`, `airline_iata_icao.csv`  

- Aircraft types, maximum take-off weights, wake turbulence categories, airline types, and code conversions.  
- Used for tactical inputs generation.

---

## Heuristics Computation Inputs

This subset is sufficient to **compute travel-time heuristics** for the pathfinder. See 
[heuristics_computation.toml](toml_examples/heuristics_computation.toml) for example of TOML
file configuring the [compute_air_rail_heuristics.py](/script/strategic/compute_air_rail_heuristics.py) script.

### 1. Flight Schedules

**File:** `flight_schedules.csv` (same structure as above)

### 2. Airports Coordinates

**File:** `airports_coordinates.csv` (same structure as above)

### 3. GTFS (Rail Timetables)

**Folder:** `GTFS/`  

### 4. Rail Stations Considered

**File:** `rail_stations_considered.csv`

| Column | Description | Example |
|--------|------------|---------|
| stop_id | GTFS stop ID | 0077821 |

### 5. Rail Stations Considered + NUTS

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

**Note:** For heuristics computation, only these files are necessary. All other files are optional.
