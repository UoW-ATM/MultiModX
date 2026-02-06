# Pre-Tactical Passenger Replanning Pipeline

The **pre-tactical passenger replanning** builds on the capabilities of the [Strategic Pipeline](../strategic/index.md) 
to evaluate the impact of replanning services (retiming (e.g. delaying), cancelling and adding flight and/or rail services)
on passenger itineraries. These replanned operations are as reaction to disruptions in the system ⚡.

See additional documentation for information on:

- [Input Files](../examples/input_format.md#4-pretactical-replanning-pipeline-inputs)
- [TOML examples](../examples/toml_examples.md#4-pretactical-pipeline)
- [Performance indicators computation](../performance_indicators/index.md#pre-tactical-replanned-operations-indicators)


## 1. How to Use

### 1. Define a TOML configuration

Specify:

- Planned network paths
- Replanned scenario folder with the replanning actions (cancellation, addition, temporal modification of air ️✈️ and rail 🚆 services)
- Output folder structure
- Replanning constraints, to control which itineraries are valid as alternatives, and optimisation objectives, to select how to assign passengers to alternatives.

See [TOML](../examples/toml_examples.md#4-pretactical-pipeline) for examples and description.

### 2. Run the replanning script

```bash
python replanning_passengers_script.py -tf your_config.toml -np 100 -ni 50 -mc 1 -hpc -ppv 0
```

---

## 2. Intended Audience

This documentation targets:

### Researchers
- Study disruption management strategies
- Analyse passenger reallocation mechanisms
- Compare policy or operational scenarios

### Scenario Analysts
- Run replanning experiments under different disruption assumptions
- Inspect stranded demand and reassignment performance
- Feed results into downstream evaluation tools

### Developers
- Maintain or extend the pre-tactical passenger management code
- Modify constraints, objectives, or solver configurations

---


## 3. Key Design Characteristics

- **Modular**  
  Each processing stage produces explicit outputs for traceability.

- **Configuration-driven**  
  Behaviour is controlled via TOML configuration files, not code changes.

- **Clear data lineage**  
  Every major pipeline milestone corresponds to one or more output CSV files.

- **Constraint-aware**  
  Alternative itineraries respect operational constraints such as:
  - Alliances
  - Maximum connections
  - Capacity limits

- **Optimisation-based**  
  Passenger reassignment leverages configurable optimisation solvers.

---


## 4. High-level pipeline flow


The pipeline identifies passengers affected by the replanning and **reassign** otherwise stranded passengers in the
replanned network. This reassigment considers services capacities and reassigment policies (e.g. maintain operators,
enable mutimodality).

The pipeline reads planned network outputs from the Strategic Pipeline and applies replanning actions (cancellations ❌, 
additions ️✈️, modified schedules 🚆). It then identifies the impacted passengers and generates **reassigned itineraries** and
related performance indicators 📈).

This flow is implemented in:  
[`script/pre-tactical/replanning_passengers_script.py`](https://github.com/UoW-ATM/MultiModX/blob/main/script/pre-tactical/replanning_passengers_script.py) in the repository.


Under disruption scenarios (e.g., cancelled flights or trains, significant delays), the **planned multimodal transport network** may no longer deliver feasible itineraries for all passengers.
The pipeline identifies the status of passengers and, for those stranded, finds possible alternative itineraries and reallocate them.



### Inputs

#### **Configuration**

- `toml_config` — Replanning and network paths defined in a TOML file (experiment paths, replanned actions folder, 
planned network info) (see [TOML](../examples/toml_examples.md#4-pretactical-pipeline) section). The configuration file 
includes, crucially, the flexibility to be used to find suitable alternatives for stranded passengers.

#### **Planned Network Inputs**
These are taken from outputs of the strategic pipeline (considering the **originally planned** network):

- Passenger assignments to itineraries prior to disruption  
- Flight schedules (processed)  
- Rail timetables (processed GTFS)  
- Rail GTFS stops and stops considered  
- Other network parameters, such as:
  - Processing time of passengers at infrastructure nodes
  - Minimum connecting times intra-mode
  - Regions access/egress
  - Multimodal connectivity between layers with minimum connecting times (MCTs)

#### **Replanned Network Inputs**
These reflect modifications applied due to disruption:

- Cancelled flights/trains  
- Modified (replanned) flight schedules  
- Modified (replanned) rail timetables  
- Additional flights/trains added post-disruption

See [Input Files](../examples/input_format.md#4-pretactical-replanning-pipeline-inputs) section for examples.



### High-Level Pipeline Steps


The pre-tactical replanning pipeline:

1. **Load configuration**: Reads the [TOML](../examples/toml_examples.md#4-pretactical-pipeline) configuration file and the replanning operations from the `replanned_actions` folder.
2. **Applies the replanning to the network**: Read the _planned network_ and modifies its supply to reflect replanning actions (cancellations, replanned, added services) .
3. **Identifies how passengers are impacted**: Classify passengers between not affected, affected but that can keep their itinerary (e.g., delayed, connections can be kept), and passengers who would be stranded (i.e., cancelled services and missed connections) (kept or delayed but connected).
4. **Find alternatives for stranded passengers**: Recompute capacities available in services (removing stranded passengers), and using the capabilities of the [Strategic Evaluator](../strategic/index.md) compute possible itineraries for the stranded passengers. Finally, filtering alternatives to keep the ones that respect the defined operational constraints (e.g. keep (or not) operator, keep (or not) origin and/or destination infrastructure nodes (airports, rail stations)) 
5. **Assigns passengers to new itineraries**: Assign stranded passengers into alternatives under capacity and constraints, based on lexicographic optimisation defined in TOML configuration file (e.g. minimise stranded passengers, arrive as close as originally planned).
6. **Save results with passenger status**: Save passenger status along with performance summaries.




### Mermaid Operational Flow Diagram

```mermaid
flowchart TD

 classDef withmargines fill-opacity:0.0,color:#FFFFFF,stroke-width:0px;


%% ================= LOAD CONFIGURATION AND INPUT =================
subgraph STAGE1["**① Load configuration**"]
    foo1["<p style='width:0px;height:0px;margin:0'>/p>"]:::withmargines

    A[Load Config & TOML]
    B[Setup Output Folders]
    C[Read Planned Network Data]
    D[Read Replanned Actions]
    
    foo1 ~~~ A --> B
    B --> C
    C --> D
end

%% ================= APPLY REPLANNIG =================
subgraph STAGE2["**② Apply replanning to network**"]
    
    foo2["<p style='width:0px;height:0px;margin:0'>/p>"]:::withmargines
    E[Preprocess Replanned Flight & Rail]
    F[Adjust Planned]

    foo2 ~~~ E
    D --> E
    E --> F

end

F--> CSV_PROC[(processed)]


%% ================= IDENTIFY PAX IMPACTED =================
subgraph STAGE3["**③ Identify passengers impacted**"]
    foo3["<p style='width:0px;height:0px;margin:0'>/p>"]:::withmargines
    
    G[Identify Affected Passengers]
    foo3 ~~~ G
    F --> G
    
end

G--> CSV_RES0[(0.pax_assigned_to_itineraries_options_status_replanned_#.csv)]
G--> CSV_RES1[(1.pax_assigned_to_itineraries_options_kept_#.csv)]
CSV_RES1--> CSV_PAX_UNNAF[(1.1.pax_assigned_kept_unnafected_#.csv)]
CSV_RES1--> CSV_PAX_AFF_OK1.2.pax_assigned_kept_affected_delayed_or_connections_kept_[(1.2.pax_assigned_kept_affected_delayed_or_connections_kept_#.csv)]
G--> CSV_RES2[(2.pax_assigned_need_replanning_#.csv)]

%% ================= FIND ALTERNATIVES =================
subgraph STAGE4["**④ Find alternatives for stranded passengers**"]
    foo4["<p style='width:0px;height:0px;margin:0'>/p>"]:::withmargines

    H[Compute Service Capacities]
    I[Generate OD Demand for Reassignment]
    J[Compute Alternative Itineraries]
    K[Filter Under Operational Constraints]
    
    foo4 ~~~ H
    G --> H
    H --> I
    I --> J
    J --> K
end

H--> CSV_CAP[(services_w_capacity_#.csv)]
H--> CSV_WO_CAP[(services_wo_capacity_#.csv)]
CSV_CAP-->CSV_CAP_FIN[(services_capacities_#.csv)]
CSV_WO_CAP-->CSV_CAP_FIN

I--> CSV_OD[(demand_missing_reaccomodate_#.cvs)]
J--> CSV_ALT[(pax_need_replanning_w_it_options_#.csv)]
K--> CSV_ALT_CONST[(pax_need_replanning_w_it_options_filtered_w_constraints_#.csv)]
K--> CSV_STRAND[(pax_stranded_#.csv)]

L --> CSV_SOLV[(pax_reassigned_results_solver_#.csv)]
L --> CSV_DEM_ASS[(pax_demand_assigned_#.csv)]


%% ================= ASSIGN PASSENGERS =================
subgraph STAGE5["**⑤ Assign passengers to alternatives**"]
    foo5["<p style='width:0px;height:0px;margin:0'>/p>"]:::withmargines

    L[Assign Passengers to New Services]

    foo5 ~~~ L
    K --> L
end


%% ================= SAVE RESULTS =================
subgraph STAGE6["**⑥ Save results of passenger status**"]
    foo6["<p style='width:0px;height:0px;margin:0'>/p>"]:::withmargines

    M[Save Outputs & Summaries]
    
    foo6 ~~~ M
    L --> M
end

M --> CSV_RES3[(3.pax_reassigned_to_itineraries_#.csv)]
CSV_RES3--> CSV_RES31[(3.1.pax_affected_all_#.csv)]
M --> CSV_RES4[(4.pax_assigned_to_itineraries_replanned_stranded_#.csv)]
M --> CSV_RES5[(5.pax_demand_assigned_summary_#.csv)]



```

### Pipeline Outputs
Not exhaustive list of outputs

| Folder                                                                        | Output file                                                        | Description                                                                                                                               |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| pax_replanned                                                                 | `0.pax_assigned_to_itineraries_options_status_replanned_#.csv`     | Passenger itinerary status after replanning                                                                                               |
| pax_replanned                                                                 | `1.pax_assigned_to_itineraries_options_kept_#.csv`                 | Passengers whose itineraries remain valid (could be delayed)                                                                              |
| pax_replanned                                                                 | `1.1.pax_assigned_kept_unnafected_#.csv`                           | Unaffected passengers (subset from 1, i.e., not delayed)                                                                                  |
| pax_replanned                                                                 | `1.2.pax_assigned_kept_affected_delayed_or_connections_kept_#.csv` | Kept but impacted passengers (subset from 1)                                                                                              |
| pax_replanned                                                                 | `2.pax_assigned_need_replanning_#.csv`                             | Passengers needing reassignment (otherwise stranded)                                                                                      |
| pax_replanned                                                                 | `3.pax_reassigned_to_itineraries_#.csv`                            | Reassigned passengers to new itineraries                                                                                                  |
| pax_replanned                                                                 | `3.1.pax_affected_all_#.csv`                                       | All affected passengers (reassigned and delayed)                                                                                          |
| pax_replanned                                                                 | `4.pax_assigned_to_itineraries_replanned_stranded_#.csv`           | All passengers needed to be replanned but ended up stranded                                                                               |
| pax_replanned                                                                 | `5.pax_demand_assigned_summary_#.csv`                              | Summary of assigned/unfulfilled demand                                                                                                    |
| paths_itineraries | `demand_missing_reaccomodate_#.csv`                                | OD demand needing reassignment (comptued from 2)                                                                                          |
| paths_itineraries | `services_capacities_#.csv`                                        | Services seats capacities                                                                                                                 |
| paths_itineraries | `services_capacities_available.csv`                                | Available seats per service                                                                                                               |
| paths_itineraries | `services_w_capacity_#.csv` / `services_wo_capacity_#.csv`         | Services with/without available capacity                                                                                                  |
| paths_itineraries | `potential_paths_#.csv`                                            | Potential paths for stranded passengers itineraries                                                                                       |
| paths_itineraries | `possible_itineraries_#.csv`                                       | Possible itineratires for stranded passengers                                                                                             |
| paths_itineraries | `pax_need_replanning_w_it_options_#.csv`                           | Passengers stranded that have options (itineraries)                                                                                       |
| paths_itineraries | `pax_need_replanning_w_it_options_filtered_w_constraints_#.csv`    | Passengers stranded that have options (itineraries) once operational constraints are considered (e.g. keep operator, allow multimodality) |
| paths_itineraries | `pax_stranded_#.csv`                                               | Passengers stranded                                                                                                                       |
| paths_itineraries | `pax_demand_assigned_#.csv`                                        | Demand volumes assigned per new itinerary                                                                                                 |
| paths_itineraries | `pax_reassigned_results_solver_#.csv`                              | Outcome of assigment solver (which considers lexicographic optimisation to select itinerary from options available)                       |
| processed | `flight_schedules_proc_#.csv`, `rail_timetable_proc_#.csv`, etc. | Replanned network (schedules, timetables once the replannign has been performed)                                                          |



### KPI Computation After Replanning

The file  
[`performance_indicators/kpi_lib_replanned.py`](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/kpi_lib_replanned.py) in the repository  
provides routines to compute **Key Performance Indicators (KPIs)** from the outcome of the **passenger reassignment process** after disruption. See the [Performance Indicators](../performance_indicators/index.md#replanned) documentation for more detail.

These KPIs are derived from the **replanned passenger itineraries**, **service schedules**, and **assignment results**, and are intended to quantify the impacts of disruption and replanning strategies.

Typical indicators include:

- **Total travel time changes**
- **Missed connections and stranded passengers**
- **Load factor variations**
- **Mode share impacts** (air / rail / multimodal)

KPI computation is typically performed **after** the replanning assignment has completed and relies on the final reassigned schedules and passenger data.

