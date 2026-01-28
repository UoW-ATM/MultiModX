# Performance Indicators Computation

Dedicated scripts are provided to compute performance indicators from the outcome of the different pipelines 
(strategic, pretactical and tactical).

The indicators focus mainly on passenger-centric mobility aspects as defined in the 
[Digital Catalogue of Indicators](https://nommon.atlassian.net/wiki/external/MzA2ZTJmMjU5MDUyNDNlYzlkNDBmNTMwOTRlMDY4MGY)
developed as part of the **Multimodal and Passengers Experience Performance Framework**.

The main script to estimate the performance indicators is the [`mmx_kpis.py`](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/mmx_kpis.py).
This with different parameters in the configuration TOML file can compute Strategic (planned network), Replanned and 
Tactical. Ad-hoc scripts are also provided for particular computations for the pre-tactical and tactical evaluation.




## Strategic indicators

The Strategic Indicators are computed using the [`mmx_kpis.py`](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/mmx_kpis.py)  script.

??? info "Script `mmx_kpis.py` parameters"
	The script accepts some parameters:

	- **-tf**: path to configuration TOML file
	- **-ex**: name of the folder with processed data  (after running strategic pipeline)
	- **-c**: if we want to compare 2 experiments (need to have computed the indicators first for each experiment individually)
	- **-ppv**: post-processing version (default 0): defines the number that is in file names, e.g. possible_itineraries_1.csv
  	- **-sf**: sufix to be added to the figures when generating them.


Examples of usage
```bash
python3 mmx_kpis.py -ex processed_cs10.pp00.so00_c1
python3 mmx_kpis.py -c processed_cs10.pp00.so00_c1 processed_cs10.pp10.so00_c1
python3 mmx_kpis.py -c processed_cs10.pp00.so00_c2 processed_c1_replan -ppv 0 1
```

Which indicators to compute and where to store them (as csv or plots) is controlled by a dedicated
TOML file (`mmx_kpis.toml`), described here:


??? info "MMX PIs TOML description for strategic and pre-tactical evaluator (`mmx_kpis.toml`)"
    ```toml
    {{ read_file("examples/toml/mmx_kpis.toml") | indent(4) }}
    ```

The `mmx_kpis.py` script computes the indicators that are defined in the TOML considering any variant (e.g. sum, avg) and
possible filtering (e.g. NUTS, airports), stores the results in a dictionary and then save these in CSV and/or plots (again)
as indicated in the TOML.



## Pre-tactical (replanned operations) indicators

For the Pre-tactical pipeline, the result PIs are computed using the same script and TOML as for the 
[Strategic indicators](#strategic-indicators) case. Therefore, see that section for more information.

PI computation is performed **after**:

1. Passenger reassignment is completed
2. Final itineraries and service loads are available

PIs depend on the **final state** of the replanned network and passenger assignments.



### MMX Replanning PIs

Besides the network indicators, the MultiModX repository provides an ad-hoc script `mmx_replanning_pis.py` which computes
indicators to describe the replanning of the network. Please, refer to that script to have more information. This script
does not have a dedicated TOML configuration file, but it is controlled by using parameters when calling the
script.


??? info "Script `mmx_replanning_pis.py` parameters"

	- **-pr**: Folder with the replanned results
	- **-v**: Version of the case study
	- **-cs**: Which case study to analyse
  	- **-nd**: Which network definition parameters to do
  	- **-pp**: Which policy package to do
	- **-so**: Which scheduler optimiser to do
	- **-dp**: Which disruption package to do
	- **-dm**: Which disruption management to do
	- **-pa**: Which version of pax assigment to do
	- **-pamin**: If provided instead of computing indicators compute aggregation across PAs
	- **-pamax**: If provided instead of computing indicators compute aggregation across PAs

### Purpose

The PI module evaluates the **system-level and passenger-level impacts** of disruptions and replanning decisions.

It enables:
- Comparison of alternative replanning strategies
- Quantification of passenger inconvenience
- Assessment of operational robustness


### Usage

Typical workflow:

1. Run the pre-tactical replanning pipeline (see [Pre-Tactical Passenger Replanning Pipeline](../pre-tactical/index.md))
2. Collect final passenger and service CSV outputs
3. Invoke PI routines from `kpi_lib_replanned.py`
4. Analyse PIs for scenario comparison and reporting

### Intended Use

The PI module supports:
- Research analysis
- Scenario benchmarking
- Policy and operational evaluation
- Input generation for higher-level decision support tools


### Inputs

KPIs are computed using outputs from the replanning pipeline, including:

- Reassigned passenger itineraries
- Final replanned schedules (air and rail)
- Passenger delays and missed connections
- Service capacity utilisation

### Key Indicators

??? info "Typical PIs computed on replanned network"

	- **Passenger-Centric**
		- Total travel time increase
	  	- Delay distributions
	  	- Missed connections
	  	- Denied boarding events
	  	- Number of stranded passengers
	
	  - **Service-Centric**
		- Load factor variation
		- Capacity utilisation
		- Service saturation levels
	
	  - **System-Level**
		- Mode share changes (air / rail / multimodal)
		- Reaccommodated vs unmet demand
		- Network resilience metrics




## Tactical indicators

In addition to the `mmx_kpis.py` script, a [`postprocessing_tactical.py`](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/postprocessing_tactical.py)
script is also provided. This script post-process the execution of the tactical evaluator to add the modelling of the 
passenger itineraries which are not directly supported by the Tactical Evaluator. This script is configured with a 
dedicated TOML file.

??? info "Tactical post-processing TOML description (`postprocessing.toml`)"
    ```toml
    {{ read_file("examples/toml/postprocessing.toml") | indent(4) }}
    ```


Finally, as mentioned, the [`mmx_kpis.py`](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/mmx_kpis.py) 
can compute the performance indicators related to the tactical execution of the mobility network. The input required is
the TOML configuration file defining the indicators for the tactical execution. Note that the definition of the input
also requires a few more parameters to characterise the tactical execution.

??? info "MMX PIs TOML description for tactical evaluator (`mmx_kpis_tactical.toml`)"
    ```toml
    {{ read_file("examples/toml/mmx_kpis_tactical.toml") | indent(4) }}
    ```




