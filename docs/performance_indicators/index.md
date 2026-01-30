# Performance Indicators Computation


## Strategic indicators


## Pre-tactical (replanned operations) indicators



## Tactical indicators

Configuration is in [mmx_kpis.toml](../examples/toml_examples.md): Paths to data, where to save the results, which indicators to compute

-ex: name of the folder with processed data  (after running strategic pipeline)

-c: if we want to compare 2 experiments (need to have computed the indicators first for each experiment individually)

-ppv: post-processing version (default 0): defines the number that is in file names, e.g. possible_itineraries_1.csv

The results are saved into a specified folder 'indicators' (path in .toml) as csv or plots.
	# Examples of usage
 
	python3 mmx_kpis.py -ex processed_cs10.pp00.so00_c1
 
	python3 mmx_kpis.py -c processed_cs10.pp00.so00_c1 processed_cs10.pp10.so00_c1
 
	python3 mmx_kpis.py -c processed_cs10.pp00.so00_c2 processed_c1_replan -ppv 0 1

Scripts available:
- **Strategic PIs**: [mmx_kpis.py](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/mmx_kpis.py) with [mmx_kpis.toml](../examples/toml_examples.md)
- **Pre-Tactical PIs**: From replanning of operations with [mmx_replanning_pis.py](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/mmx_replanning_pis.py).
See main function in code.
- **Tactical PIs**: From Mercury [mmx_kpis.py](https://github.com/UoW-ATM/MultiModX/blob/main/performance_indicators/mmx_kpis.py) with [mmx_kpis_tactical.toml](../examples/toml_examples.md).




## Replanned
This document describes the **Key Performance Indicator (KPI)** computation applied to the outputs of the pre-tactical passenger replanning pipeline.

KPI routines are implemented in:

`performance_indicators/kpi_lib_replanned.py`

---

## Purpose

The KPI module evaluates the **system-level and passenger-level impacts** of disruptions and replanning decisions.

It enables:
- Comparison of alternative replanning strategies
- Quantification of passenger inconvenience
- Assessment of operational robustness

---

## Inputs

KPIs are computed using outputs from the replanning pipeline, including:

- Reassigned passenger itineraries
- Final replanned schedules (air and rail)
- Passenger delays and missed connections
- Service capacity utilisation

---

## Key Indicators

Typical KPIs include:

### Passenger-Centric

- Total travel time increase
- Delay distributions
- Missed connections
- Denied boarding events
- Number of stranded passengers

### Service-Centric

- Load factor variation
- Capacity utilisation
- Service saturation levels

### System-Level

- Mode share changes (air / rail / multimodal)
- Reaccommodated vs unmet demand
- Network resilience metrics

---

## Execution Timing

KPI computation is performed **after**:

1. Passenger reassignment is completed
2. Final itineraries and service loads are available

KPIs depend on the **final state** of the replanned network and passenger assignments.

---

## Usage

Typical workflow:

1. Run the pre-tactical replanning pipeline
2. Collect final passenger and service CSV outputs
3. Invoke KPI routines from `kpi_lib_replanned.py`
4. Analyse KPIs for scenario comparison and reporting

---

## Intended Use

The KPI module supports:
- Research analysis
- Scenario benchmarking
- Policy and operational evaluation
- Input generation for higher-level decision support tools
