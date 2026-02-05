# TOML configuration examples

All scripts in MultiModX are configured using [TOML](https://toml.io/en/) configuration files.

The data available on Zenodo (see [Input Files](input_format.md)) contains some examples, 
here a description of the different elements of the configuration  files is provided for:

1. [Strategic pipeline](#1-strategic-pipeline)
2. [Policy package definition](#2-policy-package)
3. [Heuristics computation](#3-heuristics-computation)
4. [Pre-tactical pipeline](#4-pretactical-pipeline)
5. For [Performance Indicators](../performance_indicators/index.md), the information is directly in the description of the scripts.

---

## 1. Strategic Pipeline


??? info "Strategic pipeline TOML description"
    ```toml
    {{ read_file("examples/toml/strategic_pipeline.toml") | indent(4) }}
    ```

---
## 2. Policy Package

??? info "Policy package TOML description"
    ```toml
    {{ read_file("examples/toml/policy_package.toml") }}
    ```

---
## 3. Heuristics Computation

??? info "Heuristics computation TOML description"
    ```toml
    {{ read_file("examples/toml/heuristics_computation.toml") }}
    ```

---
## 4. Pretactical Pipeline