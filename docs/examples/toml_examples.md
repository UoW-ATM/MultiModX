# TOML configuration examples

All scripts in MultiModX are configured using [TOML](https://toml.io/en/) configuration files.

The data available on Zenodo (see [Input Files](input_format.md)) contains some examples, 
here a description of the different elements of the configuration  files is provided for:

1. [Strategic pipeline](#strategic-pipeline)
2. [Policy package definition](#policy-package)
3. [Heuristics computation](#heuristics-computation)
4. [Pre-tactical pipeline](#pretactical-pipeline)
5. For [Performance Indicators](../performance_indicators/index.md), the information is directly in the description of the scripts.



## Strategic Pipeline


```toml
{{ read_file("examples/toml/strategic_pipeline.toml") }}
```

## Policy Package
```toml
{{ read_file("examples/toml/policy_package.toml") }}
```

## Heuristics Computation
```toml
{{ read_file("examples/toml/heuristics_computation.toml") }}
```

## Pretactical Pipeline