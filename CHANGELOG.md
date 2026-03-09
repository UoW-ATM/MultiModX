# Changelog

## [2.0.0] - 2026-03-09
### Added
- Possibility to use more than one demand
  - Multi logit model for demand distribution
  - Possibility to indicate which logit model to use for which demand
  - International mobility (to-from Spain to international destinations)
  - Logit models weights in release (libs/logit_model/)
- Reproducibility
  - Documentation
  - Data sample in Zenodo (CS_sample v0.1)

### Changed
- TOML configuration files more flexible: able to define for each demand which logit model to use
- Some parameters moved to TOML (e.g. default seats in flights and trains)

### Fixed
- Memory run in filtering and clustering of alternatives

## [1.0.0] - 2025-07-18
### Added
- Initial release with one demand and one logit model possible
- Focus on intra-Spain mobility

### Notes
- Version used to generate results for the Exploratory Research Report (ERR) deliverables of MultiModX: https://cordis.europa.eu/project/id/101114815
