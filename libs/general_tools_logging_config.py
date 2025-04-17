import os
import tomli
import tomli_w
import logging
from pathlib import Path


# Define custom logging levels
IMPORTANT_INFO = 35  # Between WARNING (30) and ERROR (40)


def important_info(self, message, *args, **kwargs):
    if self.isEnabledFor(IMPORTANT_INFO):
        self._log(IMPORTANT_INFO, message, args, **kwargs)


def setup_logging(verbosity, log_to_console=True, log_to_file=None, file_reset=False, file_level=None):
    # Define the log levels in order of increasing verbosity
    levels = [logging.ERROR, IMPORTANT_INFO, logging.WARNING, logging.INFO, logging.DEBUG]

    # ERROR and WARNING are always considered, if verbosity=1 then IMPORTANT_INFO too
    level = levels[min(len(levels) - 1, verbosity)]  # Ensure the level does not exceed DEBUG

    # Create the main logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove default handlers if they exist
    logger.handlers = []

    # Format for log messages
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # Console handler (if enabled)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)  # Set console log level
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        logger.important_info(f"Logging to console enabled set to {logging.getLevelName(level)}")

    # File handler (if enabled)
    if log_to_file:
        # Set mode: 'w' to overwrite (reset) the file or 'a' to append
        file_mode = 'w' if file_reset else 'a'
        file_handler = logging.FileHandler(log_to_file, mode=file_mode)

        # Ensure file_level defaults to WARNING if not passed
        file_logging_level = file_level if file_level is not None else logging.WARNING
        file_handler.setLevel(file_logging_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        logger.important_info(f"Logging to file: {log_to_file} set to {logging.getLevelName(file_handler.level)}")


def save_information_config_used(toml_config, args):

    # Function to recursively convert booleans and None to string
    def convert_boolean_none_to_str(value):
        if isinstance(value, bool):  # Convert booleans to strings
            return str(value)
        elif value is None:  # Convert None to the string 'None'
            return "None"
        else:
            return value  # Return other types unchanged

    # Function to convert Namespace to dictionary
    def namespace_to_dict(namespace):
        # Convert Namespace to dict
        return {key: convert_boolean_none_to_str(value) for key, value in vars(namespace).items()}

    def convert_paths_to_strings(obj):
        """Recursively convert PosixPath objects to strings in a dictionary or list."""
        if isinstance(obj, dict):  # If it's a dictionary, recurse on its values
            return {key: convert_paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):  # If it's a list, recurse on each item
            return [convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, Path):  # If it's a PosixPath, convert it to a string
            return str(obj)
        else:
            return obj  # Return other types unchanged

    # Convert Namespace to dictionary
    args_dict = namespace_to_dict(args)

    # Convert PosixPaths to strings
    toml_config_str = convert_paths_to_strings(toml_config)

    # Add the converted args dictionary to toml_config
    toml_config_str = {**toml_config_str, 'args': args_dict}

    config_used_path = Path(toml_config['general']['experiment_path']) / toml_config['general'][
        'output_folder'] / 'config_used.toml'
    os.makedirs(config_used_path.parent, exist_ok=True)

    with open(config_used_path, "wb") as fp:
        tomli_w.dump(toml_config_str, fp)


def process_strategic_config_file(toml_file, end_output_folder=None):
    with open(Path(toml_file), mode="rb") as fp:
        toml_config = tomli.load(fp)

    toml_config['network_definition']['network_path'] = toml_config['general']['experiment_path']
    if 'pre_processed_input_folder' in toml_config['general'].keys():
        toml_config['network_definition']['pre_processed_input_folder'] = toml_config['general']['pre_processed_input_folder']

    if end_output_folder is not None:
        toml_config['general']['output_folder'] = toml_config['general']['output_folder'] + end_output_folder

    if 'output' not in toml_config.keys():
        toml_config['output'] = {}

    if 'output_folder' in toml_config['general']:
        toml_config['network_definition']['processed_folder'] = toml_config['general']['output_folder']
        toml_config['output']['output_folder'] = (Path(toml_config['general']['experiment_path']) /
                                                  toml_config['general']['output_folder'] /
                                                  'paths_itineraries')

    toml_config['demand']['demand'] = toml_config['general']['experiment_path'] + toml_config['demand']['demand']

    if 'policy_package' in toml_config.keys():
        path_policy_package = (Path(toml_config['general']['experiment_path']) /
                               toml_config['policy_package']['policy_package'])

        with open(path_policy_package, mode="rb") as fp:
            policies_to_apply_config = tomli.load(fp)

        toml_config['policy_package'] = policies_to_apply_config

    if 'sensitivities_logit' in toml_config['other_param'].keys():
        toml_config['other_param']['sensitivities_logit']['sensitivities'] = (Path(toml_config['general']['experiment_path']) /
                                                                              toml_config['other_param']['sensitivities_logit']['sensitivities'])

    if 'heuristics_precomputed' in toml_config['other_param'].keys():
        toml_config['other_param']['heuristics_precomputed']['heuristics_precomputed_air'] = (Path(toml_config['general']['experiment_path']) /
                                                                                              toml_config['other_param']['heuristics_precomputed']['heuristics_precomputed_air'])
        toml_config['other_param']['heuristics_precomputed']['heuristics_precomputed_rail'] = (Path(toml_config['general']['experiment_path']) /
                                                                                               toml_config['other_param']['heuristics_precomputed']['heuristics_precomputed_rail'])

    if 'tactical_input' in toml_config['other_param'].keys():
        toml_config['other_param']['tactical_input']['aircraft']['ac_type_icao_iata_conversion'] = (Path(toml_config['general']['experiment_path']) /
                                                                                                     toml_config['other_param']['tactical_input']['aircraft']['ac_type_icao_iata_conversion'])

        toml_config['other_param']['tactical_input']['aircraft']['ac_mtow'] = (
                    Path(toml_config['general']['experiment_path']) /
                    toml_config['other_param']['tactical_input']['aircraft']['ac_mtow'])

        toml_config['other_param']['tactical_input']['aircraft']['ac_wtc'] = (
                    Path(toml_config['general']['experiment_path']) /
                    toml_config['other_param']['tactical_input']['aircraft']['ac_wtc'])


        toml_config['other_param']['tactical_input']['airlines']['airline_ao_type'] = (Path(toml_config['general']['experiment_path']) /
                                                                                       toml_config['other_param']['tactical_input']['airlines']['airline_ao_type'])

        toml_config['other_param']['tactical_input']['airlines']['airline_iata_icao'] = (
                    Path(toml_config['general']['experiment_path']) /
                    toml_config['other_param']['tactical_input']['airlines']['airline_iata_icao'])

    return toml_config