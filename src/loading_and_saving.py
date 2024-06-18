import os
import json
import logging

logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FOLDER = os.path.join(base_dir, "results")

def load_opt_control_from_json(problem_spec,
                               solver_spec,
                               type_of_control):
    # Create the results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if solver_spec is not None:
        solver_spec_str = '_'.join([f"{key}_{value}" for key, value in solver_spec.items()])
    else:
        solver_spec_str = ""
    filename = f'opt_control_{type_of_control}_{problem_spec.get_long_name()}_{solver_spec_str}.json'
    filepath = os.path.join(RESULTS_FOLDER, filename)

    # Check if the file exists
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            # Load the data
            loaded_data = json.load(file)
            # Convert string keys back to tuples
            opt_control = {eval(key): value for key, value in loaded_data.items()}
        logger.info(f"Optimal control {filename} found in saved file.")
        return opt_control
    else:
        logger.info(f"Optimal control {filename} not found in saved file.")
        return None


def save_opt_control_to_json(opt_control,
                             problem_spec,
                             solver_spec,
                             type_of_control):
    # Create the results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if solver_spec is not None:
        solver_spec_str = '_'.join([f"{key}_{value}" for key, value in solver_spec.items()])
    else:
        solver_spec_str = ""
    filename = f'opt_control_{type_of_control}_{problem_spec.get_long_name()}_{solver_spec_str}.json'
    filepath = os.path.join(RESULTS_FOLDER, filename)

    # Convert the tuple keys to strings since JSON doesn't support tuple keys
    formatted_opt_control = {str(k): v for k, v in opt_control.items()}

    with open(filepath, "w") as file:
        json.dump(formatted_opt_control, file, indent=4)

    logger.info(f"Saved opt_control to {filepath}")