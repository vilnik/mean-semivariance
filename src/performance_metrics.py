import scipy.stats as stats
import numpy as np
import loading_and_saving
import logging
import random
import plotting_utils

logger = logging.getLogger(__name__)

def find_avg_of_closest_controls(opt_control,
                          time,
                          current_val):
    # Convert dict keys to a list and sort by the portfolio value
    sorted_keys = sorted(opt_control.keys(), key=lambda x: x[1])
    lower_bound_key, upper_bound_key = None, None

    for key in sorted_keys:
        if key[0] == time:
            if key[1] <= current_val:
                lower_bound_key = key
            elif key[1] > current_val and upper_bound_key is None:
                upper_bound_key = key
                break

    if lower_bound_key is None:  # No smaller key found, use the smallest key available
        lower_bound_key = sorted_keys[0]
    if upper_bound_key is None:  # No larger key found, use the largest key available
        upper_bound_key = sorted_keys[-1]

    # Calculate average control
    avg_control = (opt_control[lower_bound_key] + opt_control[upper_bound_key]) / 2
    return avg_control

def evaluate_control(problem_spec,
                     opt_control,
                     num_sim,
                     type_of_control,
                     seed_nr):
    logger.info(f"Evaluating optimal control for {problem_spec.get_short_name()} using control {type_of_control} based on {num_sim} simulations.")
    problem_spec_params = problem_spec.get_params()

    random.seed(a=seed_nr)
    # Compute mean, variance and semivariance
    all_port_vals_paths = [[] for _ in range(num_sim)]  # Pre-initialize list for each simulation
    all_opt_controls = []
    for sim in range(num_sim):
        all_port_vals_paths[sim].append(problem_spec_params["init_port_val"])
        for time in range(0, problem_spec_params["time_T"]):
            step_rv = random.random()
            if step_rv < problem_spec_params["up_prob"]:
                y = problem_spec_params["up_ret"]
            elif step_rv < (1 - problem_spec_params["shock_prob"]):
                y = problem_spec_params["down_ret"]
            else:
                y = problem_spec_params["shock_ret"]

            current_val = all_port_vals_paths[sim][-1]
            current_opt_control = find_avg_of_closest_controls(opt_control, time, current_val)
            all_opt_controls.append(current_opt_control)
            next_port_val = current_val * (1 + problem_spec_params["interest"]) + current_opt_control * (y - problem_spec_params["interest"])
            all_port_vals_paths[sim].append(next_port_val)

    # Assuming all_port_vals_paths is populated with simulation data
    term_port_vals = [path[-1] for path in all_port_vals_paths]  # Extract terminal values from each simulation

    # Plot term_port_vals
    plotting_utils.plot_term_port_vals_hist(problem_spec,
                                                 type_of_control,
                                                 term_port_vals)

    mean_terminal_value = np.mean(term_port_vals)
    variance_terminal_value = np.var(term_port_vals)

    # Semivariance calculation: consider only values below the mean
    semivariance_terminal_value = np.mean(
        [(x - mean_terminal_value) ** 2 if x < mean_terminal_value else 0 for x in term_port_vals]
    )

    semivariance_terminal_value_2 = np.sum(
        [(x - mean_terminal_value) ** 2 for x in term_port_vals if x < mean_terminal_value]
    ) / len(term_port_vals)

    are_close = np.isclose(semivariance_terminal_value, semivariance_terminal_value_2)
    if not are_close:
        logger.error("Semivariance calculations are not consistent.")
        raise Exception("Semivariance calculations are not consistent.")

    mean_variance_tradeoff = mean_terminal_value - problem_spec_params["risk_aversion"] / 2 * variance_terminal_value
    mean_semivariance_tradeoff = mean_terminal_value - problem_spec_params["risk_aversion"] / 2 * semivariance_terminal_value

    avg_opt_control = np.mean(all_opt_controls)
    return {"mean_variance_tradeoff": mean_variance_tradeoff,
            "mean_semivariance_tradeoff": mean_semivariance_tradeoff,
            "avg_opt_control": avg_opt_control}