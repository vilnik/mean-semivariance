import logging
import numpy as np
import loading_and_saving
import math
import datetime
from multiprocessing import cpu_count, Pool
import concurrent.futures
import plotting_utils

logger = logging.getLogger(__name__)

def compute_term_g_and_h(time_T,
                         lower_terminal_port_val,
                         upper_terminal_port_val,
                         num_terminal_port_values):
    g_seq = {}
    h_seq = {}
    epsilon = 1e-10
    for port_val in np.linspace(lower_terminal_port_val, upper_terminal_port_val, num = num_terminal_port_values):
        g_seq[(time_T, port_val)] = port_val
        for term_port_val in np.linspace(lower_terminal_port_val, upper_terminal_port_val, num = num_terminal_port_values):
            h_seq[(time_T, port_val, term_port_val)] = 1 if abs(port_val - term_port_val) < epsilon else 0

    return g_seq, h_seq

def get_port_vals(problem_spec,
                  time_t):
    problem_spec_params = problem_spec.get_params()
    if problem_spec_params["shock_prob"] > 0 and problem_spec_params["shock_ret"] > 0:
        lower_port_val = problem_spec_params["init_port_val"] * (1 + problem_spec_params["down_ret"]) ** time_t
        upper_port_val = problem_spec_params["init_port_val"] * (1 + problem_spec_params["shock_ret"]) ** time_t
    elif problem_spec_params["shock_prob"] > 0 and problem_spec_params["shock_ret"] < 0:
        lower_port_val = problem_spec_params["init_port_val"] * (1 + problem_spec_params["shock_ret"]) ** time_t
        upper_port_val = problem_spec_params["init_port_val"] * (1 + problem_spec_params["up_ret"]) ** time_t
    else:
        lower_port_val = problem_spec_params["init_port_val"] * (1 + problem_spec_params["down_ret"]) ** time_t
        upper_port_val = problem_spec_params["init_port_val"] * (1 + problem_spec_params["up_ret"]) ** time_t

    return lower_port_val, upper_port_val

def get_term_port_vals_from_g_and_h_seqs(g_seq,
                                             h_seq,
                                             time_T):
    g_seq_terminal_values = set([key[1] for key, exp_term_port_val in g_seq.items() if key[0] == time_T])
    h_seq_terminal_values = set([key[1] for key, prob_term_port_val in h_seq.items() if key[0] == time_T])

    epsilon = 1e-9
    for g_val, h_val in zip(g_seq_terminal_values, h_seq_terminal_values):
        assert abs(g_val - h_val) < epsilon

    return sorted(g_seq_terminal_values)

def get_realistic_term_port_vals_from_g_and_h_seqs(problem_spec,
                                                g_seq,
                                                h_seq,
                                                time_t,
                                                port_val):
    # Obtain terminal portfolio values based on given g_seq and h_seq
    problem_spec_params = problem_spec.get_params()
    all_term_port_values = get_term_port_vals_from_g_and_h_seqs(g_seq, h_seq, problem_spec_params["time_T"])

    # Adjust term_port_values to only include realistic ones
    problem_spec_params = problem_spec.get_params()
    if problem_spec_params["shock_prob"] > 0 and problem_spec_params["shock_ret"] > 0:
        upper_term_port_value = port_val * (1 + problem_spec_params["shock_ret"]) ** (problem_spec_params["time_T"] - time_t)
        lower_term_port_value = port_val * (1 + problem_spec_params["down_ret"]) ** (problem_spec_params["time_T"] - time_t)
    elif problem_spec_params["shock_prob"] > 0 and problem_spec_params["shock_ret"] < 0:
        upper_term_port_value = port_val * (1 + problem_spec_params["up_ret"]) ** (problem_spec_params["time_T"] - time_t)
        lower_term_port_value = port_val * (1 + problem_spec_params["shock_ret"]) ** (problem_spec_params["time_T"] - time_t)
    else:
        upper_term_port_value = port_val * (1 + problem_spec_params["up_ret"]) ** (problem_spec_params["time_T"] - time_t)
        lower_term_port_value = port_val * (1 + problem_spec_params["down_ret"]) ** (problem_spec_params["time_T"] - time_t)

    all_term_port_values.sort()

    realistic_term_port_values = []

    for value in all_term_port_values:
        if lower_term_port_value <= value <= upper_term_port_value:
            realistic_term_port_values.append(value)

    # Expand the realistic range by including one adjacent value on each side
    # This ensures we capture the nearest outside values if needed for additional context or smoothing
    first_index = all_term_port_values.index(realistic_term_port_values[0]) if realistic_term_port_values else None
    last_index = all_term_port_values.index(realistic_term_port_values[-1]) if realistic_term_port_values else None

    if first_index is not None and first_index > 0:
        # Include one value before the first realistic value, if possible
        realistic_term_port_values.insert(0, all_term_port_values[first_index - 1])
    if last_index is not None and last_index < len(all_term_port_values) - 1:
        # Include one value after the last realistic value, if possible
        realistic_term_port_values.append(all_term_port_values[last_index + 1])

    return realistic_term_port_values

def time_in_h_seq(time_t,
                  h_seq):
    # We only need to check the first element of the tuple for "time"
    return any(key[0] == time_t for key in h_seq.keys())


def time_in_g_seq(time_t,
                  g_seq):
    # We only need to check the first element of the tuple for "time"
    return any(key[0] == time_t for key in g_seq.keys())


def interpolate_h_seq(time_t,
                      port_val,
                      h_seq,
                      term_port_val):
    epsilon=1e-5

    # Separate out key_vals for the given time
    h_seq_time_t = [(key[1], key[2], prob_term_port_val) for key, prob_term_port_val in h_seq.items() if key[0] == time_t]

    # If no vals found for the given time, return None
    if not h_seq_time_t:
        return None

    # Sort based on the port_val
    h_seq_time_t.sort(key = lambda x: x[0])

    lower_h_seq_time_t_port_val = None
    upper_h_seq_time_t_port_val = None

    # Find upper and lower h_seq_time_t based on port_val
    for i, (h_seq_time_t_port_val, _, _) in enumerate(h_seq_time_t):
        if abs(h_seq_time_t_port_val - port_val) < epsilon:
            lower_h_seq_time_t_port_val = h_seq_time_t_port_val
            upper_h_seq_time_t_port_val = h_seq_time_t_port_val
            break
        elif h_seq_time_t_port_val < port_val:
            lower_h_seq_time_t_port_val = h_seq_time_t_port_val
        else:
            upper_h_seq_time_t_port_val = h_seq_time_t_port_val
            break

    # Filter h_seq_time_t to only include lower_h_seq_port_val and upper_h_seq_port_val
    lower_h_seq_time_t = [item for item in h_seq_time_t if item[0] == lower_h_seq_time_t_port_val]
    upper_h_seq_time_t = [item for item in h_seq_time_t if item[0] == upper_h_seq_time_t_port_val]

    # Sort based on term_port_val for both lists
    lower_h_seq_time_t.sort(key = lambda x: x[1])
    upper_h_seq_time_t.sort(key = lambda x: x[1])

    # Function to calculate weight
    def calculate_weight(close_val, target_val):
        distance = abs(target_val - close_val)
        return 1 / max(distance, 0.0000001)  # Avoid division by zero

    # Interpolate the probability for lower_h_seq_time_t
    lower_h_seq_time_t_lower_prob_term_port_val = None
    lower_h_seq_time_t_upper_prob_term_port_val = None
    lower_lower_weight = None
    lower_upper_weight = None

    for _, h_seq_term_port_val, prob_term_port_val in lower_h_seq_time_t:
        if abs(h_seq_term_port_val - term_port_val) < epsilon:
            lower_h_seq_time_t_lower_prob_term_port_val = prob_term_port_val
            lower_h_seq_time_t_upper_prob_term_port_val = prob_term_port_val
            lower_lower_weight = 0.5
            lower_upper_weight = 0.5
            break
        elif h_seq_term_port_val < term_port_val:
            lower_h_seq_time_t_lower_prob_term_port_val = prob_term_port_val
            lower_lower_weight = calculate_weight(h_seq_term_port_val, term_port_val)
        else:
            lower_h_seq_time_t_upper_prob_term_port_val = prob_term_port_val
            lower_upper_weight = calculate_weight(h_seq_term_port_val, term_port_val)
            break

    # Interpolate the probability for upper_h_seq_time_t
    upper_h_seq_time_t_lower_prob_term_port_val = None
    upper_h_seq_time_t_upper_prob_term_port_val = None
    upper_lower_weight = None
    upper_upper_weight = None

    for _, h_seq_term_port_val, prob_term_port_val in upper_h_seq_time_t:
        if abs(h_seq_term_port_val - term_port_val) < epsilon:
            upper_h_seq_time_t_lower_prob_term_port_val = prob_term_port_val
            upper_h_seq_time_t_upper_prob_term_port_val = prob_term_port_val
            upper_lower_weight = 0.5
            upper_upper_weight = 0.5
            break
        elif h_seq_term_port_val < term_port_val:
            upper_h_seq_time_t_lower_prob_term_port_val = prob_term_port_val
            upper_lower_weight = calculate_weight(h_seq_term_port_val, term_port_val)
        else:
            upper_h_seq_time_t_upper_prob_term_port_val = prob_term_port_val
            upper_upper_weight = calculate_weight(h_seq_term_port_val, term_port_val)
            break

    # Calculate weights based on port_val position
    if lower_h_seq_time_t_port_val is not None and upper_h_seq_time_t_port_val is not None and lower_h_seq_time_t_port_val != upper_h_seq_time_t_port_val:
        port_val_weight_lower = 1 / max(port_val - lower_h_seq_time_t_port_val, 0.0000001)
        port_val_weight_upper = 1 / max(upper_h_seq_time_t_port_val - port_val, 0.0000001)
    elif abs(lower_h_seq_time_t_port_val - upper_h_seq_time_t_port_val) < epsilon:
        port_val_weight_lower = 0.5
        port_val_weight_upper = 0.5
    else:
        logger.error("Something is wrong with lower_h_seq_time_t_port_val and/or upper_h_seq_time_t_port_val.")
        raise Exception("Something is wrong with lower_h_seq_time_t_port_val and/or upper_h_seq_time_t_port_val.")

    # Calculate weighted average probabilities for lower and upper series
    weighted_prob_lower = 0
    weighted_prob_upper = 0
    weight_lower = 0
    weight_upper = 0

    if lower_h_seq_time_t_lower_prob_term_port_val is not None and lower_lower_weight is not None:
        weighted_prob_lower += lower_h_seq_time_t_lower_prob_term_port_val * lower_lower_weight
        weight_lower += lower_lower_weight

    if lower_h_seq_time_t_upper_prob_term_port_val is not None and lower_upper_weight is not None:
        weighted_prob_lower += lower_h_seq_time_t_upper_prob_term_port_val * lower_upper_weight
        weight_lower += lower_upper_weight

    if upper_h_seq_time_t_lower_prob_term_port_val is not None and upper_lower_weight is not None:
        weighted_prob_upper += upper_h_seq_time_t_lower_prob_term_port_val * upper_lower_weight
        weight_upper += upper_lower_weight

    if upper_h_seq_time_t_upper_prob_term_port_val is not None and upper_upper_weight is not None:
        weighted_prob_upper += upper_h_seq_time_t_upper_prob_term_port_val * upper_upper_weight
        weight_upper += upper_upper_weight

    # Compute the final weighted average probability
    total_prob = 0
    total_weight = 0

    if weight_lower > 0:
        total_prob += (weighted_prob_lower / weight_lower) * port_val_weight_lower
        total_weight += port_val_weight_lower

    if weight_upper > 0:
        total_prob += (weighted_prob_upper / weight_upper) * port_val_weight_upper
        total_weight += port_val_weight_upper

    if total_weight <= 0:
        raise Exception("Something is wrong with total_weight.")
        raise Exception("Something is wrong with total_weight.")

    avg_prob = total_prob / total_weight

    return avg_prob


def interpolate_g_seq(time_t,
                      port_val,
                      g_seq):
    # Separate out key_vals for the given time
    g_seq_time_t = [(key[1], exp_term_port_val) for key, exp_term_port_val in g_seq.items() if key[0] == time_t]

    # If no vals found for the given time, return None
    if not g_seq_time_t:
        return None

    # Sort based on the key_val
    g_seq_time_t.sort(key = lambda x: x[0])

    lower_exp_term_port_val = None
    upper_exp_term_port_val = None
    lower_weight = None
    upper_weight = None

    for i, (g_seq_port_val, exp_term_port_val) in enumerate(g_seq_time_t):
        if g_seq_port_val < port_val:
            lower_exp_term_port_val = exp_term_port_val
            lower_weight = 1 / (port_val - g_seq_port_val) if port_val - g_seq_port_val != 0 else float('inf')
        elif g_seq_port_val == port_val:
            return exp_term_port_val  # If the exact val is found, return it
        else:
            upper_exp_term_port_val = exp_term_port_val
            upper_weight = 1 / (g_seq_port_val - port_val) if g_seq_port_val - port_val != 0 else float('inf')
            break

    # Compute weighted avg if both exist, else return the existing val
    if lower_exp_term_port_val is not None and upper_exp_term_port_val is not None:
        total_weight = lower_weight + upper_weight
        return (lower_exp_term_port_val * lower_weight + upper_exp_term_port_val * upper_weight) / total_weight
    elif lower_exp_term_port_val is not None:
        return lower_exp_term_port_val
    else:
        return upper_exp_term_port_val


def compute_exp_h_at_time_t_given_control_and_term_port_val(problem_spec,
                                                         h_seq,
                                                         time_t,
                                                         port_val,
                                                         term_port_val,
                                                         control):
    assert time_in_h_seq(time_t + 1, h_seq)
    problem_spec_params = problem_spec.get_params()

    # Compute the exp val of port at time_t + 1 given a control u
    up_port_val_one_period = (port_val * (1 + problem_spec_params["interest"]) + control * (problem_spec_params["up_ret"] - problem_spec_params["interest"]))
    down_port_val_one_period = (port_val * (1 + problem_spec_params["interest"]) + control * (problem_spec_params["down_ret"] - problem_spec_params["interest"]))
    shock_port_val_one_period = (port_val * (1 + problem_spec_params["interest"]) + control * (problem_spec_params["shock_ret"] - problem_spec_params["interest"]))

    # Find closest val of h at time_t + 1 given control u and term_port_val
    exp_h_seq_val = problem_spec_params["up_prob"] * interpolate_h_seq(time_t + 1, up_port_val_one_period, h_seq, term_port_val) + \
                    (1 - problem_spec_params["up_prob"] - problem_spec_params["shock_prob"]) * interpolate_h_seq(time_t + 1, down_port_val_one_period, h_seq, term_port_val) + \
                    (problem_spec_params["shock_prob"]) * interpolate_h_seq(time_t + 1, shock_port_val_one_period, h_seq, term_port_val)

    return exp_h_seq_val

def compute_exp_g_at_time_t_given_control(problem_spec,
                                            g_seq,
                                            time_t,
                                            port_val,
                                            control):
    assert time_in_g_seq(time_t + 1, g_seq)
    problem_spec_params = problem_spec.get_params()

    # Compute the exp val of port at time_t + 1 given a control u
    up_port_val_one_period = (port_val * (1 + problem_spec_params["interest"]) + control * (problem_spec_params["up_ret"] - problem_spec_params["interest"]))
    down_port_val_one_period = (port_val * (1 + problem_spec_params["interest"]) + control * (problem_spec_params["down_ret"] - problem_spec_params["interest"]))
    shock_port_val_one_period = (port_val * (1 + problem_spec_params["interest"]) + control * (problem_spec_params["shock_ret"] - problem_spec_params["interest"]))

    # Find closest vals of g at time_t
    exp_g_seq_val = problem_spec_params["up_prob"] * interpolate_g_seq(time_t + 1, up_port_val_one_period, g_seq) + \
                    (1 - problem_spec_params["up_prob"] - problem_spec_params["shock_prob"]) * interpolate_g_seq(time_t + 1, down_port_val_one_period, g_seq) + \
                    (problem_spec_params["shock_prob"]) * interpolate_g_seq(time_t + 1, shock_port_val_one_period, g_seq)

    return exp_g_seq_val

def compute_mean_risk_J(problem_spec,
                        control,
                           g_seq,
                           h_seq,
                           time_t,
                           port_val,
                           term_port_values,
                           type_of_control):
    # Compute expected g for the given control at time t
    exp_g_at_time_t_given_control = compute_exp_g_at_time_t_given_control(problem_spec,
                                                                              g_seq,
                                                                              time_t,
                                                                              port_val,
                                                                              control)

    integral = 0
    problem_spec_params = problem_spec.get_params()
    for term_port_val in term_port_values:
        if (type_of_control == "mean_variance_num") or (type_of_control == "mean_semivariance_num" and term_port_val < exp_g_at_time_t_given_control):
            exp_h_at_time_t_given_control_and_term_port_val = compute_exp_h_at_time_t_given_control_and_term_port_val(problem_spec,
                                                                                                                h_seq,
                                                                                                                time_t,
                                                                                                                port_val,
                                                                                                                term_port_val,
                                                                                                                control)

            if exp_h_at_time_t_given_control_and_term_port_val > 0:
                integral += exp_h_at_time_t_given_control_and_term_port_val * (
                        term_port_val ** 2 - 2 * term_port_val * exp_g_at_time_t_given_control + exp_g_at_time_t_given_control ** 2)

    mean_risk_J = exp_g_at_time_t_given_control - problem_spec_params["risk_aversion"] / 2 * integral

    return {"control": control,
            "mean_risk_J": mean_risk_J,
            "integral": integral}

def compute_opt_control_at_time_t_given_port_val(problem_spec,
                                                     solver_spec,
                                                     type_of_control,
                                                     g_seq,
                                                     h_seq,
                                                     port_val,
                                                     time_t,
                                                     opt_control_range,
                                                     plot_mean_risk_reward):
    # Realistic terminal portfolio values
    realistic_term_port_values = get_realistic_term_port_vals_from_g_and_h_seqs(problem_spec,
                                                                                g_seq,
                                                                                h_seq,
                                                                                time_t,
                                                                                port_val)

    # Generate list of controls for exploration
    if opt_control_range is None:
        lower_control = 0
        upper_control = port_val
    else:
        lower_control = opt_control_range[0]
        upper_control = opt_control_range[1]

    steps = int(round((upper_control - lower_control) / solver_spec["control_step_size"])) + 1
    controls = list(np.linspace(lower_control, upper_control, num = steps))

    max_mean_risk_J = None
    opt_mean_risk_control = None

    # Create a generator that produces the required sequence of arguments for each call
    def argument_generator():
        for control in controls:
            yield (problem_spec, control, g_seq, h_seq, time_t, port_val, realistic_term_port_values, type_of_control)

    # Parallel processing to compute mean-risk values for each control
    control_and_J = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers = cpu_count() - 2) as executor:
        args_iterables = zip(*argument_generator())
        for mean_risk_J_results in executor.map(compute_mean_risk_J, *args_iterables):
            control = mean_risk_J_results["control"]
            mean_risk_J = mean_risk_J_results["mean_risk_J"]
            control_and_J[control] = mean_risk_J
            # Update opt control and its value
            if max_mean_risk_J is None or mean_risk_J > max_mean_risk_J:
                opt_mean_risk_control = control
                max_mean_risk_J = mean_risk_J

    if plot_mean_risk_reward:
        # Call the plot function
        plotting_utils.plot_control_vs_J(problem_spec,
                                          solver_spec,
                                          control_and_J,
                                          type_of_control)

    return opt_mean_risk_control

def find_closest_control(opt_control,
                         time_t,
                         target_pv):
    closest_control = None
    smallest_diff = float('inf')  # Initialize with an infinitely large number

    for (t, pv), control in opt_control.items():
        if t == time_t:
            diff = abs(pv - target_pv)
            if diff < smallest_diff:
                smallest_diff = diff
                closest_control = control

    return closest_control

def update_g_and_h_sequences(problem_spec,
                             g_seq,
                             h_seq,
                             opt_control,
                             time_t):
    port_vals = set(pv for (t, pv), control in opt_control.items() if t == time_t)
    # Loop through all unique portfolio values at time_t
    for port_val in port_vals:
        # Update g sequence using the opt control
        closest_opt_control = find_closest_control(opt_control, time_t, port_val)

        exp_g_at_time_t_given_opt_control = compute_exp_g_at_time_t_given_control(problem_spec,
                                                                                      g_seq,
                                                                                      time_t,
                                                                                      port_val,
                                                                                      closest_opt_control)

        g_seq[(time_t, port_val)] = exp_g_at_time_t_given_opt_control

        # Obtain realistic terminal portfolio values based on current g and h sequences
        realistic_term_port_values = get_realistic_term_port_vals_from_g_and_h_seqs(problem_spec,
                                                                                      g_seq,
                                                                                      h_seq,
                                                                                      time_t,
                                                                                      port_val)

        # Update h sequence for each realistic terminal portfolio value using the computed average opt control
        for term_port_val in realistic_term_port_values:
            exp_h_at_time_t_given_control_and_term_port_val = compute_exp_h_at_time_t_given_control_and_term_port_val(
                problem_spec,
                h_seq,
                time_t,
                port_val,
                term_port_val,
                closest_opt_control)
            h_seq[(time_t, port_val, term_port_val)] = exp_h_at_time_t_given_control_and_term_port_val

    return g_seq, h_seq

def adjust_opt_control(opt_control,
                       time_t,
                       averaging = "all"):
    if averaging == "all":
        # Extract all opt controls at time_t
        opt_controls_at_time_t = [control for (t, pv), control in opt_control.items() if t == time_t]

        # Calculate the average of these controls, if any are found
        if opt_controls_at_time_t:
            avg_opt_control = sum(opt_controls_at_time_t) / len(opt_controls_at_time_t)
            # Update all entries for time_t to use the average control
            for key in list(opt_control.keys()):
                if key[0] == time_t:
                    opt_control[key] = avg_opt_control
        else:
            #Todo: Implement regression fitting
            logger.error("Other averaging methods not implemented.")
            raise ValueError("No opt controls found for averaging at time_t")
    else:
        # If you have other averaging methods, they would be implemented here
        logger.error("Other averaging methods not implemented.")
        raise Exception("Other averaging methods not implemented.")

    return opt_control

def get_opt_control_range(opt_control,
                          time_t):
    port_vals_at_time_t = [port_val for (time, port_val) in opt_control.keys() if
                           time == time_t]
    if len(port_vals_at_time_t) > 30:
        # Assuming opt_control[(time_t, port_val)] gives you the control value you need
        lower_control = min(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 0.8
        upper_control = max(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 1.2
        opt_control_range = [lower_control, upper_control]
    elif len(port_vals_at_time_t) > 20:
        lower_control = min(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 0.75
        upper_control = max(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 1.25
        opt_control_range = [lower_control, upper_control]
    elif len(port_vals_at_time_t) > 10:
        lower_control = min(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 0.7
        upper_control = max(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 1.3
        opt_control_range = [lower_control, upper_control]
    elif len(port_vals_at_time_t) > 5:
        lower_control = min(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 0.65
        upper_control = max(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 1.35
        opt_control_range = [lower_control, upper_control]
    elif len(port_vals_at_time_t) > 2:
        lower_control = min(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 0.5
        upper_control = max(opt_control[(time_t, pv)] for pv in port_vals_at_time_t) * 1.5
        opt_control_range = [lower_control, upper_control]
    else:
        opt_control_range = None

    return opt_control_range

def compute_mean_ret(problem_spec):
    problem_spec_params = problem_spec.get_params()
    mean_ret = (problem_spec_params["up_ret"] - problem_spec_params["interest"]) * problem_spec_params["up_prob"] + \
               (problem_spec_params["down_ret"] - problem_spec_params["interest"]) * (1 - problem_spec_params["up_prob"] - problem_spec_params["shock_prob"]) + \
               (problem_spec_params["shock_ret"] - problem_spec_params["interest"]) * (problem_spec_params["shock_prob"])

    return mean_ret


def compute_var_ret(problem_spec):
    problem_spec_params = problem_spec.get_params()

    mean_ret = compute_mean_ret(problem_spec)
    var_ret = problem_spec_params["up_prob"] * (
                problem_spec_params["up_ret"] - problem_spec_params["interest"] - mean_ret) ** 2 + \
              (1 - problem_spec_params["up_prob"] - problem_spec_params["shock_prob"]) * (
                          problem_spec_params["down_ret"] - problem_spec_params["interest"] - mean_ret) ** 2 + \
              problem_spec_params["shock_prob"] * (
                          problem_spec_params["shock_ret"] - problem_spec_params["interest"] - mean_ret) ** 2

    return var_ret


def compute_semivar_ret(problem_spec):
    problem_spec_params = problem_spec.get_params()

    mean_ret = compute_mean_ret(problem_spec)

    semivar_components = [
        (problem_spec_params["up_ret"] - problem_spec_params["interest"] - mean_ret) ** 2 if problem_spec_params[
                                                                                                 "up_ret"] -
                                                                                             problem_spec_params[
                                                                                                 "interest"] < mean_ret else 0,
        (problem_spec_params["down_ret"] - problem_spec_params["interest"] - mean_ret) ** 2 if problem_spec_params[
                                                                                                   "down_ret"] -
                                                                                               problem_spec_params[
                                                                                                   "interest"] < mean_ret else 0,
        (problem_spec_params["shock_ret"] - problem_spec_params["interest"] - mean_ret) ** 2 if problem_spec_params[
                                                                                                    "shock_ret"] -
                                                                                                problem_spec_params[
                                                                                                    "interest"] < mean_ret else 0
    ]

    semivar_ret = problem_spec_params["up_prob"] * semivar_components[0] + \
                  (1 - problem_spec_params["up_prob"] - problem_spec_params["shock_prob"]) * semivar_components[1] + \
                  problem_spec_params["shock_prob"] * semivar_components[2]

    return semivar_ret

def determine_opt_control(problem_spec,
                    solver_spec,
                    type_of_control,
                    plot_mean_risk_reward):
    logger.info(f"Determining optimal control for {problem_spec.get_short_name()} using control {type_of_control}.")

    opt_control = loading_and_saving.load_opt_control_from_json(problem_spec,
                                                                 solver_spec,
                                                                 type_of_control)

    if opt_control is None or plot_mean_risk_reward:
        logger.info(f"Computing optimal control.")
        opt_control = {}
        tolerance = 1e-5
        problem_spec_params = problem_spec.get_params()
        if type_of_control == "mean_semivariance_num" or type_of_control == "mean_variance_num":
            lower_term_port_val, upper_term_port_val = get_port_vals(problem_spec, problem_spec_params["time_T"])
            g_seq, h_seq = compute_term_g_and_h(problem_spec_params["time_T"],
                                                lower_term_port_val - 0.1,
                                                upper_term_port_val + 0.1,
                                                solver_spec["num_port_values_per_time"])
            time_t = problem_spec_params["time_T"] - 1
            while time_t >= 0:
                lower_port_val, upper_port_val = get_port_vals(problem_spec, time_t)
                num_values = 1 if lower_port_val == upper_port_val else solver_spec["num_port_values_per_time"]
                counter = 0
                for port_val in np.linspace(lower_port_val, upper_port_val, num = num_values):
                    logger.info(f"Computing multi-period control at time {time_t} using portfolio value {port_val} and control type {type_of_control}.")

                    opt_control_range = get_opt_control_range(opt_control,
                                                              time_t)

                    opt_mean_risk_control_num = compute_opt_control_at_time_t_given_port_val(problem_spec,
                                                                                                     solver_spec,
                                                                                                     type_of_control,
                                                                                                     g_seq,
                                                                                                     h_seq,
                                                                                                     port_val,
                                                                                                     time_t,
                                                                                                     opt_control_range,
                                                                                                     plot_mean_risk_reward)

                    # Check if opt_mean_risk_control_num is close to lower_control or upper_control
                    if opt_control_range is not None:
                        # Check if opt_mean_risk_control_num is close to lower_control or upper_control
                        if math.isclose(opt_mean_risk_control_num, opt_control_range[0], abs_tol=tolerance) or \
                                math.isclose(opt_mean_risk_control_num, opt_control_range[1], abs_tol=tolerance):
                            logger.error(f"The numerical optimal control is {opt_mean_risk_control_num}, which is too close to one of the control limits [{opt_control_range[0]}, {opt_control_range[1]}].")
                            raise Exception(f"The numerical optimal control is {opt_mean_risk_control_num}, which is too close to one of the control limits [{opt_control_range[0]}, {opt_control_range[1]}].")

                    logger.info(f"opt control at time {time_t} using portfolio value {port_val} and control typ {type_of_control} is {opt_mean_risk_control_num}.")
                    complete = 100 * ((problem_spec_params["time_T"] - time_t - 1) / problem_spec_params["time_T"] + (counter + 1) / num_values * 1 / problem_spec_params["time_T"])
                    counter = counter + 1
                    logger.info(f"Percentage complete {complete}. Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
                    opt_control[(time_t, port_val)] = opt_mean_risk_control_num

                opt_control = adjust_opt_control(opt_control,
                                                 time_t,
                                                 averaging="all")

                g_seq, h_seq = update_g_and_h_sequences(problem_spec,
                                                        g_seq,
                                                         h_seq,
                                                         opt_control,
                                                         time_t)

                time_t = time_t - 1

        elif type_of_control == "mean_semivariance_theo" and problem_spec_params["time_T"] == 1:
            mean_ret = compute_mean_ret(problem_spec)
            semivar_ret = compute_semivar_ret(problem_spec)
            opt_control[(0, problem_spec_params["init_port_val"])] = mean_ret / semivar_ret * 1 / problem_spec_params["risk_aversion"]

        elif type_of_control == "mean_variance_theo":
            time_t = problem_spec_params["time_T"] - 1
            while time_t >= 0:
                lower_port_val, upper_port_val = get_port_vals(problem_spec, time_t)
                num_values = 1 if lower_port_val == upper_port_val else solver_spec["num_port_values_per_time"]
                mean_ret = compute_mean_ret(problem_spec)
                var_ret = compute_var_ret(problem_spec)
                for port_val in np.linspace(lower_port_val, upper_port_val, num = num_values):
                    opt_control[(time_t, port_val)] = mean_ret / (problem_spec_params["risk_aversion"] * var_ret) * 1 / (1 + problem_spec_params["interest"])**(problem_spec_params["time_T"] - time_t - 1)

                time_t = time_t - 1
        else:
            logger.error("Unknown type_of_control and/or incompatible combination of type_of_control and time_T")
            raise Exception("Unknown type_of_control and/or incompatible combination of type_of_control and time_T")

        loading_and_saving.save_opt_control_to_json(opt_control,
                                                 problem_spec,
                                                 solver_spec,
                                                 type_of_control)

    return opt_control