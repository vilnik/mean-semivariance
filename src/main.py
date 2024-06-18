import os
from datetime import datetime
import logging
import json
import mean_risk_control
import performance_metrics
import plotting_utils
import problem_specs

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FOLDER = os.path.join(base_dir, "results")

def setup_logging():
    # Navigate up one directory from 'src' and create 'logs' at that level
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_directory = os.path.join(base_directory, "logs")
    os.makedirs(log_directory, exist_ok=True)

    # Generate a timestamped filename for the log
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = os.path.join(log_directory, f'mean_semivariance_{timestamp}.log')

    # Configure the root logger to write to the timestamped file, with a specific format for each log entry
    logging.basicConfig(filename=log_filename, filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # Log a message indicating that logging setup is complete
    logging.info('Logging setup complete. Logging to: %s', log_filename)

def main():
    setup_logging()
    # Retrieve a logger instance for this module
    logger = logging.getLogger(__name__)
    logger.info("Application started.")

    # Optimal control step size
    one_period_solver_spec = {"control_step_size": 0.05}

    # =============================================================================
    # One-period
    # =============================================================================

    ## Number of terminal values comparison
    # -------------------------------------
    one_period_problem_spec = problem_specs.ProblemSpec("one_period_interest_no_shocks_small_gamma")
    one_period_problem_spec_params = one_period_problem_spec.get_params()
    num_port_values_per_time_list = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
    opt_mean_semivariance_controls_num = {}
    opt_mean_variance_controls_num = {}
    for num_port_values_per_time in num_port_values_per_time_list:
        one_period_solver_spec["num_port_values_per_time"] = num_port_values_per_time
        opt_mean_semivariance_control_num = mean_risk_control.determine_opt_control(one_period_problem_spec,
                                                                           one_period_solver_spec,
                                                                           "mean_semivariance_num",
                                                                           False)
        opt_mean_semivariance_controls_num[num_port_values_per_time] = opt_mean_semivariance_control_num[(0, one_period_problem_spec_params["init_port_val"])]

        opt_mean_variance_control_num = mean_risk_control.determine_opt_control(one_period_problem_spec,
                                                             one_period_solver_spec,
                                                             "mean_variance_num",
                                                             False)

        opt_mean_variance_controls_num[num_port_values_per_time] = opt_mean_variance_control_num[(0, one_period_problem_spec_params["init_port_val"])]


    opt_mean_semivariance_control_theo = mean_risk_control.determine_opt_control(one_period_problem_spec,
                                                                     None,
                                                                     "mean_semivariance_theo",
                                                                     False)

    opt_mean_variance_control_theo = mean_risk_control.determine_opt_control(one_period_problem_spec,
                                                                     None,
                                                                     "mean_variance_theo",
                                                                     False)

    plotting_utils.plot_opt_controls_one_period(one_period_problem_spec,
                                                opt_mean_semivariance_controls_num,
                                                 opt_mean_semivariance_control_theo,
                                                 "mean_semivariance_num",
                                                 "mean_semivariance_theo")

    plotting_utils.plot_opt_controls_one_period(one_period_problem_spec,
                                                opt_mean_variance_controls_num,
                                                 opt_mean_variance_control_theo,
                                                 "mean_variance_num",
                                                 "mean_variance_theo")

    ## J vs V
    # -------------------------------------
    one_period_solver_spec["num_port_values_per_time"] = 100
    mean_risk_control.determine_opt_control(one_period_problem_spec,
                                   one_period_solver_spec,
                                   "mean_semivariance_num",
                                   True)

    mean_risk_control.determine_opt_control(one_period_problem_spec,
                                  one_period_solver_spec,
                                  "mean_variance_num",
                                  True)

    # =============================================================================
    # Multi-period
    # =============================================================================
    num_sim = 100000
    multi_period_solver_spec = {"control_step_size": 0.05,
                                "num_port_values_per_time": 200}

    # All multi-period problem spec names
    multi_period_problem_spec_names = [
        "multi_period_no_interest_no_shocks_small_gamma",
        "multi_period_interest_no_shocks_small_gamma",
        "multi_period_interest_no_shocks_large_gamma",
        "multi_period_interest_pos_shocks_small_gamma",
        "multi_period_interest_neg_shocks_small_gamma"
    ]

    all_performance_results = {}
    for multi_period_problem_spec_name in multi_period_problem_spec_names:
        multi_period_problem_spec = problem_specs.ProblemSpec(multi_period_problem_spec_name)
        if multi_period_problem_spec is not None:
            # Mean-semivariance optimal control numeric
            mean_semivariance_opt_control_num = mean_risk_control.determine_opt_control(multi_period_problem_spec,
                                                                                  multi_period_solver_spec,
                                                                                  "mean_semivariance_num",
                                                                                  False)

            plotting_utils.plot_opt_control(multi_period_problem_spec,
                                            mean_semivariance_opt_control_num,
                                            None,
                                            "mean_semivariance_num",
                                            None,
                                            None)

            mean_semivariance_num_performance = performance_metrics.evaluate_control(multi_period_problem_spec,
                                                 mean_semivariance_opt_control_num,
                                                 num_sim,
                                                 "mean_semivariance_num",
                                                 1)

            all_performance_results[(multi_period_problem_spec_name, "mean_semivariance_num")] = mean_semivariance_num_performance

            # Mean-variance optimal control numeric
            mean_variance_opt_control_num = mean_risk_control.determine_opt_control(multi_period_problem_spec,
                                                                 multi_period_solver_spec,
                                                                 "mean_variance_num",
                                                                 False)

            mean_variance_opt_control_theo = mean_risk_control.determine_opt_control(multi_period_problem_spec,
                                                                 multi_period_solver_spec,
                                                                 "mean_variance_theo",
                                                                 False)

            plotting_utils.plot_opt_control(multi_period_problem_spec,
                                            mean_variance_opt_control_num,
                                            mean_variance_opt_control_theo,
                                            "mean_variance_num",
                                            "mean_variance_theo",
                                            multi_period_problem_spec.get_params()["interest"]==0)

            mean_variance_num_performance = performance_metrics.evaluate_control(multi_period_problem_spec,
                                                 mean_variance_opt_control_theo,
                                                 num_sim,
                                                 "mean_variance_num",
                                                 1)

            all_performance_results[(multi_period_problem_spec_name, "mean_variance_num")] = mean_variance_num_performance

    # Round values to two decimals
    all_performance_results_rounded = {str(k): {inner_k: round(inner_v, 2) for inner_k, inner_v in v.items()} for k, v in all_performance_results.items()}
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    file_path = os.path.join(RESULTS_FOLDER, "all_performance_results.json")

    # Save the rounded dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(all_performance_results_rounded, json_file, indent=4)

if __name__ == "__main__":
    main()
