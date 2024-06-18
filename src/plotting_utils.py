import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import seaborn as sns
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FOLDER = os.path.join(base_directory, "results")

def plot_control_vs_J(prolem_spec,
                      solver_spec,
                      control_u_and_J,
                      type_of_control):
    # Create the results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    filename = f"control_vs_J_{type_of_control}_{prolem_spec.get_short_name()}.png"
    # Join the results_folder path with the filename
    filepath = os.path.join(RESULTS_FOLDER, filename)

    color = '#1f77b4'

    # Determine the y-axis label based on the type of control
    if type_of_control == 'mean_semivariance_num':
        y_axis_label = 'Mean-Semivariance Reward'
    elif type_of_control == 'mean_variance_num':
        y_axis_label = 'Mean-Variance Reward'
    else:
        y_axis_label = 'Reward'  # Default label

    # Sort controls for a coherent line plot
    controls = sorted(control_u_and_J.keys())
    J_values = [control_u_and_J[u] for u in controls]

    # Create the figure with high resolution
    plt.figure(figsize=(10, 6), dpi=300)

    # Plotting control_u vs mean_risk_J
    plt.plot(controls, J_values, color=color, linestyle='-')

    # Customize the plot with labels
    plt.xlabel('Control $u$', fontsize=14)
    plt.ylabel(y_axis_label, fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust figure layout to ensure everything fits without a title
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the figure to the specified filepath
    plt.savefig(filepath, dpi=300)
    plt.close()  # Close the plot to free up memory
    logger.info(f"Plot saved to {filepath}")

def plot_term_port_vals_hist(problem_spec,
                                   type_of_control,
                                   term_port_vals):
    # Create the results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    filename = f"term_port_vals_{type_of_control}_{problem_spec.get_short_name()}.png"
    filepath = os.path.join(RESULTS_FOLDER, filename)  # Construct the file path

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")  # Set the seaborn style for the plot

    # Calculate the density
    density = gaussian_kde(term_port_vals)
    xs = np.linspace(min(term_port_vals), max(term_port_vals), 200)
    density._compute_covariance()

    # Calculate mean, semivariance, and variance
    mean_term_port_val = np.mean(term_port_vals)
    semivariance = np.mean([(x - mean_term_port_val) ** 2 if x < mean_term_port_val else 0 for x in term_port_vals])
    variance = np.var(term_port_vals)

    # Plot histogram with relative frequency and density curve
    sns.histplot(term_port_vals, bins=50, stat='density', alpha=0.75, color='skyblue', label='Histogram')
    plt.plot(xs, density(xs), 'r-', label='Density Curve')  # Plot the density curve

    # Highlight the downside risk area
    downside_xs = xs[xs < mean_term_port_val]
    plt.fill_between(downside_xs, density(downside_xs), color='red', alpha=0.3, label='Downside Risk Area')

    # Annotations positioned slightly more to the left from the mean
    text_position_x = mean_term_port_val - (max(term_port_vals) - min(term_port_vals)) * 0.25  # Adjust text position

    plt.axvline(mean_term_port_val, color='k', linestyle='dashed', linewidth=1)
    plt.text(text_position_x, plt.ylim()[1] * 0.95, f'Mean: {mean_term_port_val:.2f}', horizontalalignment='left')
    plt.text(text_position_x, plt.ylim()[1] * 0.9, f'Semivariance: {semivariance:.2f}', horizontalalignment='left')
    plt.text(text_position_x, plt.ylim()[1] * 0.85, f'Variance: {variance:.2f}', horizontalalignment='left')

    plt.xlabel('Terminal Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()

    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_opt_control(problem_spec,
                     opt_control_num,
                     opt_control_theo,
                     type_of_control_num,
                     type_of_control_theo,
                     opt_control_theo_is_constant=False):
    # Create the results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    filename = f"opt_control_{type_of_control_num}_{type_of_control_theo}_{problem_spec.get_short_name()}.png"
    filepath = os.path.join(RESULTS_FOLDER, filename)  # Construct the file path

    # Organize numerical data by time
    data_by_time_num = defaultdict(list)
    for (time, portfolio_value), opt_control in opt_control_num.items():
        data_by_time_num[time].append((portfolio_value, opt_control))

    plt.figure(figsize=(10, 6), dpi=300)

    # Define a color palette
    professional_colors = ['darkblue', 'darkgreen', 'darkred', 'goldenrod', 'purple', 'sienna', 'grey']
    color_map = {}
    color_idx = 0

    for time, values in sorted(data_by_time_num.items()):
        sorted_values = sorted(values)
        portfolio_values, opt_controls = zip(*sorted_values)

        if time not in color_map:
            color_map[time] = professional_colors[color_idx % len(professional_colors)]
            color_idx += 1

        if len(portfolio_values) > 1:
            plt.plot(portfolio_values, opt_controls, label=f'Time={time} (num)', color=color_map[time])
        else:
            plt.scatter(portfolio_values, opt_controls, label=f'Time={time} (num)', color=color_map[time], marker='o', s=50)

    # Organize theoretical data by time
    if opt_control_theo:
        data_by_time_theo = defaultdict(list)
        for (time, portfolio_value), opt_control in opt_control_theo.items():
            data_by_time_theo[time].append((portfolio_value, opt_control))

        for time, values in sorted(data_by_time_theo.items()):
            sorted_values = sorted(values)
            portfolio_values, opt_controls = zip(*sorted_values)

            color = 'black' if opt_control_theo_is_constant else color_map.get(time, 'grey')

            if opt_control_theo_is_constant and time == 4:
                label_theo = 'Time=0,1,2,3,4 (theo)'
                plt.plot(portfolio_values, opt_controls, label=label_theo, linestyle='--', color=color)
            elif not opt_control_theo_is_constant:
                label_theo = f'Time={time} (theo)'
                if time == 0:
                    # For theoretical values at time=0, use a circle with the color assigned to time=0 in the num case
                    plt.scatter(portfolio_values, opt_controls, edgecolors=color_map[time], facecolors='none', marker='o', linewidths=2, s=100, label=label_theo)
                else:
                    plt.plot(portfolio_values, opt_controls, label=label_theo, linestyle='--', color=color_map[time])

    plt.xlabel('Portfolio Value')
    plt.ylabel('Equilibrium Control')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_opt_controls_one_period(problem_spec,
                                 opt_mean_risk_controls_num,
                                 opt_mean_risk_control_theo,
                                 type_of_control_num,
                                 type_of_control_theo):
    # Create the results folder if it doesn't exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    filename = f'opt_controls_{type_of_control_num}_{type_of_control_theo}_{problem_spec.get_short_name()}.png'
    filepath = os.path.join(RESULTS_FOLDER, filename)  # Construct the file path

    # Determine the label based on the type of control
    if type_of_control_num == 'mean_semivariance_num':
        opt_mean_risk_controls_num_label = 'Mean-Semivariance Control (num)'
    elif type_of_control_num == 'mean_variance_num':
        opt_mean_risk_controls_num_label = 'Mean-Variance Control (num)'
    else:
        raise Exception("Unknown type_of_control_num")

    if type_of_control_theo == 'mean_semivariance_theo':
        opt_mean_risk_controls_theo_label = 'Mean-Semivariance Control (theo)'
    elif type_of_control_theo == 'mean_variance_theo':  # Corrected to check type_of_control_theo
        opt_mean_risk_controls_theo_label = 'Mean-Variance Control (theo)'
    else:
        raise Exception("Unknown type_of_control_theo")

    color = '#1f77b4'  # A single color for both num and theo for consistency

    # Create the figure with high resolution
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot numerical optimal controls
    num_port_values_per_time_list = list(opt_mean_risk_controls_num.keys())
    plt.plot(num_port_values_per_time_list, list(opt_mean_risk_controls_num.values()),
             label=opt_mean_risk_controls_num_label, marker='o', linestyle='-', color=color)

    # Plot theoretical optimal controls with dashed lines in the same color
    plt.axhline(y=list(opt_mean_risk_control_theo.values())[0], label=opt_mean_risk_controls_theo_label, linestyle='--',
                color=color)

    # Customize the plot with labels, title, and a grid
    plt.xlabel('Number of Portfolio Values per Time', fontsize=14)
    plt.ylabel('Control Value', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust figure layout to make room for the legend below
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Place the legend below the figure, in 2x2 columns, with a smaller font size for a professional layout
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=False, shadow=False, ncol=2, fontsize=12)

    # Save the figure with a specified dpi in the results folder
    plt.savefig(filepath, dpi=300)
    plt.close()
