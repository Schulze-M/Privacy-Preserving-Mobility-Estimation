import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from tqdm import tqdm
import pandas as pd


def validate_coordinates(latitude: float, longitude: float) -> bool:
    '''
    Validate geographic coordinates.
    
    :param latitude: Latitude value (must be between -90 and 90).
    :param longitude: Longitude value (must be between -180 and 180).
    :return: True if valid, False otherwise.
    '''

    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return True
    return False

def test_cpp_results(traj: list, start: dict, prefixes: dict):
    '''
    Test if the start_nodes and prefixes computed by C++ are correct
    '''

    # prefix start list and prefix dictionary
    start_nodes: dict(tuple(float, float), int) = dict()
    prefix_dict: dict(tuple(float, float), dict(tuple, int)) = dict()

    # get the start nodes, with the number of times they appear as a start node
    for i in tqdm(traj):
        if tuple(i[0]) not in start_nodes:
            start_nodes[tuple(map(float, i[0]))] = 1
        else:
            start_nodes[tuple(map(float, i[0]))] += 1

    # get the prefixes of the trajectories, with the number of tirmes they appear as a pefix
    # the key is the prefix, the value is a of dictionary, with the key being the next element and the value the number of times it appears as the suffix
    for traj in tqdm(traj):
        for i in range(0, len(traj) -1):
            prefix = tuple(map(float, traj[i]))
            suffix = tuple(map(float, traj[i + 1]))
            if prefix not in prefix_dict:
                prefix_dict[prefix] = {suffix: 1}
            else:
                if suffix not in prefix_dict[prefix]:
                    prefix_dict[prefix].update({suffix: 1})
                else:
                    prefix_dict[prefix][suffix] += 1

    # test the results
    assert start_nodes == start, "The start dictionaries are not equal"
    print("The two start dictionaries are equal")

    assert prefix_dict == prefixes, "The prefix dictionaries are not equal"
    print("The two prefix dictionaries are equal")

def plot_eval_results(file: str, folder: str, dataset_name: str):
    '''
    Plot the evaluation results of the C++ implementation
    '''
    # read the data from the csv file
    df = pd.read_csv(file)
    
    # Now extract into NumPy arrays if you like:
    x             = df['eps'].values
    y_fitness     = df['mean_fit'].values
    y_fitness_std = df['std_fit'].values
    y_f1          = df['mean_f1'].values
    y_f1_std      = df['std_f1'].values
    y_precision   = df['mean_prec'].values
    y_precision_std= df['std_prec'].values
    y_recall      = df['mean_rec'].values
    y_recall_std  = df['std_rec'].values
    y_acc     = df['mean_acc'].values
    y__acc_std = df['std_acc'].values
    y_jaccard = df['mean_jaccard'].values
    y_jaccard_std = df['std_jaccard'].values
    y_fnr = df['mean_fnr'].values
    y_fnr_std = df['std_fnr'].values
    y_tp = df['mean_tp'].values
    y_tp_std = df['std_tp'].values
    y_fn = df['mean_fn'].values
    y_fn_std = df['std_fn'].values
    y_baseline = np.ones_like(x)

    # Set the color palette
    c = mpl.colormaps['tab10'].colors

    # Set the figure size
    # B x H
    fig1, ax1 = plt.subplots(figsize=(9, 4)) # (9, 4) for 2x1 layout

    # Plot all graphs
    ax1.plot(x, y_fitness, c=c[0], linewidth=2, label="Fitness", marker='x', markersize=12)
    ax1.plot(x, y_f1, c=c[2], linewidth=2, label="F1-Score", marker='.', markersize=12)
    ax1.plot(x, y_precision, c=c[1], linewidth=2, label="Precision", marker='.', markersize=12)
    ax1.plot(x, y_recall, c=c[4], linewidth=2, label="Recall", marker='.', markersize=12)
    ax1.plot(x, y_baseline, c="black", linewidth=2, label="Baseline", linestyle="--")

    # Plot standard deviation
    ax1.fill_between(x=x, y1=y_fitness - y_fitness_std, y2=y_fitness + y_fitness_std, alpha=0.1, facecolor=c[0])
    ax1.fill_between(x=x, y1=y_f1 - y_f1_std, y2=y_f1 + y_f1_std, alpha=0.125, facecolor=c[2])
    ax1.fill_between(x=x, y1=y_precision - y_precision_std, y2=y_precision + y_precision_std, alpha=0.125, facecolor=c[1])
    ax1.fill_between(x=x, y1=y_recall - y_recall_std, y2=y_recall + y_recall_std, alpha=0.125, facecolor=c[4])

    # Label the axes:
    ax1.set_xlabel("$\\varepsilon$", fontsize=14)
    ax1.set_ylabel("Eval Results", fontsize=14)

    # Limits of the axes:
    ax1.set_xlim(0.05, 1.05)
    ax1.set_ylim(0.0, 1.3)
    ax1.set_xticks([0.1, 0.2, 0.5, 0.8, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax1.grid(True)

    # Show the legend on the plot:
    ax1.legend(loc='best', ncol=3, fontsize=12, frameon=False, handlelength=1.5, handleheight=0.5, borderpad=0.5, labelspacing=0.5)
    
    # Save the figure
    fig1.savefig(f'{folder}Results_{dataset_name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    """
    --------------------------------------------------------------------------------------------
    Plot the second figure with accuracy, jaccard, and FNR
    --------------------------------------------------------------------------------------------
    """

    # Plot the second figure
    fig2, ax2 = plt.subplots(figsize=(9, 4)) # (9, 4) for 2x1 layout

    ax2.plot(x, y_acc, c=c[0], linewidth=2, label="Accuracy", marker='x', markersize=12)
    ax2.plot(x, y_jaccard, c=c[1], linewidth=2, label="Jaccard", marker='.', markersize=12)
    ax2.plot(x, y_fnr, c=c[2], linewidth=2, label="FNR", marker='.', markersize=12)
    ax2.plot(x, y_baseline, c='black', linewidth=2, label="Baseline", linestyle='--')
    ax2.plot(x, np.zeros_like(x), c='gray', linewidth=2, label="Baseline - FNR", linestyle='dashdot')

    ax2.fill_between(x=x, y1=y_acc - y__acc_std, y2=y_acc + y__acc_std, alpha=0.125, facecolor=c[0])
    ax2.fill_between(x=x, y1=y_jaccard - y_jaccard_std, y2=y_jaccard + y_jaccard_std, alpha=0.125, facecolor=c[1])
    ax2.fill_between(x=x, y1=y_fnr - y_fnr_std, y2=y_fnr + y_fnr_std, alpha=0.125, facecolor=c[2])

    ax2.set_xlabel("$\\varepsilon$", fontsize=14)
    ax2.set_ylabel("Eval Results", fontsize=14)

    # Limits of the axes:
    ax2.set_xlim(0.05, 1.05)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xticks([0.1, 0.2, 0.5, 0.8, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax2.grid(True)

    # Show the legend on the plot:
    ax2.legend(loc='best', ncol=3, fontsize=12, frameon=False, handlelength=1.5, handleheight=0.5, borderpad=0.5, labelspacing=0.5)
    
    # Save the figure
    fig2.savefig(f'{folder}Results_two_{dataset_name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    """
    --------------------------------------------------------------------------------------------
    Plot the third figure with TP, FP, TN, FN
    --------------------------------------------------------------------------------------------
    """
    if dataset_name == 'msnbc':
        # For MSNBC, we scale down the baseline to 11044.0
        baseline = np.full_like(x, 4260.0)
    elif dataset_name == 'paths_mil':
        # For MSNBC Mod, we scale down the baseline to 11044.0
        baseline = np.full_like(x, 11002.0)
    else:
        baseline = np.full_like(x, 9190.0)  # 9190 for 10,000 trajectories and 11044 for 1,000,000 trajectories

    # Plot the third figure
    fig3, ax3 = plt.subplots(figsize=(9, 4)) # (9, 4) for 2x1 layout

    ax3.plot(x, y_tp, c=c[0], linewidth=2, label="TP", marker='.', markersize=12)
    ax3.plot(x, y_fn, c=c[1], linewidth=2, label="FN", marker='x', markersize=12)
    ax3.plot(x, y_tp + y_fn, c=c[2], linewidth=2, label="TP + FN", marker='.', markersize=12)
    ax3.plot(x, np.zeros_like(x), c='red', linewidth=2, label="FP & TN", linestyle='-.')
    ax3.plot(x, baseline, c='black', linewidth=2, label="Baseline", linestyle='--')

    ax3.fill_between(x=x, y1=y_tp - y_tp_std, y2=y_tp + y_tp_std, alpha=0.125, facecolor=c[0])
    ax3.fill_between(x=x, y1=y_fn - y_fn_std, y2=y_fn + y_fn_std, alpha=0.125, facecolor=c[1])

    ax3.set_xlabel("$\\varepsilon$", fontsize=14)
    ax3.set_ylabel("Results - Confusion Matrix", fontsize=14)

    # Limits of the axes:
    ax3.set_xticks([0.1, 0.2, 0.5, 0.8, 1.0])
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax3.grid(True)

    # Show the legend on the plot:
    ax3.legend(loc='best', ncol=3, fontsize=12, frameon=False, handlelength=1.5, handleheight=0.5, borderpad=0.5, labelspacing=0.5)
    # Save the figure
    fig3.savefig(f'{folder}Results_conf_{dataset_name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig3)

def plot_error_results(file: str, folder: str, dataset_name: str):
    """
    For each epsilon in `file`, plot mean error vs subset_max_length (±std)
    with one connected curve for the implementation and the same baseline curve.
    """
    # Read data
    df = pd.read_csv(file)
    df_base = pd.read_csv('../results/errors_noDP_paths.csv')
    df_msnbc = pd.read_csv('../results/errors_msnbc.csv')
    df_msnbc_base = pd.read_csv('../results/errors_noDP_msnbc.csv')
    df_paths_mil = pd.read_csv('../results/errors_paths_mil.csv')
    df_paths_mil_base = pd.read_csv('../results/errors_noDP_paths_mil.csv')
    
    # Sort baseline by subset_max_length once
    df_base_sorted = df_base.sort_values('subset_max_length')
    x_bas = df_base_sorted['subset_max_length']
    y_bas = df_base_sorted['mean_error']

    # Sort base msnbc by subset_max_length
    df_msnbc_sorted = df_msnbc_base.sort_values('subset_max_length')
    x_msnbc_base = df_msnbc_sorted['subset_max_length']
    y_msnbc_base = df_msnbc_sorted['mean_error']  # Scale down by 3.0

    # Sort paths_mil by subset_max_length
    df_paths_mil_sorted = df_paths_mil_base.sort_values('subset_max_length')
    x_paths_mil_base = df_paths_mil_sorted['subset_max_length']
    y_paths_mil_base = df_paths_mil_sorted['mean_error']  # Scale down by 3.0

    # Get sorted list of eps values
    eps_values = sorted(df['eps'].unique(), key=float)

    for eps in eps_values:
        # Filter & sort for this epsilon
        df_eps = df[df['eps'] == eps].sort_values('subset_max_length')
        df_eps_msnbc = df_msnbc[df_msnbc['eps'] == eps].sort_values('subset_max_length')
        df_eps_paths_mil = df_paths_mil[df_paths_mil['eps'] == eps].sort_values('subset_max_length')

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot implementation curve
        x_imp = df_eps['subset_max_length']
        y_imp = df_eps['mean_error']
        ax.plot(x_imp, y_imp, marker='o', label='Berlin: 10,000', linestyle='-')
        ax.fill_between(x_imp, y_imp - df_eps['std_error'], y_imp + df_eps['std_error'], alpha=0.2, color='blue')

        # Plot the same baseline curve
        ax.plot(x_bas, y_bas, marker='D', linestyle='-.', label='Baseline Berlin: 10,000', color='limegreen')
        ax.fill_between(x_bas, y_bas - df_base_sorted['std_error'], y_bas + df_base_sorted['std_error'], alpha=0.2, color='limegreen')

        # Plot msnbc curve
        x_msnbc = df_eps_msnbc['subset_max_length']
        y_msnbc = df_eps_msnbc['mean_error']
        ax.plot(x_msnbc, y_msnbc, marker='*', linestyle=':', label='MSNBC', color='hotpink')
        ax.fill_between(x_msnbc, y_msnbc - df_eps_msnbc['std_error'], y_msnbc + df_eps_msnbc['std_error'], alpha=0.2, color='hotpink')

        # Plot baseline msnbc curve
        ax.plot(x_msnbc_base, y_msnbc_base, marker='x', linestyle='-.', label='Baseline MSNBC', color='blue')
        ax.fill_between(x_msnbc_base, y_msnbc_base - df_msnbc_sorted['std_error'], y_msnbc_base + df_msnbc_sorted['std_error'], alpha=0.2, color='blue')

        # Plot paths_mil curve
        x_paths_mil = df_eps_paths_mil['subset_max_length']
        y_paths_mil = df_eps_paths_mil['mean_error']
        ax.plot(x_paths_mil, y_paths_mil, marker='+', linestyle='--', label='Berlin: 1,000,000', color='purple')
        ax.fill_between(x_paths_mil, y_paths_mil - df_eps_paths_mil['std_error'], y_paths_mil + df_eps_paths_mil['std_error'], alpha=0.2, color='purple')

        # Plot baseline paths_mil curve
        ax.plot(x_paths_mil_base, y_paths_mil_base, marker='x', linestyle='-.', label='Baseline Berlin: 1,000,000', color='red')
        ax.fill_between(x_paths_mil_base, y_paths_mil_base - df_paths_mil_sorted['std_error'], y_paths_mil_base + df_paths_mil_sorted['std_error'], alpha=0.2, color='red')

        # Plot msnbc_mod curve
        # x_msnbc_mod = df_eps_msnbc_mod['subset_max_length']
        # y_msnbc_mod = df_eps_msnbc_mod['mean_error']/3.0
        # ax.plot(x_msnbc_mod, y_msnbc_mod, marker='^', linestyle='-.', label='MSNBC Mod')
        # ax.fill_between(x_msnbc_mod, y_msnbc_mod - df_eps_msnbc_mod['std_error']/3.0, y_msnbc_mod + df_eps_msnbc_mod['std_error']/3.0, alpha=0.2, color='green')

        # # Plot baseline msnbc_mod curve
        # ax.plot(x_msnbc_mod_base, y_msnbc_mod_base, marker='x', linestyle='--', label='Baseline MSNBC Mod')
        # ax.fill_between(x_msnbc_mod_base, y_msnbc_mod_base - df_msnbc_mod_sorted['std_error']/3.0, y_msnbc_mod_base + df_msnbc_mod_sorted['std_error']/3.0, alpha=0.2, color='green')

        # Labels & title
        ax.set_xlabel("Subset max length", fontsize=14)
        ax.set_ylabel("Mean Error", fontsize=14)
        plt.yticks(fontsize=12)
        # plt.xticks([0.1, 0.2, 0.5, 0.8, 1.0])
        plt.xticks([4, 8, 12, 16, 20], fontsize=12)
        ax.set_title(f"Mean Error vs Max Length (ε = {eps})", fontsize=14)
        ax.grid(True)
        # Save main figure without legend
        plot_path = f"{folder}Results_error_eps{eps}.pdf"
        fig.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        # Extract legend handles and labels
        handles, labels = ax.get_legend_handles_labels()

        # Save legend separately
        fig_legend = plt.figure(figsize=(6, 1.5))
        fig_legend.legend(handles, labels, loc='center', ncol=3, frameon=False)
        legend_path = f"{folder}legend.pdf"
        fig_legend.savefig(legend_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig_legend)