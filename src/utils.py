import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
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

def plot_eval_results(file: str, folder: str):
    '''
    Plot the evaluation results of the C++ implementation
    '''

    # Set the color palette
    c = mpl.colormaps['tab10'].colors

    # Set the figure size
    # B x H
    plt.figure(figsize=(9, 4))

    # read the data from the csv file
    df = pd.read_csv(file)
    
    # Now extract into NumPy arrays if you like:
    x             = df['eps'].values
    y_fitness     = df['mean_fit'].values
    y_fitness_std = df['std_fit'].values
    y_f1          = df['mean_f1'].values
    y_f1_std      = df['std_f1'].values

    # Plot all graphs
    plt.plot(x, y_fitness, c=c[0], linewidth=2, label="Fitness", marker='.', markersize=12)
    plt.plot(x, y_f1, c=c[2], linewidth=2, label="F1-Score", marker='.', markersize=12)

    # Plot standard deviation
    plt.fill_between(x=x, y1=y_fitness - y_fitness_std, y2=y_fitness + y_fitness_std, alpha=0.1, facecolor=c[0])
    plt.fill_between(x=x, y1=y_f1 - y_f1_std, y2=y_f1 + y_f1_std, alpha=0.125, facecolor=c[2])

    # Label the axes:
    plt.xlabel("$\\varepsilon$", fontsize=14)
    plt.ylabel("Eval Results", fontsize=14)

    # Limits of the axes:
    plt.xlim(0.05, 1.05)
    plt.ylim(0.0, 1.15)
    plt.xticks([0.1, 0.2, 0.5, 0.8, 1.0])
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12) # rotation=-45
    
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    # Show the legend on the plot:
    plt.legend(loc='best', ncol=7, fontsize=12, frameon=False, handlelength=1.5, handleheight=0.5, borderpad=0.5, labelspacing=0.5)
    
    # Save the figure
    plt.savefig(f'{folder}Results.pdf', bbox_inches='tight', pad_inches=0)

def plot_error_results(file: str, folder: str):
    '''
    Plot the error results of the C++ implementation
    '''

    # Set the color palette
    c = mpl.colormaps['tab10'].colors

    # Set the figure size
    # B x H
    plt.figure(figsize=(9, 4))

    # read the data from the csv file
    df = pd.read_csv(file)
    
    # Plot
    plt.figure()
    for length, group in df.groupby('subset_max_length'):
        x = group['eps']
        y = group['mean_error']
        err = group['std_error']
        # plot mean line
        plt.plot(x, y, marker='o', label=f'max length = {length}')
        # fill ± std deviation
        plt.fill_between(x, y - err, y + err, alpha=0.3)

    # Plot standard deviation
    # plt.fill_between(x=x, y1=y_fitness - y_fitness_std, y2=y_fitness + y_fitness_std, alpha=0.1, facecolor=c[0])
    # plt.fill_between(x=x, y1=y_f1 - y_f1_std, y2=y_f1 + y_f1_std, alpha=0.125, facecolor=c[2])

    # Label the axes:
    plt.xlabel("$\\varepsilon$", fontsize=14)
    plt.ylabel('Mean Error')
    plt.title('Mean Error vs ε with ±1σ Bands')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f'{folder}Results_erros.pdf', bbox_inches='tight', pad_inches=0)