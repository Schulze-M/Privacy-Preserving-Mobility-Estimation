import argparse
import os
import pickle
from pprint import pprint

import ppme

from datacstructure.trie import PrefixTree as trie
from utils import validate_coordinates, test_cpp_results, plot_eval_results, plot_error_results

# Constants
DATA_FOLDER = '../datasets/'
DATA_CITY_NAME = ''

def load_data() -> trie:
    '''
    Load data from the data folder, attributes and trajectories.
    Attributes are the features of the trajectories, those contain:
        - Trip distance: Records the distance for each trip.
        - Trip time: The total time of the trip
        - Departure time: The time of trip start, with 5-min duration for each value.
        - Sample points: Total number of trajectory sampling points for each trip.
    
    Trajectories are the actual data, each line contains a list of trajectories, which are tuples of latitude and longitude.
    '''

    # Initialize the parser
    parser = argparse.ArgumentParser(description='''Process trajectory data, to create a trie structure. This trie can be use to synthesize data.
                                        This program is a part of the master thesis project: "Privacy-Presering Mobility Estimation".
                                        The goal of this project is to create a data structure that can be used to synthesize data, while preserving the privacy of the users.
                                        The data structure is a trie, which is a tree-like structure that can be used to store and search for data.
                                        The trie is used to store the trajectories of the users, and the data is synthesized by traversing the trie, using a (n-1)th order Markov Chain.''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     usage='%(prog)s <trajectory_file> [options]',
                                     add_help=True,
                                     allow_abbrev=True,
                                     conflict_handler='resolve',
                                     prog='Trajectory Data Synthezir',
                                     epilog='Enjoy the program! :)'
                                    )

    # Add the parser arguments
    parser.add_argument('trajectory_file', type=str, help='The name of the trajectory file')
    parser.add_argument('-t', '--test', type=bool, help='Test the results of the C++ implementation', default=False)
    parser.add_argument('-e', '--evaluate', help='Evaluate the results of the C++ implementation', default=False, action='store_true')
    parser.add_argument('-n', '--number', type=int, help='The number of evaluation rounds per epsilon', default=100)
    parser.add_argument('-p', '--plot', help='Plot the results of the C++ implementation', default=False, action='store_true')
    parser.add_argument('-eps', '--epsilon', type=float, help='The epsilon value to use for DP', default=0.1)
    parser.add_argument('--noDP', help='Create a Trie without using differntial privacy', default=False, action='store_true')

    # parse the arguments
    args = parser.parse_args()

    # create the path to the data files
    trajs_path = os.path.join(DATA_FOLDER, DATA_CITY_NAME, args.trajectory_file)
    
    # load the trajectories
    with open(trajs_path, 'rb') as file:
        trajs = pickle.load(file)

    # create the trie
    print(len(trajs))

    # Create Trie with rejection sampling
    trie = ppme.trie(trajs, args.trajectory_file.replace('.pkl', ''), args.epsilon, args.evaluate, args.number)

    # Create trie without DP
    if args.noDP:
        ppme.no_dp_trie(trajs, args.trajectory_file.replace('.pkl', ''), args.evaluate, args.number)

    pprint(trie)

    dataset_name = args.trajectory_file.replace('.pkl', '')

    if args.plot:
        plot_eval_results(f'../results/data_{dataset_name}.csv', '../results/', dataset_name)
        # plot_error_results(f'../results/errors_{dataset_name}.csv', '../results/', dataset_name)

if __name__ == '__main__':
    load_data()