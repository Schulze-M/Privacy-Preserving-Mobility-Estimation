import argparse
import os
import pickle
from pprint import pprint

import numpy as np
from tqdm import tqdm
import time

import ppme

from datacstructure.trie import PrefixTree as trie
from utils import validate_coordinates
from collections import defaultdict

# Constants
DATA_FOLDER = '../datasets/'
DATA_CITY_NAME = 'Xian/'

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
    parser = argparse.ArgumentParser(description='Process the data from the dataset', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     usage='%(prog)s [options]',
                                     epilog='Enjoy the program! :)'
                                    )

    # Add the parser arguments
    parser.add_argument('traject_file', type=str, help='The name of the trajectory file')
    parser.add_argument('-a', '--attr_file', type=str, help='The name of the attribute file', default='attrs_Xian.pkl')

    # parse the arguments
    args = parser.parse_args()

    # create the path to the data files
    attr_path = os.path.join(DATA_FOLDER, DATA_CITY_NAME, args.attr_file)
    trajs_path = os.path.join(DATA_FOLDER, DATA_CITY_NAME, args.traject_file)

    # Load the attributes
    with open(attr_path, 'rb') as file:
        attrs = pickle.load(file)
    
    # load the trajectories
    with open(trajs_path, 'rb') as file:
        trajs = pickle.load(file)

    # validate the coordinates -> should be (latitude, longitude)
    is_normal_coords = validate_coordinates(trajs[0][0][0], trajs[0][0][1])

    # reverse the order of the trajectories for each list in the list, if the coordinates are not normal
    if not is_normal_coords:
        trajs = [np.array([[coord[1], coord[0]] for coord in array]) for array in trajs]

    # get the prefixes of the trajectories
    # TODO currently fixed at the first 10_000 trajectories, change this to the full dataset
    # TODO is done due to runtime -> all elements would take 16h to insert
    # get_prefixis(trajs[:10_000])

    # TODO start counts are needed to be computed
    start = time.time()
    test = ppme.process_prefix_py(trajs)

    print(f"Processing finished after: {(time.time() - start) / 60} minutes")
    print(test)


def get_prefixis(trajectories) -> trie:
    '''
    Get the prefixes of the trajectories
    '''

    # prefix start list and prefix dictionary
    start_nodes: dict(str, int) = dict()
    prefix_dict = defaultdict(lambda: defaultdict(int))

    # get the start nodes, with the number of times they appear as a start node
    start_nodes = np.unique([str(traj[0]) for traj in trajectories], return_counts=True)
    start_nodes = dict(zip(start_nodes[0], start_nodes[1]))

    # get the prefixes of the trajectories, with the number of times they appear as a prefix
    # the key is the prefix, the value is a list of dictionaries, with the key being the next element and the value the number of times it appears as the suffix
    for traj in tqdm(trajectories):
        for i in range(1, len(traj)):
            prefix = str(traj[:i])
            suffix = str(traj[i])
            if prefix not in prefix_dict:
                prefix_dict[prefix] = [{suffix: 1}]
            else:
                if suffix not in prefix_dict[prefix]:
                    prefix_dict[prefix].append({suffix: 1})
                else:
                    prefix_dict[prefix][suffix] += 1


    for traj in tqdm(trajectories):
        trie().insert(traj)

    tree = trie().insert(trajectories[1])

    print(tree)
    print(tree.search('[30.67032481 104.10906611]'))
    return trie


if __name__ == '__main__':
    load_data()