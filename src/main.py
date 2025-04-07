import argparse
import os
import pickle
from pprint import pprint

import numpy as np
from tqdm import tqdm
import ppme

from datacstructure.trie import PrefixTree as trie
from utils import validate_coordinates, test_cpp_results

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
    parser = argparse.ArgumentParser(description='Process the data from the dataset', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     usage='%(prog)s [options]',
                                     epilog='Enjoy the program! :)'
                                    )

    # Add the parser arguments
    parser.add_argument('traject_file', type=str, help='The name of the trajectory file')
    # parser.add_argument('-a', '--attr_file', type=str, help='The name of the attribute file', default='paths.pkl')
    parser.add_argument('-t', '--test', type=bool, help='Test the results of the C++ implementation', default=False)

    # parse the arguments
    args = parser.parse_args()

    # create the path to the data files
    # attr_path = os.path.join(DATA_FOLDER, DATA_CITY_NAME, args.attr_file)
    trajs_path = os.path.join(DATA_FOLDER, DATA_CITY_NAME, args.traject_file)

    # Load the attributes
    # with open(attr_path, 'rb') as file:
    #     attrs = pickle.load(file)
    
    # load the trajectories
    with open(trajs_path, 'rb') as file:
        trajs = pickle.load(file)

    # validate the coordinates -> should be (latitude, longitude)
    # is_normal_coords = validate_coordinates(trajs[0][0][0], trajs[0][0][1])

    # reverse the order of the trajectories for each list in the list, if the coordinates are not normal
    # if not is_normal_coords:
    #     trajs = [np.array([[coord[1], coord[0]] for coord in array]) for array in trajs]

    # create the trie
    print(len(trajs))
    start, prefix, triplet = ppme.process_prefix_py(trajs)
    # pprint(prefix, indent=4)
    print('Prefix length:', len(prefix))
    print('Start node length:', len(start))
    print('Triplet length:', len(triplet))

    with open("triplets.txt", "a") as f:
        for triplet, cnt in triplet.items():
            f.write(f"{triplet}: {cnt}\n")

    # pprint(triplet, indent=4)
    
    # test the results
    if args.test:
        test_cpp_results(trajs[:100_000], start, prefix)


if __name__ == '__main__':
    load_data()