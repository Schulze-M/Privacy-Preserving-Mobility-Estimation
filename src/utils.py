from tqdm import tqdm

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