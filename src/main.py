from datacstructure.trie import PrefixTree as trie

def load_data(data_path: str) -> trie:
    '''
    Load data from the data folder
    '''
    with open(data_path, 'r') as file:
        for line in file:
            trajectories = line.strip().split(',')
            trie.insert(trajectories)
    return trie


if __name__ == '__main__':
    load_data(input('Enter the path to the data file: \n'))