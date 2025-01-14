# Description: This file contains the implementation of the Trie data structure

# TrieNode class
class TrieNode:
    def __init__(self, trajectory: str = '') -> None:
        '''
        Initialize a TrieNode object
        With a trajectory as str() and a empty dictionary of children
        '''
        self.trajectory = trajectory
        self.children = dict()
        
    def __str__(self):
        return self.trajectory

    def __repr__(self):
        return self.trajectory

    def __eq__(self, other):
        return self.trajectory == other.trajectory

    def __ne__(self, other):
        return not self.__eq__(other)


# Prefix Tree class
class PrefixTree:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, trajectories: list) -> None:
        '''
        Insert a list of trajectories into the prefix tree
        '''
        node = self.root
        for traject in trajectories:
            if traject not in node.children:
                node.add_child(traject)
            node = node.children[traject]
        node.is_end_of_word = True

    def add_child(self, trajectory: str):
        '''
        Add a child to the current node
        '''
        self.children[trajectory] = TrieNode(trajectory)

    def search(self, trajectory: str) -> bool:
        '''
        Search for a trajectory in the prefix tree
        '''
        node = self.root
        for traject in trajectory:
            if traject not in node.children:
                return False
            node = node.children[traject]
        return node.is_end_of_word
