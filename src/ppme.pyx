# distutils: language = c++
# cython: language_level=3

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time

cimport numpy as cnp
from cython cimport boundscheck
from cpython.unicode cimport PyUnicode_FromString
from libc.math cimport round
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.set cimport set


# Import C++ header file
cdef extern from "./cpp_trie/include/main.h":
    # Declare C++ function and types for Cython
    cdef cppclass Station:
        string data

    cdef cppclass CountStation:
        string suffix
        double count

    cdef cppclass Triplet:
        string s1
        string s2
        string s3

    ctypedef vector[Station] Trajectory
    ctypedef unordered_map[Station, vector[CountStation]] PrefixMap
    ctypedef unordered_map[Station, double] StartMap
    ctypedef unordered_map[Triplet, double] TripletMap

    StartMap process_start(const vector[Trajectory]& trajectories)
    PrefixMap process_prefix(const vector[Trajectory]& trajectories)
    TripletMap create_triplet_map(const vector[Trajectory]& trajectories)
    pair[double, double] process_triplets(TripletMap triplet, double epsilon, const vector[Trajectory]& trajectories)
    PrefixMap process_test(const Trajectory trajec, const StartMap start)

cdef extern from "./cpp_trie/include/trie.h":
    cdef cppclass TrieNode:
        double count
        unordered_map[Station, TrieNode*] children

    cdef cppclass Trie:
        Trie() except +
        TrieNode* root
        void insertTrajectory(const vector[Station]& trajectory, double countValue)
     
# @boundscheck(False)
# convert NumPy array to a fixed sized C++ array
cdef Station py_to_Station(object s) noexcept:
    cdef Station coord
    if s is None:
        raise ValueError("Found None value in trajectory; station names must be valid strings.")
    if not isinstance(s, str):
        raise ValueError(f"Expected a string for station name, got {type(s)} instead.")
    
    cdef bytes encoded = s.encode("utf-8")
    if not encoded:
        raise ValueError("Empty station name encountered.")
    
    coord.data = encoded
    return coord

# Convert NumPy arrays to C++ Trajectory
cdef Trajectory np_to_trajectory(cnp.ndarray[object, ndim=1] arr):
    cdef Trajectory traj
    cdef Station coord
    cdef size_t i

    # Allocate memory for the Trajectory
    traj.reserve(arr.shape[0])

    # For each element in the 1D array, create a Station.
    for i in range(arr.shape[0]):
        coord = py_to_Station(arr[i])  # assign the scalar value
        traj.push_back(coord)

    # Shrink the Trajectory to fit the number of elements
    traj.shrink_to_fit()
    
    return traj

# Convert Python list of NumPy arrays to C++ vector of Trajectories
cdef vector[Trajectory] list_to_vector(list py_list):
    cdef vector[Trajectory] trajectories
    cdef cnp.ndarray[object, ndim=1] arr

    # Allocate memory for the vector
    trajectories.reserve(len(py_list))

    for item in py_list:
        try:  # Check if the element is a NumPy array
            # Directly convert each NumPy array into a Trajectory
            # Ensure the array is contiguous and uses object dtype.
            arr = np.ascontiguousarray(item, dtype=object)
            trajectories.push_back(np_to_trajectory(arr))
        except Exception:
            # Handle the case where the element is not a NumPy array
            raise TypeError(f"Expected NumPy array, got {type(arr)} instead.\n")

    # Shrink the vector to fit the number of elements
    trajectories.shrink_to_fit()

    return trajectories

cdef dict triplet_map_to_dict(TripletMap triplet_map):
    py_triplet_map = {}
    cdef Triplet key
    cdef double value
    cdef pair[Triplet, double] pair

    # Iterate over the TripletMap
    for pair in triplet_map:
        key = pair.first
        value = pair.second

        first_str = (<string>key.s1.data).decode()
        second_str = (<string>key.s2.data).decode()
        third_str = (<string>key.s3.data).decode()

        # Use Python strings as keys in the dictionary
        py_triplet_map[(first_str, second_str, third_str)] = value

    # TODO - Remove this test
    test = list()
    for k, v in py_triplet_map.items():
        if v < 100.0:
            test.append(k)
    print(f"Triplet map size: {len(test)}")

    return py_triplet_map

# Main function to process prefixes
def process_prefix_py(list py_trajectories):
    print("Processing prefixes...")
    start = time.time()
    cdef vector[Trajectory] trajectories = list_to_vector(py_trajectories)
    end = time.time()
    print("Done converting to C++ Trajectories")
    print(f"Time taken to convert: {end - start} seconds\n")

    # Generate tripletmap
    start = time.time()
    cdef TripletMap triplet_map = create_triplet_map(trajectories)
    end = time.time()

    print(f"Done creating triplet map in {end - start} seconds\n")

    # Process triplet map
    start = time.time()
    print("Begin Trie generation...")

    result_list = []
    Eps = [0.1, 0.2, 0.5, 0.8, 1.0]
    for eps in Eps:
        fit = []
        f1 = []
        for i in range(100):
            # cdef TripletMap triplet_map = process_triplets(trajectories)
            result = process_triplets(triplet=triplet_map, epsilon=eps, trajectories=trajectories)
            fit.append(result.first)
            f1.append(result.second)
            # print("Done")
        # get mean of results
        # print("Finishes one EPSILON!!!")
        mean = np.mean(fit)
        mean_f1 = np.mean(f1)
        # get std of results
        std = np.std(fit)
        std_f1 = np.std(f1)

        with open("eval_fit.txt", "a") as f:
            f.write(f"EPSILON: {eps}, mean: {mean}, std: {std}\n")
        with open("eval_f1.txt", "a") as f:
            f.write(f"EPSILON: {eps}, mean: {mean_f1}, std: {std_f1}\n")

        # Create a tuple with the results
        # result_list.append((eps, mean, std))

    end = time.time()

    print(f"Done processing triplet map in {(end - start) / 60} minutes\n")

    start = time.time()
    #py_triplet_map = triplet_map_to_dict(triplet_map)
    end = time.time()

    return result_list

#cdef object _node_to_py(TrieNode* node) except *:
#    """
#    Recursively convert a C++ TrieNode* into a Python dict
#      { 'count': <float>, 'children': { station_name: <child_dict>, … } }
#    """
#    cdef dict py_node = {
#        'count': node.count,
#       'children': {}
#    }
#    cdef pair[Station, TrieNode*] entry
#    for entry in node.children:
#        # unpack
#        st    = entry.first
#        child = entry.second
#        # decode std::string bytes to Python str
#        name = (<string>st.data).decode('utf-8')
#        py_node['children'][name] = _node_to_py(child)
#    return py_node
#
#def trie_to_dict(Trie& trie) -> dict:
#    """
#    Top-level: run the recursive converter on trie.root
#    """
#    if trie.root == NULL:
#        raise ValueError("Trie has no root")
#    return _node_to_py(trie.root)
#
# def draw_trie(Trie& trie, filename: str = None, view: bool = False):
#     """
#     Render a C++ Trie as a Graphviz Digraph.
#     
#     - trie: your wrapped C++ Trie instance
#     - filename: if given, the .gv (or .pdf/.png) file to write & render
#     - view: whether to open the viewer after rendering
#     Returns a graphviz.Digraph object.
#     """
#     from graphviz import Digraph
#     # Convert to nested dict
#     py_root = trie_to_dict(trie)
#     
#     dot = Digraph(comment="Trie")
#     dot.node('root', label=f"root\ncount={py_root['count']:.1f}")
#     
#     def _recurse(node: dict, node_id: str):
#         for station, child in node['children'].items():
#             # create a unique id per node
#             child_id = f"{node_id}_{station}"
#             # Graphviz node label shows station name + its count
#             dot.node(child_id, label=f"{station}\ncount={child['count']:.1f}")
#             dot.edge(node_id, child_id)
#             _recurse(child, child_id)
#     
#     _recurse(py_root, 'root')
#     
#     if filename is not None:
#         dot.render(filename, view=view, cleanup=True)
#     return dot