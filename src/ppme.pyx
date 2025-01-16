# distutils: language = c++
# cython: language_level=3

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time

cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp cimport bool


# Import C++ header file
cdef extern from "./cpp_trie/include/main.h":
    # Declare C++ function and types for Cython
    cdef cppclass Coordinate:
        double data[2]

    ctypedef vector[Coordinate] Trajectory
    ctypedef unordered_map[Coordinate, unordered_map[Coordinate, double]] PrefixMap
    ctypedef unordered_map[Coordinate, double] StartMap

    StartMap process_start(const vector[Trajectory]& trajectories)
    PrefixMap process_prefix(const vector[Trajectory]& trajectories)


# convert NumPy array to a fixed sized C++ array
cdef Coordinate np_to_coordinate(cnp.ndarray[cnp.npy_float64, ndim=1] arr):
    cdef Coordinate coord

    coord.data[0] = arr[0]
    coord.data[1] = arr[1]
    return coord


# Convert NumPy arrays to C++ Trajectory
cdef Trajectory np_to_trajectory(cnp.ndarray[cnp.npy_float64, ndim=2] arr):
    cdef Trajectory traj

    for coord in arr:
        traj.push_back(np_to_coordinate(coord)) # Push the converted array into the Trajectory
    return traj

# Convert Python list of NumPy arrays to C++ vector of Trajectories
cdef vector[Trajectory] list_to_vector(list py_list):
    cdef vector[Trajectory] trajectories

    for arr in py_list:
        if isinstance(arr, cnp.ndarray):  # Check if the element is a NumPy array
            # Directly convert each NumPy array into a Trajectory
            trajectories.push_back(np_to_trajectory(arr))  
        else:
            # Handle the case where the element is not a NumPy array
            raise TypeError(f"Expected NumPy array, got {type(arr)} instead.")

    return trajectories

# Convert C++ StartMap back to Python dictionary
cdef dict start_map_to_dict(StartMap start_map):
    py_start_map = {}
    cdef Coordinate key
    cdef double value
    cdef pair[Coordinate, double] pair

    # Iterate over the StartMap
    for pair in start_map:
        key = pair.first
        value = pair.second

        # Convert the key (Trajectory) to a hashable Python tuple
        hashable_key = tuple((key.data[0], key.data[1]))
        py_start_map[hashable_key] = value

    return py_start_map

# Convert C++ PrefixMap back to Python dictionary
cdef result_map_to_dict(PrefixMap result):
    py_result = {}
    cdef Coordinate key, inner_key
    cdef unordered_map[Coordinate, double] inner_map
    cdef pair[Coordinate, unordered_map[Coordinate, double]] outer_pair
    cdef pair[Coordinate, double] inner_pair

    # Iterate over the PrefixMap
    for outer_pair in result:
        key = outer_pair.first
        inner_map = outer_pair.second

        # Convert the outer key (Trajectory) to a hashable Python tuple
        hashable_key = tuple((key.data[0], key.data[1]))

        py_inner_map = {}
        # Iterate over the inner map
        for inner_pair in inner_map:
            inner_key = inner_pair.first
            value = inner_pair.second

            # Convert the inner key (Trajectory) to a hashable Python tuple
            hashable_inner_key = tuple((inner_key.data[0], inner_key.data[1]))
            py_inner_map[hashable_inner_key] = value

        py_result[hashable_key] = py_inner_map

    return py_result

# Main function to process prefixes
def process_prefix_py(list py_trajectories):
    print("Processing prefixes...")
    start = time.time()
    cdef vector[Trajectory] trajectories = list_to_vector(py_trajectories)
    end = time.time()
    print("Done converting to C++ Trajectories")
    print(f"Time taken to convert: {end - start} seconds\n")

    print("Begin to compute start map...")
    start = time.time()
    cdef StartMap start_map = process_start(trajectories)
    end = time.time()
    print("Finished computing start map")
    print(f"Time taken to compute start map: {end - start} seconds\n")

    start = time.time()
    print("Begin processing prefixes...")
    cdef PrefixMap result = process_prefix(trajectories)
    end = time.time()
    print("Done processing prefixes")
    print(f"Time taken to compute: {(end - start) / 60} minutes\n")

    print("Converting StartMap to Python dictionary...")
    start = time.time()
    py_start_map = start_map_to_dict(start_map)
    end = time.time()
    print("Done converting to Python dictionary")
    print(f"Time taken to convert to Python object: {end - start} seconds\n")

    print("Converting PrefixMap to Python dictionary...")
    start = time.time()
    py_prefix_map = result_map_to_dict(result)
    end = time.time()
    print("Done converting to Python dictionary")
    print(f"Time taken to convert to Python object: {(end - start) / 60} minutes\n")

    return py_start_map, py_prefix_map