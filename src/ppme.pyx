# distutils: language = c++
# cython: language_level=3

from collections import defaultdict
from tqdm import tqdm
import numpy as np

cimport numpy as cnp
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp cimport bool


# Import your C++ header file
cdef extern from "./cpp_trie/include/main.h":
    # Declare C++ function and types for Cython
    cdef cppclass Coordinate:
        float data[2]

    ctypedef vector[Coordinate] Trajectory
    ctypedef unordered_map[Coordinate, unordered_map[Coordinate, float]] ResultMap

    ResultMap process_prefix(const vector[Trajectory]& trajectories)


# convert NumPy array to a fixed sized C++ array
cdef Coordinate np_to_coordinate(cnp.ndarray[cnp.float64_t, ndim=1] arr):
    cdef Coordinate coord

    coord.data[0] = arr[0]
    coord.data[1] = arr[1]
    return coord


# Convert NumPy arrays to C++ Trajectory
cdef Trajectory np_to_trajectory(cnp.ndarray[cnp.float64_t, ndim=2] arr):
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


# Convert C++ ResultMap back to Python dictionary
cdef result_map_to_dict(ResultMap result):
    py_result = {}
    cdef Coordinate key, inner_key
    cdef unordered_map[Coordinate, float] inner_map
    cdef pair[Coordinate, unordered_map[Coordinate, float]] outer_pair
    cdef pair[Coordinate, float] inner_pair

    # Iterate over the ResultMap
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

    cdef vector[Trajectory] trajectories = list_to_vector(py_trajectories)

    print("Done converting to C++ Trajectories")

    cdef ResultMap result = process_prefix(trajectories)

    print("Done processing prefixes")
    print("Converting ResultMap to Python dictionary...")

    return result_map_to_dict(result)