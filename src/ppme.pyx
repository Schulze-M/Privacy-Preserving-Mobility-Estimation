# distutils: language = c++
# cython: language_level=3

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time

cimport numpy as cnp
from cython cimport boundscheck
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
    cdef cppclass Coordinate:
        double data[2]

    cdef cppclass CountCoordinate:
        double data[3]

    ctypedef vector[Coordinate] Trajectory
    ctypedef unordered_map[Coordinate, vector[CountCoordinate]] PrefixMap
    ctypedef unordered_map[Coordinate, double] StartMap

    StartMap process_start(const vector[Trajectory]& trajectories)
    PrefixMap process_prefix(const vector[Trajectory]& trajectories)
    PrefixMap process_test(const Trajectory trajec, const StartMap start)

@boundscheck(False)
# convert NumPy array to a fixed sized C++ array
cdef Coordinate np_to_coordinate(double[::1] arr) noexcept nogil:
    cdef Coordinate coord

    coord.data[0] = arr[0]
    coord.data[1] = arr[1]
    return coord


# Convert NumPy arrays to C++ Trajectory
cdef Trajectory np_to_trajectory(cnp.ndarray[cnp.npy_float64, ndim=2] arr):
    cdef Trajectory traj
    cdef set[pair[double, double]] seen
    cdef pair[double, double] rounded_pair
    cdef double rounded_x, rounded_y
    cdef Coordinate coord
    cdef size_t i

    # Allocate memory for the Trajectory
    traj.reserve(arr.shape[0])

    # Iterate over the NumPy array
    # Check if the rounded values are already seen
    # Convert each NumPy array into a Coordinate and push it into the Trajectory
    for i in range(arr.shape[0]):
        coord = np_to_coordinate(arr[i])
        rounded_x = round(coord.data[0] * 10000) / 10000
        rounded_y = round(coord.data[1] * 10000) / 10000

        rounded_pair.first = rounded_x
        rounded_pair.second = rounded_y

        # Check if the rounded values are already seen
        if seen.find(rounded_pair) == seen.end():
            traj.push_back(coord)
            seen.insert(rounded_pair)

    # Shrink the Trajectory to fit the number of elements
    traj.shrink_to_fit()
    
    return traj

# Convert Python list of NumPy arrays to C++ vector of Trajectories
cdef vector[Trajectory] list_to_vector(list py_list):
    cdef vector[Trajectory] trajectories
    cdef cnp.ndarray[cnp.npy_float64, ndim=2] arr

    # Allocate memory for the vector
    trajectories.reserve(len(py_list))

    for arr in py_list:
        try:  # Check if the element is a NumPy array
            # Directly convert each NumPy array into a Trajectory
            trajectories.push_back(np_to_trajectory(arr))
        except Exception:
            # Handle the case where the element is not a NumPy array
            raise TypeError(f"Expected NumPy array, got {type(arr)} instead.\n")

    # Shrink the vector to fit the number of elements
    trajectories.shrink_to_fit()

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
    cdef CountCoordinate coord1, coord2, cnt
    cdef vector[CountCoordinate] inner_vector
    cdef pair[Coordinate, vector[CountCoordinate]] outer_pair
    cdef CountCoordinate inner_pair

    # Iterate over the PrefixMap
    for outer_pair in result:
        key = outer_pair.first
        inner_vector = outer_pair.second

        # Convert the outer key (Trajectory) to a hashable Python tuple
        hashable_key = tuple((key.data[0], key.data[1]))

        py_inner_list = []
        # Iterate over the inner map
        for inner_pair in inner_vector:
            py_coord_list = []

            # Extract elements from CountCoordinate's data array
            py_coord_list.append(float(inner_pair.data[0]))
            py_coord_list.append(float(inner_pair.data[1]))
            py_coord_list.append(float(inner_pair.data[2]))

            py_inner_list.append(py_coord_list)

        py_result[hashable_key] = py_inner_list

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