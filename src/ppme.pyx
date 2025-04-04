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
    cdef cppclass Station:
        string data

    cdef cppclass CountStation:
        string suffix
        double count

    ctypedef vector[Station] Trajectory
    ctypedef unordered_map[Station, vector[CountStation]] PrefixMap
    ctypedef unordered_map[Station, double] StartMap

    StartMap process_start(const vector[Trajectory]& trajectories)
    PrefixMap process_prefix(const vector[Trajectory]& trajectories)
    PrefixMap process_test(const Trajectory trajec, const StartMap start)

# @boundscheck(False)
# convert NumPy array to a fixed sized C++ array
cdef Station py_to_Station(object s) noexcept:
    cdef Station coord

    # Encode the Python string to bytes (UTF-8)
    coord.data = s.encode("utf-8")
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

# Convert C++ StartMap back to Python dictionary
cdef dict start_map_to_dict(StartMap start_map):
    py_start_map = {}
    cdef Station key
    cdef double value
    cdef pair[Station, double] pair

    # Iterate over the StartMap
    for pair in start_map:
        key = pair.first
        value = pair.second

        # key.data is a C++ string; Cython converts it to a Python str automatically.
        py_start_map[key.data] = value

    return py_start_map

# Convert C++ PrefixMap back to Python dictionary
cdef result_map_to_dict(PrefixMap result):
    py_result = {}
    #cdef Station key
    cdef CountStation count_coord
    # cdef vector[CountStation] suffix_list
    # cdef pair[Station, vector[CountStation]] outer_pair

    # Iterate over the PrefixMap
    for outer_pair in result:
        key = outer_pair.first
        suffix_list = outer_pair.second

        prefix_str = key.data.decode("utf-8")

        py_inner_list = []

        # Iterate over the inner map
        for count_coord in suffix_list:
            # Extract elements from CountStation's data array
            # Create a list with the suffix string and count.
            suffix_str = count_coord.suffix.decode("utf-8")
            count = count_coord.count
            py_inner_list.append([suffix_str, count])

        py_result[prefix_str] = py_inner_list

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