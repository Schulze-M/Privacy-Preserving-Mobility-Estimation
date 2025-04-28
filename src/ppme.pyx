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
        string first
        string second
        string third

    ctypedef vector[Station] Trajectory
    ctypedef unordered_map[Station, vector[CountStation]] PrefixMap
    ctypedef unordered_map[Station, double] StartMap
    ctypedef unordered_map[Triplet, double] TripletMap

    StartMap process_start(const vector[Trajectory]& trajectories)
    PrefixMap process_prefix(const vector[Trajectory]& trajectories)
    double process_triplets(const vector[Trajectory]& trajectories, double epsilon)
    PrefixMap process_test(const Trajectory trajec, const StartMap start)

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
    cdef CountStation count_coord

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

cdef dict triplet_map_to_dict(TripletMap triplet_map):
    py_triplet_map = {}
    cdef Triplet key
    cdef double value
    cdef pair[Triplet, double] pair

    # Iterate over the TripletMap
    for pair in triplet_map:
        key = pair.first
        value = pair.second

        first_str = (<string>key.first.data).decode()
        second_str = (<string>key.second.data).decode()
        third_str = (<string>key.third.data).decode()

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

    #print("Begin to compute start map...")
    #start = time.time()
    #cdef StartMap start_map = process_start(trajectories)
    #end = time.time()
    #print("Finished computing start map")
    #print(f"Time taken to compute start map: {end - start} seconds\n")
#
    #start = time.time()
    #print("Begin processing prefixes...")
    #cdef PrefixMap result = process_prefix(trajectories)
    #end = time.time()
    #print("Done processing prefixes")
    #print(f"Time taken to compute: {(end - start) / 60} minutes\n")

    # Process triplet map
    start = time.time()
    print("Begin processing triplet map...")

    result_list = []
    Eps = [0.1, 0.2, 0.5, 0.8, 1.0]
    for eps in Eps:
        results = []
        for i in range(100):
            # cdef TripletMap triplet_map = process_triplets(trajectories)
            fit = process_triplets(trajectories, epsilon=eps)
            results.append(fit)
        # get mean of results
        mean = np.mean(results)
        result_list.append(mean)

    end = time.time()
    start = time.time()
    #py_start_map = start_map_to_dict(start_map)
    end = time.time()
    start = time.time()
    #py_prefix_map = result_map_to_dict(result)
    end = time.time()
    start = time.time()
    #py_triplet_map = triplet_map_to_dict(triplet_map)
    end = time.time()

    return result_list