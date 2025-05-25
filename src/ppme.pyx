# distutils: language = c++
# cython: language_level=3
import os
import csv

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
from libc.stddef cimport size_t


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

    cdef cppclass EvalResult:
        double fit
        double f1
        double precision
        double recall
        double tp
        double fp
        double fn
        double tn
        vector[double] errors
        double accuracy
        double jaccard
        double fnr

    ctypedef vector[Station] Trajectory
    ctypedef unordered_map[Station, vector[CountStation]] PrefixMap
    ctypedef unordered_map[Station, double] StartMap
    ctypedef unordered_map[Triplet, double] TripletMap

    TripletMap create_triplet_map(const vector[Trajectory]& trajectories)
    bool create_trie(TripletMap triplet, double epsilon, const vector[Trajectory]& trajectories)
    bool create_trie_no_noise(TripletMap triplet, const vector[Trajectory]& trajectories)
    EvalResult evaluate(TripletMap triplet, double epsilon, const vector[Trajectory]& trajectories)
    EvalResult evaluate_no_noise(TripletMap triplet, const vector[Trajectory]& trajectories)
    vector[double] evalErrors(TripletMap triplet, double epsilon, const vector[Trajectory]& trajectories)
    vector[double] evalErrors_noDP(TripletMap triplet, const vector[Trajectory]& trajectories)
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
    cdef size_t i, n

    n = <size_t> arr.shape[0]
    # Allocate memory for the Trajectory
    traj.reserve(n)

    # For each element in the 1D array, create a Station.
    for i in range(n):
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

# Main function to process prefixes
def trie(list py_trajectories, str dataname, eps=0.1, do_eval=False, num_evals=100):
    """
    Process the trajectories and build a trie.
    """
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

    # cdef TripletMap triplet_map = process_triplets(trajectories)
    trie_bool = create_trie(triplet=triplet_map, epsilon=eps, trajectories=trajectories)

    erros = evalErrors(triplet=triplet_map, epsilon=eps, trajectories=trajectories)
    nonDP_erros = evalErrors_noDP(triplet=triplet_map, trajectories=trajectories)
    print(erros)
    print(nonDP_erros)

    end = time.time()

    print(f"Done Trie generation in {(end - start) / 60} minutes\n")

    if do_eval:
        evaluate_trie(triplet_map, trajectories, num_evals, dataname)

    return trie_bool

"""
Create a without differential privacy
"""
def no_dp_trie(list py_trajectories, str dataname, do_eval=False, num_evals=100):
    """
    Process the trajectories and build a trie.
    """
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

    trie_bool = create_trie_no_noise(triplet=triplet_map, trajectories=trajectories)

    end = time.time()

    print(f"Done Trie generation in {(end - start) / 60} minutes\n")

    """
    Evaluate the non DP Trie.
    """
    if do_eval:
        eval_no_dp(triplet_map, trajectories, num_evals, dataname)

    return trie_bool

"""
Function to evaluate the Rejection Sampling Trie generation
"""
cdef void evaluate_trie(TripletMap triplet, vector[Trajectory] traject, int num_evals, str dataname):
    # Ensure the results directory exists
    os.makedirs("../results", exist_ok=True)

    print ("Begin evaluation...")
    start = time.time()

    # Initialize CSV file
    with open(f'../results/data_{dataname}.csv', mode='w', newline='', encoding='utf-8') as df:
        writer = csv.writer(df)
        writer.writerow(['eps','mean_fit','std_fit','mean_f1','std_f1', 
        'mean_prec', 'std_prec', 'mean_rec', 'std_rec', "mean_tp", "std_tp",
        'mean_fp', 'std_fp', 'mean_fn', 'std_fn', 'mean_tn', 'std_tn',
        'mean_acc', 'std_acc', 'mean_jaccard', 'std_jaccard', 'mean_fnr', 'std_fnr',
        'num_evals', 'max_grams'])

    with open(f'../results/errors_{dataname}.csv', mode='w', newline='', encoding='utf-8') as ef:
        writer = csv.writer(ef)
        writer.writerow([
            'eps','max_query_length','subset_id',
            'subset_max_length','mean_error','std_error','num_evals'
        ])

    Eps = [0.1, 0.2, 0.5, 0.8, 1.0]
    for eps in Eps:
        fit = []
        f1 = []
        precision = []
        recall = []
        tp = []
        fp = []
        tn = []
        fn = []
        errors = []
        accuracy = []
        jaccard = []
        fnr = []
        # Evaluate the triplet map
        for i in range(num_evals):
            result = evaluate(triplet=triplet, epsilon=eps, trajectories=traject, )
            fit.append(result.fit)
            f1.append(result.f1)
            precision.append(result.precision)
            recall.append(result.recall)
            tp.append(result.tp)
            fp.append(result.fp)
            tn.append(result.tn)
            fn.append(result.fn)
            errors.append([result.errors[j] *3.0 for j in range(5)])
            accuracy.append(result.accuracy)
            jaccard.append(result.jaccard)
            fnr.append(result.fnr)

        # Convert lists to NumPy arrays with higher precision
        fit_arr         = np.array(fit,         dtype=np.float128)
        f1_arr          = np.array(f1,          dtype=np.float128)
        precision_arr   = np.array(precision,   dtype=np.float128)
        recall_arr      = np.array(recall,      dtype=np.float128)
        tp_arr          = np.array(tp,          dtype=np.float128)
        fp_arr          = np.array(fp,          dtype=np.float128)
        tn_arr          = np.array(tn,          dtype=np.float128)
        fn_arr          = np.array(fn,          dtype=np.float128)
        accuracy_arr    = np.array(accuracy,    dtype=np.float128)
        jaccard_arr     = np.array(jaccard,     dtype=np.float128)
        fnr_arr         = np.array(fnr,         dtype=np.float128)

        print(np.count_nonzero(tp_arr))

        # Now compute means without overflow
        mean_fit         = fit_arr.mean()
        mean_f1          = f1_arr.mean()
        mean_precision   = precision_arr.mean()
        mean_recall      = recall_arr.mean()
        mean_tp          = tp_arr.mean()
        mean_fp          = fp_arr.mean()
        mean_tn          = tn_arr.mean()
        mean_fn          = fn_arr.mean()
        mean_accuracy    = accuracy_arr.mean()
        mean_jaccard     = jaccard_arr.mean()
        mean_fnr         = fnr_arr.mean()

        # Compute population standard deviations
        std_fit          = fit_arr.std(ddof=0)
        std_f1           = f1_arr.std(ddof=0)
        std_precision    = precision_arr.std(ddof=0)
        std_recall       = recall_arr.std(ddof=0)
        std_tp           = tp_arr.std(ddof=0)
        std_fp           = fp_arr.std(ddof=0)
        std_tn           = tn_arr.std(ddof=0)
        std_fn           = fn_arr.std(ddof=0)
        std_accuracy     = accuracy_arr.std(ddof=0)
        std_jaccard      = jaccard_arr.std(ddof=0)
        std_fnr          = fnr_arr.std(ddof=0)
            
        # get mean of errors
        mean_errors = np.mean(np.array(errors, dtype=np.float128), axis=0)
       
        # get std of errors
        std_errors = np.std(np.array(errors, dtype=np.float128), axis=0)

        # Save results to a dictionary
        result_dict = {
            "eps":      eps,
            "mean_fit": mean_fit,
            "std_fit":  std_fit,
            "mean_f1":  mean_f1,
            "std_f1":   std_f1,
            "mean_prec": mean_precision,
            "std_prec": std_precision,
            "mean_rec": mean_recall,
            "std_rec":  std_recall,
            "mean_tp": mean_tp,
            "std_tp": std_tp,
            "mean_fp": mean_fp,
            "std_fp": std_fp,
            "mean_fn": mean_fn,
            "std_fn": std_fn,
            "mean_tn": mean_tn,
            "std_tn": std_tn,
            "mean_acc": mean_accuracy,
            "std_acc": std_accuracy,
            "mean_jaccard": mean_jaccard,
            "std_jaccard": std_jaccard,
            "mean_fnr": mean_fnr,
            "std_fnr": std_fnr,
            "num_evals": num_evals,
        }


        # Check if 'result' folder is present, if not create it and save dict as CSV-file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        output_dir = os.path.join(project_root, 'results')
        output_file_data = os.path.join(output_dir, f'data_{dataname}.csv')
        output_file_errors = os.path.join(output_dir, f'errors_{dataname}.csv')

        # Write dictionary to CSV
        with open(output_file_data, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            # Write rows by unpacking each list
            writer.writerow(result_dict.values())

        for i in range(5):
            error_dict = {
                "eps": eps,
                "max_query_length": 20,
                "subset_id": i,
                "subset_max_length": (i +1) * (20/5),
                "mean_error": mean_errors[i],
                "std_error": std_errors[i],
                "num_evals": num_evals,
            }

            # Write dictionary to CSV
            with open(output_file_errors, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                # Write rows by unpacking each list
                writer.writerow(error_dict.values())

    print(f"Done with the evaluation process. Time taken: {(time.time() - start)/60} minutes")

    return

cdef void eval_no_dp(TripletMap triplet, vector[Trajectory] traject, int num_evals, str dataname):
    
    # Ensure the results directory exists
    os.makedirs("../results", exist_ok=True)

    print ("Begin evaluation...")
    start = time.time()

    # Initialize CSV file
    with open(f'../results/data_noDP_{dataname}.csv', mode='w', newline='', encoding='utf-8') as df:
        writer = csv.writer(df)
        writer.writerow(['eps','mean_fit','std_fit','mean_f1','std_f1', 
        'mean_prec', 'std_prec', 'mean_rec', 'std_rec', "mean_tp", "std_tp",
        'mean_fp', 'std_fp', 'mean_fn', 'std_fn', 'mean_tn', 'std_tn',
        'mean_acc', 'std_acc',
        'mean_jaccard', 'std_jaccard', 'mean_fnr', 'std_fnr',
        'num_evals'])

    with open(f'../results/errors_noDP_{dataname}.csv', mode='w', newline='', encoding='utf-8') as ef:
        writer = csv.writer(ef)
        writer.writerow([
            'eps','max_query_length','subset_id',
            'subset_max_length','mean_error','std_error','num_evals'
        ])
    
    fit = []
    f1 = []
    precision = []
    recall = []
    tp = []
    fp = []
    tn = []
    fn = []
    errors = []
    accuracy = []
    jaccard = []
    fnr = []

    # Evaluate the triplet map
    for i in range(num_evals):
        result = evaluate_no_noise(triplet=triplet, trajectories=traject)
        fit.append(result.fit)
        f1.append(result.f1)
        precision.append(result.precision)
        recall.append(result.recall)
        tp.append(result.tp)
        fp.append(result.fp)
        tn.append(result.tn)
        fn.append(result.fn)
        errors.append([result.errors[j] *3.0 for j in range(5)])
        accuracy.append(result.accuracy)
        jaccard.append(result.jaccard)
        fnr.append(result.fnr)

    # Convert lists to NumPy arrays with higher precision
    fit_arr         = np.array(fit,         dtype=np.float128)
    f1_arr          = np.array(f1,          dtype=np.float128)
    precision_arr   = np.array(precision,   dtype=np.float128)
    recall_arr      = np.array(recall,      dtype=np.float128)
    tp_arr          = np.array(tp,          dtype=np.float128)
    fp_arr          = np.array(fp,          dtype=np.float128)
    tn_arr          = np.array(tn,          dtype=np.float128)
    fn_arr          = np.array(fn,          dtype=np.float128)
    accuracy_arr    = np.array(accuracy,    dtype=np.float128)
    jaccard_arr     = np.array(jaccard,     dtype=np.float128)
    fnr_arr         = np.array(fnr,         dtype=np.float128)

    # Now compute means without overflow
    mean_fit         = fit_arr.mean()
    mean_f1          = f1_arr.mean()
    mean_precision   = precision_arr.mean()
    mean_recall      = recall_arr.mean()
    mean_tp          = tp_arr.mean()
    mean_fp          = fp_arr.mean()
    mean_tn          = tn_arr.mean()
    mean_fn          = fn_arr.mean()
    mean_accuracy    = accuracy_arr.mean()
    mean_jaccard     = jaccard_arr.mean()
    mean_fnr         = fnr_arr.mean()

    # Compute population standard deviations
    std_fit          = fit_arr.std(ddof=0)
    std_f1           = f1_arr.std(ddof=0)
    std_precision    = precision_arr.std(ddof=0)
    std_recall       = recall_arr.std(ddof=0)
    std_tp           = tp_arr.std(ddof=0)
    std_fp           = fp_arr.std(ddof=0)
    std_tn           = tn_arr.std(ddof=0)
    std_fn           = fn_arr.std(ddof=0)
    std_accuracy     = accuracy_arr.std(ddof=0)
    std_jaccard      = jaccard_arr.std(ddof=0)
    std_fnr          = fnr_arr.std(ddof=0)
        
    # get mean of errors
    mean_errors = np.mean(np.array(errors, dtype=np.float128), axis=0)
    
    # get std of errors
    std_errors = np.std(np.array(errors, dtype=np.float128), axis=0)

    # Save results to a dictionary
    result_dict = {
        "eps":      np.inf,
        "mean_fit": mean_fit,
        "std_fit":  std_fit,
        "mean_f1":  mean_f1,
        "std_f1":   std_f1,
        "mean_prec": mean_precision,
        "std_prec": std_precision,
        "mean_rec": mean_recall,
        "std_rec":  std_recall,
        "mean_tp": mean_tp,
        "std_tp": std_tp,
        "mean_fp": mean_fp,
        "std_fp": std_fp,
        "mean_fn": mean_fn,
        "std_fn": std_fn,
        "mean_tn": mean_tn,
        "std_tn": std_tn,
        "mean_acc": mean_accuracy,
        "std_acc": std_accuracy,
        "mean_jaccard": mean_jaccard,
        "std_jaccard": std_jaccard,
        "mean_fnr": mean_fnr,
        "std_fnr": std_fnr,
        "num_evals": num_evals,
    }

    # Check if 'result' folder is present, if not create it and save dict as CSV-file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    output_dir = os.path.join(project_root, 'results')
    output_file_data = os.path.join(output_dir, f'data_noDP_{dataname}.csv')
    output_file_errors = os.path.join(output_dir, f'errors_noDP_{dataname}.csv')
    
    # Write dictionary to CSV
    with open(output_file_data, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result_dict.values())

    for i in range(5):
        error_dict = {
            "eps": np.inf,
            "max_query_length": 20,
            "subset_id": i,
            "subset_max_length": (i +1) * (20/5),
            "mean_error": mean_errors[i],
            "std_error": std_errors[i],
            "num_evals": 100,
        }

        # Write dictionary to CSV
        with open(output_file_errors, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            # Write rows by unpacking each list
            writer.writerow(error_dict.values())
    print(f"Done with the evaluation process. Time taken: {(time.time() - start)/60} minutes")

    return