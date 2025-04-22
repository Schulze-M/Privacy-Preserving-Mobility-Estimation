# Privacy-Preserving-Mobility-Estimation
Code base for my master's thesis. Before executing the code a simulation of **n** users should be done. 

# Run the simulation
````
cd gtfs_simulation
python3 simulation.py $n_users
````

# Compiling the Cython code
```
cd src
python3 setup.py build_ext --inplace
```

# Running the code
```
python3 main.py $path_to_pkl_file
```

**$path_to_pkl_file** defines the path to the pickle that stores the trajectory data.
After running the script, a trie is generated and stored. This trie is a DP representation of the underlying data. Using this trie a simulation of **n** useres can be done.
