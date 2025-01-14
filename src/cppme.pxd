from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp cimport bool

# Declarations
ctypedef vector[vector[float]] Trajectory
ctypedef unordered_map[Trajectory, int] SuffixMap
ctypedef unordered_map[Trajectory, SuffixMap] ResultMap

cdef extern from "<algorithm>" namespace "std":
    bool equal(...)

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        vector()
        void push_back(const T&)

cdef extern from "<unordered_map>" namespace "std":
    cdef cppclass unordered_map[K, V]:
        unordered_map()
        V& operator[](const K&)
