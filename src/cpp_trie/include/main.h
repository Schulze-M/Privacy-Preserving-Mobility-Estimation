// main.h
#ifndef MAIN_H
#define MAIN_H

// #include "trie.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>
#include <ostream>

// Wrapper for fixed-size float array
struct Station {
    std::string data;

    bool operator==(const Station& other) const {
        return data == other.data; 
    }
};

struct CountStation {
    std::string suffix;
    double count;
};

// Define a struct for a triplet of stations.
struct Triplet {
    Station s1, s2, s3;
    bool operator<(Triplet const& o) const noexcept {
        if (s1.data != o.s1.data) return s1.data < o.s1.data;
        if (s2.data != o.s2.data) return s2.data < o.s2.data;
        return s3.data < o.s3.data;
    }
    bool operator==(Triplet const& o) const noexcept {
        return s1 == o.s1 && s2 == o.s2 && s3 == o.s3;
    }
};

struct EvalResult {
    double f1;
    double fit;
    std::vector<double> errors;
};

// Custom hash function for Coordinate
namespace std {
    template <>
    struct hash<Station> {
        size_t operator()(const Station& coord) const {
            return std::hash<std::string>()(coord.data);
        }
    };
}

// Hash and equality functors for Station (for unordered_set/map convenience)
struct StationHash {
    size_t operator()(const Station& s) const noexcept {
        return std::hash<Station>()(s);
    }
};
struct StationEqual {
    bool operator()(const Station& a, const Station& b) const noexcept {
        return a == b;
    }
};

// Hash function for Triplet.
struct TripletHash {
    std::size_t operator()(const Triplet& t) const {
        std::size_t h1 = std::hash<Station>()(t.s1);
        std::size_t h2 = std::hash<Station>()(t.s2);
        std::size_t h3 = std::hash<Station>()(t.s3);
        // Combine the hashes.
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct TripletEqual {
    bool operator()(Triplet const& a, Triplet const& b) const noexcept {
        return a == b;
    }
};

// Forward declaration of Trie class
// class Trie;

// Overload the <<-Operators for Station
std::ostream& operator<<(std::ostream& os, const Station& coord);

// Define Trajectory as a vector of vectors of floats
using Trajectory = std::vector<Station>;
using PrefixMap = std::unordered_map<Station, std::vector<CountStation>>;
using TripletMap = std::unordered_map<Triplet, double, TripletHash>;
using StartMap = std::unordered_map<Station, double>;

// Function to process start coordinates
StartMap process_start(const std::vector<Trajectory>& trajectories);

// Function to process prefixes
PrefixMap process_prefix(const std::vector<Trajectory>& trajectories);

// Generate Triplets
TripletMap create_triplet_map(const std::vector<Trajectory>& trajectories);

// Generate a trie
bool create_trie(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories);

// Function to evaluate the trie
EvalResult evaluate(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories);

PrefixMap process_test(const Trajectory trajec, const StartMap start);

// Function to add Laplace noise to triplet counts
TripletMap noise_triplets(const TripletMap& triplets, double epsilon);

// Function to get most significant triplets
TripletMap select_significant_triplets(const TripletMap& triplet_counts, double epsilon);

#endif // MAIN_H
