// main.h
#ifndef MAIN_H
#define MAIN_H

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
    Station first;
    Station second;
    Station third;

    bool operator==(const Triplet& other) const {
        return (first == other.first) &&
               (second == other.second) &&
               (third == other.third);
    }
};

// Custom hash function for Coordinate
namespace std {
    template <>
    struct hash<Station> {
        std::size_t operator()(const Station& coord) const {
            return std::hash<std::string>()(coord.data);
        }
    };
}

// Hash function for Triplet.
struct TripletHash {
    std::size_t operator()(const Triplet& t) const {
        std::size_t h1 = std::hash<Station>()(t.first);
        std::size_t h2 = std::hash<Station>()(t.second);
        std::size_t h3 = std::hash<Station>()(t.third);
        // Combine the hashes.
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

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

// Function to process triplets
TripletMap process_triplets(const std::vector<Trajectory>& trajectories);

PrefixMap process_test(const Trajectory trajec, const StartMap start);

// Function to get k top triplets
TripletMap select_top_k_triplets(const TripletMap& triplet_counts);

#endif // MAIN_H
