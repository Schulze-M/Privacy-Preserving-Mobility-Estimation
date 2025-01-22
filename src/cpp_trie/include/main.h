// main.h
#ifndef MAIN_H
#define MAIN_H

#include <vector>
#include <unordered_map>
#include <vector>
#include <array>
#include <ostream>

// Wrapper for fixed-size float array
struct Coordinate {
    double data[2];

    // Equality operator for use in unordered_map
    bool operator==(const Coordinate& other) const {
        return data[0] == other.data[0] && data[1] == other.data[1];
    }
};

struct CountCoordinate {
    double data[3];
};

// Custom hash function for Coordinate
namespace std {
    template <>
    struct hash<Coordinate> {
        std::size_t operator()(const Coordinate& coord) const {
            std::size_t h1 = std::hash<float>{}(coord.data[0]);
            std::size_t h2 = std::hash<float>{}(coord.data[1]);
            return h1 ^ (h2 << 1);
        }
    };
}

// Überladung des <<-Operators für Coordinate
std::ostream& operator<<(std::ostream& os, const Coordinate& coord);

// Define Trajectory as a vector of vectors of floats
using Trajectory = std::vector<Coordinate>;
using PrefixMap = std::unordered_map<Coordinate, std::vector<CountCoordinate>>;
using StartMap = std::unordered_map<Coordinate, double>;

// Function to process start coordinates
StartMap process_start(const std::vector<Trajectory>& trajectories);

// Function to process prefixes
PrefixMap process_prefix(const std::vector<Trajectory>& trajectories);

PrefixMap process_test(const Trajectory trajec, const StartMap start);

#endif // MAIN_H
