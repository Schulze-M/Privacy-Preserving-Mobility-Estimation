// trajectory.cpp
#include "main.h"
#include "laplace.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <ostream>

// Überladung des <<-Operators für Trajectory
// TODO: delete this function
std::ostream& operator<<(std::ostream& os, const Trajectory& trajectory) {
    os << "[";
    for (size_t i = 0; i < trajectory.size(); ++i) {
        os << trajectory[i];
        if (i != trajectory.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Überladung des <<-Operators für Coordinate
// TODO: delete this function
std::ostream& operator<<(std::ostream& os, const Coordinate& coord) {
    os << "(" << coord.data[0] << ", " << coord.data[1] << ")";
    return os;
}

// Simple progress bar function
void display_progress(size_t current, size_t total) {
    float progress = (float)current / total * 100.0f;
    std::cout << "\rProgress: " << progress << "% " << std::flush;
}

bool is_suffix(const Trajectory& trajectory, size_t prefix_index, size_t suffix_index) {
    // A suffix is valid only if it directly follows the prefix
    return suffix_index == prefix_index + 1;
}

// Function to print the input trajectories
// TODO: delete this function
void print_trajectories(const std::vector<Trajectory>& trajectories) {
    std::cout << "Input Trajectories:" << std::endl;
    for (size_t i = 0; i < trajectories.size(); ++i) {
        std::cout << "Trajectory " << i + 1 << ": ";
        for (const auto& point : trajectories[i]) {
            std::cout << point << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "End of Input Trajectories" << std::endl;
}

// Function to print the intermediate results stored in ResultMap
// TODO: delete this function
void print_intermediate_results(const ResultMap& result) {
    std::cout << "Intermediate Results (prefix -> suffix counts):" << std::endl;
    for (const auto& outer : result) {
        std::cout << "Prefix: " << outer.first << std::endl;
        for (const auto& inner : outer.second) {
            std::cout << "  Suffix: " << inner.first << " -> Count: " << inner.second << std::endl;
        }
    }
}

// Implement the process_prefix function
ResultMap process_prefix(const std::vector<Trajectory>& trajectories) {
    ResultMap result;
    
    size_t total = 0;
    size_t count = 0;

    // Calculate total comparisons for progress tracking
    for (const auto& trajectory : trajectories) {
        total += trajectory.size() - 1; // Each pair can only have one valid suffix in its trajectory
    }

    // for each trajectory create a prefix-suffix pair and save it in the result map
    for (const auto& trajectory : trajectories) {
        for (size_t i = 0; i < trajectory.size() -1; i++) {
            const auto& prefix = trajectory[i];

            // Check if the suffix directly follows the prefix
            if (is_suffix(trajectory, i, i + 1)) {
                // Increment the count for the prefix-suffix pair and save it in the result map
                const auto& suffix = trajectory[i + 1];
                result[prefix][suffix] += 1;
            }
            count++;
            if (count % 100 == 0) { // Update progress every 1000 iterations
                display_progress(count, total);
            }
        }
    }
    std::cout << std::endl; // Ensure the progress bar ends on a new line
    return result;
}