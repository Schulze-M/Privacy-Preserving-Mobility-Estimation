// trajectory.cpp
#include "main.h"
#include "laplace.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <mutex>
#include <omp.h>
#include <chrono>

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
void display_progress(size_t current, size_t total, std::chrono::steady_clock::time_point start) {
    float progress = (float)current / total;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

    std::cout << "\rProgress: " << progress * 100.0f << "%" << " | Time Elapsed: " << elapsed << "s";
    if (progress > 0) {
        auto estimated = (elapsed / progress) - elapsed;
        std::cout << " | EST: " << (int)estimated << "s";
    }
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
void print_intermediate_results(const PrefixMap& result) {
    std::cout << "Intermediate Results (prefix -> suffix counts):" << std::endl;
    for (const auto& outer : result) {
        std::cout << "Prefix: " << outer.first << std::endl;
        for (const auto& inner : outer.second) {
            std::cout << "  Suffix: " << inner.first << " -> Count: " << inner.second << std::endl;
        }
    }
}

StartMap process_start(const std::vector<Trajectory>& trajectories) {
    StartMap result;
    
    // Count the number of times each start coordinate appears in the trajectories
    for (const auto& trajectory : trajectories) {
        result[trajectory[0]] += 1;
    }

    return result;
}

// Implement the process_prefix function
PrefixMap process_prefix(const std::vector<Trajectory>& trajectories) {
    PrefixMap result;
    std::mutex result_mutex;
    
    size_t total = 0;
    size_t count = 0;

    // Calculate total comparisons for progress tracking
    for (const auto& trajectory : trajectories) {
        total += trajectory.size() - 1; // Each pair can only have one valid suffix in its trajectory
    }

    auto start_time = std::chrono::steady_clock::now();

    // for each trajectory create a prefix-suffix pair and save it in the result map
    #pragma omp parallel for schedule(static, 1) num_threads(8)
    for (const auto& trajectory : trajectories) {
        for (size_t i = 0; i < trajectory.size() -1; i++) {
            const auto& prefix = trajectory[i];

            // Check if the suffix directly follows the prefix
            if (is_suffix(trajectory, i, i + 1)) {
                // Increment the count for the prefix-suffix pair and save it in the result map
                const auto& suffix = trajectory[i + 1];

                #pragma omp critical
                {
                    result_mutex.lock();
                    result[prefix][suffix] += 1;
                    result_mutex.unlock();
                }
            }

            // #pragma omp atomic
            // count++;
            
            // if (count % 100 == 0) { // Update progress every 1000 iterations
            //     display_progress(count, total, start_time);
            // }
        }
    }

    // add laplace noise to the counts
    // std::random_device rd;
    // Laplace laplace(rd());

    // for (auto& outer : result) {
    //     for (auto& inner : outer.second) {
    //         inner.second += laplace.return_a_random_variable();
    //     }
    // }

    std::cout << std::endl; // Ensure the progress bar ends on a new line
    return result;
}