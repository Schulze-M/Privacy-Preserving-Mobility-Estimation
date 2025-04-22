// trajectory.cpp
#include "main.h"
#include "laplace.h"
#include "gaussian.h"
#include "trie.h"
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
std::ostream& operator<<(std::ostream& os, const Station& coord) {
    os << "(" << coord.data << ")";
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

    // Calculate total comparisons for progress tracking
    for (const auto& trajectory : trajectories) {
        total += trajectory.size() - 1; // Each pair can only have one valid suffix in its trajectory
    }

    // Process each trajectory
    #pragma omp parallel for schedule(static, 1) num_threads(8)
    for (size_t traj_idx = 0; traj_idx < trajectories.size(); ++traj_idx) {
        const auto& trajectory = trajectories[traj_idx];
        PrefixMap local_result;

        for (size_t i = 0; i < trajectory.size() - 1; ++i) {
            const auto& prefix = trajectory[i];

            // Check if the suffix directly follows the prefix
            if (is_suffix(trajectory, i, i + 1)) {
                const auto& suffix = trajectory[i + 1];

                // Update local_result for the prefix-suffix pair
                auto& suffix_list = local_result[prefix];
                bool found = false;

                for (auto& count_coord : suffix_list) {
                    if (count_coord.suffix == suffix.data) {
                        count_coord.count += 1;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    suffix_list.push_back(CountStation{suffix.data, 1});
                }
            }
        }

        // Optimize local_result memory
        for (auto& [prefix, suffix_list] : local_result) {
            suffix_list.shrink_to_fit(); // Release unused capacity
        }

        // Merge local_result into the global result with a lock
        std::lock_guard<std::mutex> lock(result_mutex);
        for (auto& [prefix, suffix_list] : local_result) {
            auto& global_suffix_list = result[prefix];

            for (auto& local_count_coord : suffix_list) {
                bool found = false;

                for (auto& global_count_coord : global_suffix_list) {
                    if (global_count_coord.suffix == local_count_coord.suffix) {
                        global_count_coord.count += local_count_coord.count;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    global_suffix_list.push_back(local_count_coord);
                }
            }

            // Optimize global_suffix_list memory
            global_suffix_list.shrink_to_fit(); // Release unused capacity
        }
    }

    // Optimize final result memory
    for (auto& [prefix, suffix_list] : result) {
        suffix_list.shrink_to_fit(); // Release unused capacity
    }

    // ***** ADD LAPALCE NOISE HERE *****
    // Define epsilon (privacy parameter) and sensitivity for the count query.
    double epsilon = 1.0;       // Adjust epsilon as needed.
    double sensitivity = 1.0;   // Typically 1 for count queries.
    double scale = sensitivity / epsilon;

    // Initialize the Laplace noise generator (using a fixed seed or dynamic as required)
    Laplace laplace_noise(scale, 42);

    // Iterate over all prefix-suffix counts and add Laplace noise.
    for (auto& [prefix, suffix_list] : result) {
        for (auto& count_coord : suffix_list) {
            double noise = laplace_noise.return_a_random_variable(scale);
            // clip counts at zero. -> negative counts are unlikely, either a station is taken or not.
            double noisy_count = std::max(0.0, count_coord.count + noise);
            count_coord.count = noisy_count;
        }
    }
    // ***** END ADD LAPALCE NOISE *****

    std::cout << std::endl; // Ensure the progress bar ends on a new line
    return result;
}

// Function to process triplets
TripletMap process_triplets(const std::vector<Trajectory>& trajectories) {
    TripletMap result;
    std::mutex result_mutex;

    // Process each triplet
    #pragma omp parallel for schedule(static, 1) num_threads(8)
    for (size_t traj_idx = 0; traj_idx < trajectories.size(); ++traj_idx) {
        const auto& trajectory = trajectories[traj_idx];
        TripletMap local_counts;

        // Skip trajectories that have less than three stations.
        if (trajectory.size() < 3) {
            continue;
        }

        // Generate triplets only if trajectory has at least 3 stations.
        for (size_t i = 0; i < trajectory.size() - 2; ++i) {
            const auto& s1 = trajectory[i];
            const auto& s2 = trajectory[i + 1];
            const auto& s3 = trajectory[i + 2];

            if (s1.data.empty() || s2.data.empty() || s3.data.empty()) {
                std::cerr << "Empty station data in triplet: "
                          << "[" << s1.data << " | " << s2.data << " | " << s3.data << "]\n";
                continue;
            }

            Triplet t{s1, s2, s3};
            local_counts[t] += 1.0;
        }

        // Merge local counts into the global triplet_counts map.
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            for (const auto& entry : local_counts) {
                result[entry.first] += entry.second;
            }
        }
    }

    double epsilon = 1.0;       // Adjust epsilon as needed.
    double sensitivity = 1.0;   // Typically 1 for count queries.
    double delta = 1e-8;   // Adjust delta as needed.

    double goal_f1 = 0.85;
    double f1 = 0.0;
    TripletMap k_selected;

    while (f1 <= goal_f1)
    {
        Gaussian gaussian_noise(sensitivity, epsilon, delta, 1337);
        // Iterate over all triplet counts and add Gaussian noise.
        for (auto& entry : result) {
            double noise = gaussian_noise.sample();
            entry.second += noise;
            if (entry.second < 0.0) {
                entry.second = 0.0;
            }
        }

        k_selected = select_top_k_triplets(result); // Select top K triplets

        // Create a trie
        Trie trie;
        for (const auto& entry : k_selected) {
            trie.insert(entry.first, entry.second);
        }
        // trie.print(); // Print the trie

        f1 = trie.calculateF1Score(trajectories);
        std::cout << "F1-Score of the trie: " << f1 << std::endl;
    }
    

    return k_selected;
}

// Function to select top K triplets based on their counts
TripletMap select_top_k_triplets(const TripletMap& triplet_counts) {
    // Get random k, between size of map and size^2
    size_t k = rand() % triplet_counts.size() +1; // Random number between 1 and map size

    std::cout << "Selected k: " << k << std::endl;
    
    // Create a vector of pairs from the map
    std::vector<std::pair<Triplet, double>> triplet_vector(triplet_counts.begin(), triplet_counts.end());

    // Sort the vector based on counts in descending order
    std::sort(triplet_vector.begin(), triplet_vector.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Select the top K triplets
    TripletMap top_k_triplets;
    for (size_t i = 0; i < k && i < triplet_vector.size(); ++i) {
        top_k_triplets[triplet_vector[i].first] = triplet_vector[i].second;
    }

    return top_k_triplets;
}