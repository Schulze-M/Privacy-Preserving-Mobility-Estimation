// trajectory.cpp
#include "gaussian.h"
#include "laplace.h"
#include "main.h"
#include "trie.h"

#include <algorithm>
#include <chrono>
#include <cmath>   // for log, exp, M_E
#include <fstream>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <optional>
#include <ostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

TripletMap create_triplet_map(const std::vector<Trajectory>& trajectories) {
    TripletMap result;
    std::mutex result_mutex;

    #pragma omp parallel for schedule(static, 1) num_threads(8)
    for (size_t idx = 0; idx < trajectories.size(); ++idx) {
        const auto& traj = trajectories[idx];
        if (traj.size() < 3) continue; // Skip trajectories with less than 3 stations

        // Create a local set to track seen triplets in this trajectory
        // ensures that we only count unique triplets in each trajectory
        std::unordered_set<Triplet, TripletHash, TripletEqual> seen;

        // Iterate over the trajectory to find triplets
        for (size_t i = 0; i + 2 < traj.size(); ++i) {
            const auto& s1 = traj[i];
            const auto& s2 = traj[i + 1];
            const auto& s3 = traj[i + 2];

            if (s1.data.empty() || s2.data.empty() || s3.data.empty()) {
                std::cerr << "Empty station data in triplet\n";
                continue;
            }

            // Create a triplet and check if it has been seen before
            Triplet t{s1, s2, s3};
            if (seen.insert(t).second) {
                std::lock_guard<std::mutex> lk(result_mutex);
                result[t] += 1.0;
            }
        }
    }
    return result;
}

// Function to generate a trie from the triplet map and trajectories
bool create_trie(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories) {

    // double epsilon = 0.1;       // Adjust epsilon as needed.
    double sensitivity = 1.0;   // Typically 1 for count queries.
    double gamma = 0.01;
    double e_0 = 0.01;
    
    epsilon = epsilon -e_0; // Set epsilon for Laplace noise
    double e_fnr = epsilon * 0.1;
    double e_cnt = epsilon * 0.9;

    int T = static_cast<int>(std::max((1.0 / gamma) * std::log(2.0 / e_0), 1.0 / (std::exp(1.0) * gamma)));

    double goal_f1 = 0.85;
    TripletMap original_triplet = triplet; // keep original clean
    TripletMap k_selected;

    // Initialize the random number generator
    std::random_device rd_coin;
    std::mt19937 gen(rd_coin()); // Mersenne Twister RNG
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i = 0; i < T; i++)
    {   
        // initialize Laplace noise generator
        std::random_device rd;
        Laplace lap_fnr(sensitivity/e_fnr, rd());

        // reset triplet to original for each iteration
        TripletMap triplet_noised = original_triplet;

        // noise all triplet counts, using Laplace noise
        triplet_noised = noise_triplets(triplet_noised, e_cnt);
    
        // select significant triplets
        k_selected = select_significant_triplets(triplet_noised, e_cnt); // Select top K triplets

        // Create a trie
        Trie trie;
        for (const auto& entry : k_selected) {
            trie.insert(entry.first, entry.second);
        }

        // Evaluate the trie using the trajectories -> calculate confusion matrix
        std::array<double, 4> values = trie.calculateConfusionMatrix(trajectories); // Get tp, fp, fn, tn

        // noise the values using Laplace noise
        for (size_t i = 0; i < values.size(); ++i) {
            double noise = lap_fnr.return_a_random_variable();
            // clip counts at zero. -> negative counts are unlikely, either a station is taken or not.
            values[i] = std::max(0.0, values[i] + noise);
        }
        
        // Calculate F1 score
        double recall = values[0] / (values[0] + values[2]); // Sensitivity
        double precision = values[0] / (values[0] + values[1]);
        double f1 = (precision + recall > 0.0)
            ? 2.0 * (precision * recall) / (precision + recall)
            : 0.0;

        // If the F1 score is above the goal, save the trie and return
        if(f1 >= goal_f1) {
            // Save trie to json file
            std::string json = trie.toJson();
            std::ofstream file("trie.json");
            if (file.is_open()) {
                file << json;
                file.close();
            } else {
                std::cerr << "Unable to open file for writing trie JSON." << std::endl;
            }

            auto errors = trie.evaluateCountQueries(trajectories, 10000, 20);

            // Return the trie if the noised goal F1 score is reached
            return true;
        } else if (dis(gen) <= gamma) {
            // Delete the previous Trie
            std::ofstream file("trie.json");
            if (file.is_open()) {
                file << "{}"; // Write an empty JSON object
                file.close();
            } else {
                std::cerr << "Unable to open file for writing empty trie JSON." << std::endl;
            }

            return false; // Return false with probability gamma
        }
    }

    // Clear JSON file if no trie was created
    std::ofstream file("trie.json");
    if (file.is_open()) {
        file << "{}"; // Write an empty JSON object
        file.close();
    } else {
        std::cerr << "Unable to open file for writing empty trie JSON." << std::endl;
    }
    
    // Return false if no trie was created
    return false;
}

// Function to create a from a triplet map and trajectories, without rejection sampling
bool create_trie_no_rejection(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories) {
    // noise all triplet counts, using Laplace noise
    triplet = noise_triplets(triplet, epsilon);

    // select significant triplets
    TripletMap k_selected = select_significant_triplets(triplet, epsilon); // Select top K triplets
    
    // Create a trie from the triplet map and trajectories
    Trie trie;
    for (const auto& entry : k_selected) {
        trie.insert(entry.first, entry.second);
    }
    

    // Save trie to json file
    std::string json = trie.toJson();
    std::ofstream file("trie_noReject.json");
    if (file.is_open()) {
        file << json;
        file.close();
    } else {
        std::cerr << "Unable to open file for writing trie JSON." << std::endl;
    }

    return true; // Return true if the trie was created successfully
}

// Function to create a trie without any noise -> base case
bool create_trie_no_noise(TripletMap triplet, const std::vector<Trajectory>& trajectories) {
    
    // Create a trie from the triplet map and trajectories
    Trie trie;
    for (const auto& entry : triplet) {
        trie.insert(entry.first, entry.second);
    }
    
    // Save trie to json file
    std::string json = trie.toJson();
    std::ofstream file("trie_noDP.json");
    if (file.is_open()) {
        file << json;
        file.close();
    } else {
        std::cerr << "Unable to open file for writing trie JSON." << std::endl;
    }

    return true; // Return true if the trie was created successfully
}

// Function to process triplets
EvalResult evaluate(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories) {

    // double epsilon = 0.1;       // Adjust epsilon as needed.
    double sensitivity = 1.0;   // Typically 1 for count queries.
    double gamma = 0.01;
    double e_0 = 0.01;
    
    epsilon = epsilon -e_0; // Set epsilon for Laplace noise
    double e_fnr = epsilon * 0.1;
    double e_cnt = epsilon * 0.9;

    // Calculate T based on the provided formula -> used to get the number of iterations
    int T = static_cast<int>(std::max((1.0 / gamma) * std::log(2.0 / e_0), 1.0 / (std::exp(1.0) * gamma)));

    double goal_f1 = 0.9;
    TripletMap original_triplet = triplet; // keep original clean
    TripletMap k_selected;

    // Initialize the random number generator
    std::random_device rd_coin;
    std::mt19937 gen(rd_coin()); // Mersenne Twister RNG
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i = 0; i < T; i++)
    {
        // initialize Laplace noise generator
        std::random_device rd;
        Laplace laplace(sensitivity/e_fnr, rd());

        // reset triplet to original for each iteration
        TripletMap triplet_noised = original_triplet;

        // noise all triplet counts, using Laplace noise
        triplet_noised = noise_triplets(triplet_noised, e_cnt);
        
        // select significant triplets
        k_selected = select_significant_triplets(triplet_noised, e_cnt); // Select top K triplets

        // Create a trie
        Trie trie;
        for (const auto& entry : k_selected) {
            trie.insert(entry.first, entry.second);
        }
        
        std::array<double, 4> values = trie.calculateConfusionMatrix(trajectories); // Get tp, fp, fn, tn
        std::array<double, 4> values_orig = values;

        // noise the values using Laplace noise
        for (size_t i = 0; i < values.size(); ++i) {
            double noise = laplace.return_a_random_variable();
            // clip counts at zero. -> negative counts are unlikely, either a station is taken or not.
            values[i] = std::max(0.0, values[i] + noise);
        }

        double recall = values[0] / (values[0] + values[2]); // Sensitivity
        double precision = values[0] / (values[0] + values[1]);
        double f1 = (precision + recall > 0.0)
            ? 2.0 * (precision * recall) / (precision + recall)
            : 0.0;
        
        if (f1 >= goal_f1) {
            double recall_orig = values_orig[0] / (values_orig[0] + values_orig[2]); // Sensitivity
            double precision_orig = values_orig[0] / (values_orig[0] + values_orig[1]);
            double f1_orig = (precision_orig + recall_orig > 0.0)
                ? 2.0 * (precision_orig * recall_orig) / (precision_orig + recall_orig)
                : 0.0;
            double specificity = values_orig[3] / (values_orig[1] + values_orig[3]);
            double npv = values_orig[3] / (values_orig[3] + values_orig[2]);
            double accuracy = (values_orig[0] + values_orig[3]) / (values_orig[0] + values_orig[1] + values_orig[2] + values_orig[3]);
            double jaccard = values_orig[0] / (values_orig[0] + values_orig[1] + values_orig[2]);
            double mcc = (values_orig[0] * values_orig[3] - values_orig[1] * values_orig[2]) / 
                        std::sqrt((values[0] + values_orig[1]) * (values_orig[0] + values_orig[2]) * (values_orig[3] + values_orig[1]) * (values_orig[3] + values_orig[2]));
            double fnr = values_orig[2] / (values_orig[0] + values_orig[2]);
            double p4 = 4.0 / (1.0/precision_orig + 1.0/recall_orig + 1.0/specificity + 1.0/npv);

            auto errors = trie.evaluateCountQueries(trajectories, 10000, 20);
            double fit = trie.calculateFitness(trajectories);

            // Return the eval result if trie is accepted
            return EvalResult{
                .fit = fit,
                .f1 = f1_orig,
                .precision = precision_orig,
                .recall = recall_orig,
                .errors = errors,
                .specificty = specificity,
                .npv = npv,
                .accuracy = accuracy,
                .jaccard = jaccard,
                .mcc = mcc,
                .fnr = fnr,
                .p4 = p4,
            };
        } else if (dis(gen) <= gamma) {
            // Return empty pair with probability gamma
            return EvalResult{ 0.0, 0.0, 0.0, 0.0, {0,0,0,0}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        }
    }
    
    // Return empty pair if no trie was created
    return EvalResult{ 0.0, 0.0, 0.0, 0.0, {0,0,0,0}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
}

// Function to eval trie without rejection sampling
EvalResult evaluate_no_rejection(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories) {
    
    // initialize Laplace noise generator
    std::random_device rd;
    Laplace laplace(1.0/epsilon, rd());

    // noise all triplet counts, using Laplace noise
    triplet = noise_triplets(triplet, epsilon);

    // select significant triplets
    TripletMap k_selected = select_significant_triplets(triplet, epsilon); // Select top K triplets
    
    // Create a trie from the triplet map and trajectories
    Trie trie;
    for (const auto& entry : k_selected) {
        trie.insert(entry.first, entry.second);
    }

    std::array<double, 4> values = trie.calculateConfusionMatrix(trajectories);
    double recall = values[0] / (values[0] + values[2]); // Sensitivity
    double precision = values[0] / (values[0] + values[1]);
    double f1 = (precision + recall > 0.0)
        ? 2.0 * (precision * recall) / (precision + recall)
        : 0.0;
    double specificity = values[3] / (values[1] + values[3]);
    double npv = values[3] / (values[3] + values[2]);
    double accuracy = (values[0] + values[3]) / (values[0] + values[1] + values[2] + values[3]);
    double jaccard = values[0] / (values[0] + values[1] + values[2]);
    double mcc = (values[0] * values[3] - values[1] * values[2]) / 
                std::sqrt((values[0] + values[1]) * (values[0] + values[2]) * (values[3] + values[1]) * (values[3] + values[2]));
    double fnr = values[2] / (values[0] + values[2]);
    double p4 = 4.0 / (1.0/precision + 1.0/recall + 1.0/specificity + 1.0/npv);
    
    double fit = trie.calculateFitness(trajectories);
    
    auto errors = trie.evaluateCountQueries(trajectories, 10000, 20);

    return EvalResult{
                .fit = fit,
                .f1 = f1,
                .precision = precision,
                .recall = recall,
                .errors = errors,
                .specificty = specificity,
                .npv = npv,
                .accuracy = accuracy,
                .jaccard = jaccard,
                .mcc = mcc,
                .fnr = fnr,
                .p4 = p4,
    };
}

// Function to eval base case without noise
EvalResult evaluate_no_noise(TripletMap triplet, const std::vector<Trajectory>& trajectories) {
    
    // Create a trie from the triplet map and trajectories
    Trie trie;
    for (const auto& entry : triplet) {
        trie.insert(entry.first, entry.second);
    }

    std::array<double, 4> values = trie.calculateConfusionMatrix(trajectories);
    double recall = values[0] / (values[0] + values[2]); // Sensitivity
    double precision = values[0] / (values[0] + values[1]);
    double f1 = (precision + recall > 0.0)
        ? 2.0 * (precision * recall) / (precision + recall)
        : 0.0;
    double specificity = values[3] / (values[1] + values[3]);
    double npv = values[3] / (values[3] + values[2]);
    double accuracy = (values[0] + values[3]) / (values[0] + values[1] + values[2] + values[3]);
    double jaccard = values[0] / (values[0] + values[1] + values[2]);
    double mcc = (values[0] * values[3] - values[1] * values[2]) / 
                std::sqrt((values[0] + values[1]) * (values[0] + values[2]) * (values[3] + values[1]) * (values[3] + values[2]));
    double fnr = values[2] / (values[0] + values[2]);
    double p4 = 4.0 / (1.0/precision + 1.0/recall + 1.0/specificity + 1.0/npv);
    
    double fit = trie.calculateFitness(trajectories);
    
    auto errors = trie.evaluateCountQueries(trajectories, 10000, 20);

    return EvalResult{
                .fit = fit,
                .f1 = f1,
                .precision = precision,
                .recall = recall,
                .errors = errors,
                .specificty = specificity,
                .npv = npv,
                .accuracy = accuracy,
                .jaccard = jaccard,
                .mcc = mcc,
                .fnr = fnr,
                .p4 = p4,
    };
}

// Function to noise all triplet counts
TripletMap noise_triplets(const TripletMap& triplets, double epsilon) {
    // Define epsilon (privacy parameter) and sensitivity for the count query.
    double sensitivity = 1.0;       // Adjust sensitivity as needed.
    double scale = sensitivity / epsilon;

    // add laplace noise to the counts
    std::random_device rd;
    Laplace laplace(scale, rd());

    // Iterate over all triplet counts and add Laplace noise.
    TripletMap noisy_triplet_counts;
    for (const auto& [triplet, count] : triplets) {
        double noise = laplace.return_a_random_variable(scale);
        noisy_triplet_counts[triplet] = count + noise; // Ensure non-negative counts
    }

    return noisy_triplet_counts;
}

// Function to filter out all triplets with counts below the std() of the Laplace distribution
TripletMap select_significant_triplets(
    const TripletMap& triplet_counts,
    double epsilon
) {
    // Compute std() of the Laplace distribution
    // sigma = sqrt(2)*2 / epsilon

    // random double between 1 and std::sqrt(2.0) / epsilon
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, std::sqrt(2.0) / epsilon);
    // Generate a random value
    double random_value = dis(gen);

    double delta = 0.5 * std::exp(-epsilon * random_value);

    // const double sigma = std::sqrt(2.0) / epsilon;
    const double threshold = 1.0/epsilon * std::log(1.0/(2*delta)); // 0.5525855

    // Filter triplet counts based on the threshold sigma
    TripletMap result;
    for (const auto& [triplet, count] : triplet_counts) {
        if (count >= threshold) {
            result[triplet] = count;
        }
    }

    return result;
}