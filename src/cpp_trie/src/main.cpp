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

// Generate Triplets with a cap of 100 unique 3-grams per trajectory 
TripletMap create_triplet_map(const std::vector<Trajectory>& trajectories) {
    TripletMap result;
    std::mutex result_mutex;

    #pragma omp parallel for schedule(static, 1) num_threads(8)
    for (size_t idx = 0; idx < trajectories.size(); ++idx) {
        const auto& traj = trajectories[idx];
        if (traj.size() < 3) continue; // Skip trajectories with less than 3 stations

        // Track unique triplets in this trajectory
        std::unordered_set<Triplet, TripletHash, TripletEqual> seen;
        std::vector<Triplet> unique_triplets;
        unique_triplets.reserve(std::max<size_t>(20, traj.size()));

        // Extract all unique triplets
        for (size_t i = 0; i + 2 < traj.size(); ++i) {
            const auto& s1 = traj[i];
            const auto& s2 = traj[i + 1];
            const auto& s3 = traj[i + 2];

            if (s1.data.empty() || s2.data.empty() || s3.data.empty()) {
                std::cerr << "Empty station data in triplet\n";
                continue;
            }

            Triplet t{s1, s2, s3};
            if (seen.insert(t).second) {
                unique_triplets.push_back(t);
            }
        }

        // If more than k unique triplets, randomly select k
        if (unique_triplets.size() > 20) {
            thread_local std::mt19937 rng((std::random_device())());
            std::shuffle(unique_triplets.begin(), unique_triplets.end(), rng);
            unique_triplets.resize(20);
        }

        // Contribute selected triplets to the global map
        for (const auto& t : unique_triplets) {
            std::lock_guard<std::mutex> lk(result_mutex);
            result[t] += 1.0;
        }
    }
    return result;
}

// Function to generate a trie from the triplet map and trajectories
bool create_trie(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories) {

    // double epsilon = 0.1;       // Adjust epsilon as needed.
    double sensitivity = 20.0;   // Typically 1 for count queries.
    double gamma = 0.01;
    double e_0 = 0.01;
    
    epsilon = 0.5 * (epsilon -e_0); // Set epsilon for Laplace noise
    
    // Define epsilon based on shares r_1, r_2
    double e_cnt = epsilon * 0.95;
    double e_fnr = epsilon * 0.05;

    int T = static_cast<int>(std::max((1.0 / gamma) * std::log(2.0 / e_0), 1.0 / (std::exp(1.0) * gamma)));

    double goal_f1 = 0.3;
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
EvalResult evaluate(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories, bool ablation) {

    // double epsilon = 0.1;       // Adjust epsilon as needed.
    double sensitivity = 20.0;   // Typically 1 for count queries.
    double gamma = 0.01;
    double e_0 = 0.01;
    
    epsilon = 0.5 * (epsilon - e_0); // Set epsilon for Laplace noise
    
    // Define epsilon based on shares r_1, r_2
    double e_cnt = epsilon * 0.95;
    double e_fnr = epsilon * 0.05;

    // Calculate T based on the provided formula -> used to get the number of iterations
    int T = static_cast<int>(std::max((1.0 / gamma) * std::log(2.0 / e_0), 1.0 / (std::exp(1.0) * gamma)));

    double goal_f1 = 0.65;
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
        if (ablation) {
            triplet_noised = ablation_noise_triplets(triplet_noised, e_cnt);
        } else {
            triplet_noised = noise_triplets(triplet_noised, e_cnt);
        }
        
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
            double accuracy = (values_orig[0] + values_orig[3]) / (values_orig[0] + values_orig[1] + values_orig[2] + values_orig[3]);
            double jaccard = values_orig[0] / (values_orig[0] + values_orig[1] + values_orig[2]);
            double fnr = values_orig[2] / (values_orig[0] + values_orig[2]);

            auto errors = trie.evaluateCountQueries(trajectories);
            double fit = trie.calculateFitness(trajectories);

            // Return the eval result if trie is accepted
            return EvalResult{
                .fit = fit,
                .f1 = f1_orig,
                .precision = precision_orig,
                .recall = recall_orig,
                .tp = values_orig[0],
                .fp = values_orig[1],
                .fn = values_orig[2],
                .tn = values_orig[3],
                .errors = errors,
                .accuracy = accuracy,
                .jaccard = jaccard,
                .fnr = fnr,
            };
        } else if (dis(gen) <= gamma) {
            // Return empty pair with probability gamma
            return EvalResult{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {0.0, 0.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0 };
        }
    }
    
    // Return empty pair if no trie was created
    return EvalResult{ 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {0.0, 0.0, 0.0, 0.0, 0.0}, 0.0, 0.0, 0.0 };
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
    double accuracy = (values[0] + values[3]) / (values[0] + values[1] + values[2] + values[3]);
    double jaccard = values[0] / (values[0] + values[1] + values[2]);
    double fnr = values[2] / (values[0] + values[2]);
    
    double fit = trie.calculateFitness(trajectories);
    
    auto errors = trie.evaluateCountQueries(trajectories);

    return EvalResult{
                .fit = fit,
                .f1 = f1,
                .precision = precision,
                .recall = recall,
                .tp = values[0],
                .fp = values[1],
                .fn = values[2],
                .tn = values[3],
                .errors = errors,
                .accuracy = accuracy,
                .jaccard = jaccard,
                .fnr = fnr,
    };
}

std::vector<double> evalErrors(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories) {
    
    // double epsilon = 0.1;       // Adjust epsilon as needed.
    double sensitivity = 20.0;   // Typically 1 for count queries.
    double gamma = 0.01;
    double e_0 = 0.01;
    
    epsilon = 0.5 * (epsilon - e_0); // Set epsilon for Laplace noise
    
    // Define epsilon based on shares r_1, r_2
    double e_cnt = epsilon * 0.9;
    double e_fnr = epsilon * 0.1;

    // Calculate T based on the provided formula -> used to get the number of iterations
    int T = static_cast<int>(std::max((1.0 / gamma) * std::log(2.0 / e_0), 1.0 / (std::exp(1.0) * gamma)));

    double goal_f1 = 0.3;
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
            
            // Return the eval result if trie is accepted
            return trie.evaluateCountQueries(trajectories);
        } else if (dis(gen) <= gamma) {
            // Return empty pair with probability gamma
            return {0,0,0,0,0};
        }
    }
    
    // Return empty pair if no trie was created
    return {0,0,0,0,0};
}

std::vector<double> evalErrors_noDP(TripletMap triplet, const std::vector<Trajectory>& trajectories) {
    // Create a trie from the triplet map and trajectories
    Trie trie;
    for (const auto& entry : triplet) {
        trie.insert(entry.first, entry.second);
    }

    // Evaluate the trie using the trajectories
    return trie.evaluateCountQueries(trajectories);
}

// Function to noise all triplet counts
TripletMap noise_triplets(const TripletMap& triplets, double epsilon) {
    // Define epsilon (privacy parameter) and sensitivity for the count query.
    double sensitivity = 1.0 *20.0;       // Adjust sensitivity as needed.
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

// Ablation function to add noise to triplets -> used for ablation studies ONLY
// Computes noise for each triplet count independently, without considering the original counts.
// This is different from the noise_triplets function, which adds noise based on the original counts.
TripletMap ablation_noise_triplets(const TripletMap& triplets, double epsilon) {
    // Define epsilon (privacy parameter) and sensitivity for the count query.
    double sensitivity = 1.0 * 20.0;       // Adjust sensitivity as needed.
    double scale = sensitivity / epsilon;

    // add laplace noise to the counts
    std::random_device rd;
    Laplace laplace(scale, rd());

    // Iterate over all triplet counts and add Laplace noise.
    TripletMap noisy_triplet_counts;
    for (const auto& [triplet, count] : triplets) {
        double noise = laplace.return_a_random_variable(scale);
        noisy_triplet_counts[triplet] = 0.0 + noise; // Only add noise, not the original count
    }

    return noisy_triplet_counts;
}

// Function to filter out all triplets with counts below the std() of the Laplace distribution
TripletMap select_significant_triplets(
    const TripletMap& triplet_counts,
    double epsilon
) {
    // Compute std() of the Laplace distribution
    // double sigma = sqrt(2.0) / epsilon;

    // random double between 1 and std::sqrt(2.0) / epsilon
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, std::sqrt(2.0) / epsilon);
    // Generate a random value
    const double threshold = dis(gen);

    // double delta = 0.5 * std::exp(-epsilon * random_value);

    // // const double sigma = std::sqrt(2.0) / epsilon;
    // const double threshold = 1.0/epsilon * std::log(1.0/(2*delta)); // 0.5525855

    // Filter triplet counts based on the threshold sigma
    TripletMap result;
    for (const auto& [triplet, count] : triplet_counts) {
        if (count >= threshold) {
            result[triplet] = count;
        }
    }

    return result;
}