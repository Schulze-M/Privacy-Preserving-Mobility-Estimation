#ifndef TRIE_H
#define TRIE_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include "main.h"  // Assumes Station and Triplet definitions are declared here

// A node in the trie.
class TrieNode {
public:
    // 'count' stores the accumulated frequency for sequences ending at this node.
    double count;
    
    // Children nodes keyed by Station. Custom hash and comparison for Station are defined in main.h.
    std::unordered_map<Station, TrieNode*> children;

    TrieNode() : count(0.0) {}

    ~TrieNode() {
        for (auto& child : children)
            delete child.second;
    }
};

// The Trie class supports insertion of entire trajectories. Every contiguous subsequence of length ≥ 3 is recorded.
// In addition, a fitness measure is provided that computes how many 3-grams from the given trajectories
// are represented (i.e. are present and have a positive count) in the trie.
class Trie {
private:
    TrieNode* root;

    // Recursive helper to export a node into a JSON-like string.
    // The format is: { "child_station": {"count": <value>, "children": { ... }}, ... }
    std::string toJsonNode(TrieNode* node) const {
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (const auto& kv : node->children) {
            if (!first)
                oss << ",";
            first = false;
            // Use the station's data string as the key.
            oss << "\"" << kv.first.data << "\":";
            oss << "{";
            oss << "\"count\":" << kv.second->count << ",";
            oss << "\"children\":" << toJsonNode(kv.second);
            oss << "}";
        }
        oss << "}";
        return oss.str();
    }

public:
    Trie() : root(new TrieNode()) {}
    
    ~Trie() {
        delete root;
    }

    // Insert a full trajectory (or any sequence of Stations). This method traverses each contiguous subsequence
    // of length ≥ 3 and updates the count in the corresponding node.
    void insertTrajectory(const std::vector<Station>& trajectory, double countValue) {
        size_t n = trajectory.size();
        if (n < 3)
            return;
        for (size_t start = 0; start <= n - 3; ++start) {
            TrieNode* node = root;
            for (size_t i = start; i < n; ++i) {
                const Station& s = trajectory[i];
                if (node->children.find(s) == node->children.end()) {
                    node->children[s] = new TrieNode();
                }
                node = node->children[s];
                // Only update the count for sequences of length at least 3.
                if (i - start + 1 >= 3)
                    node->count += countValue;
            }
        }
    }

    // For backwards compatibility: insert a Triplet (3-gram) by converting it to a vector and calling insertTrajectory.
    void insert(const Triplet& triplet, double countValue) {
        std::vector<Station> sequence = { triplet.first, triplet.second, triplet.third };
        insertTrajectory(sequence, countValue);
    }

    // Print the trie as a tree-like structure (for debugging).
    void print() const {
        std::cout << "Trie Structure:" << std::endl;
        printHelper(root, 0);
    }

private:
    // Helper for print(): prints indentation based on depth.
    void printHelper(TrieNode* node, int level) const {
        if (!node) return;
        for (const auto& child : node->children) {
            std::cout << std::string(level * 4, ' ') << "└── " << child.first.data;
            if (child.second->count > 0)
                std::cout << " (count: " << std::fixed << std::setprecision(2) << child.second->count << ")";
            std::cout << std::endl;
            printHelper(child.second, level + 1);
        }
    }

public:
    // Export the entire Trie as a JSON-formatted string.
    std::string toJson() const {
        return toJsonNode(root);
    }

    // ============================================================
    // New Method: calculateFitness
    //
    // This method calculates the "fitness" of the trie with respect to the given
    // trajectories. It does so by iterating over each trajectory and checking
    // whether each contiguous 3-gram from the trajectory is represented in the trie 
    // (i.e. exists in the trie with a positive count).
    //
    // The fitness is defined as:
    //      fitness = (number of triplets matched) / (total number of triplets)
    //
    // A result of 1.0 indicates perfect fitness (all triplets in the trajectories are present
    // in the trie), whereas values less than 1.0 indicate some loss.
    // ============================================================
    double calculateFitness(const std::vector<std::vector<Station>>& trajectories) const {
        size_t totalTriplets = 0;
        size_t matchedTriplets = 0;
        // Loop over each trajectory.
        for (const auto& trajectory : trajectories) {
            if (trajectory.size() < 3)
                continue;
            // For each contiguous 3-gram in the trajectory.
            for (size_t i = 0; i <= trajectory.size() - 3; ++i) {
                totalTriplets++;
                TrieNode* node = root;
                bool found = true;
                // Traverse the trie for the three stations.
                for (size_t j = i; j < i + 3; ++j) {
                    auto it = node->children.find(trajectory[j]);
                    if (it == node->children.end()) {
                        found = false;
                        break;
                    }
                    node = it->second;
                }
                // Count as a match if all 3 stations were found and the final count is positive.
                if (found && node->count > 0)
                    matchedTriplets++;
            }
        }
        return totalTriplets > 0 ? static_cast<double>(matchedTriplets) / totalTriplets : 1.0;
    }

    // Calculate the precision of the trie with respect to the allowed behavior.
    // Here we assume that the set S of all stations observed in the trajectories represents
    // the maximum allowed transitions from any node.
    //
    // For each internal node (node with outgoing transitions) in the trie, we compute:
    //    escaping_edges = |S| - (number of observed transitions)
    // The precision is then:
    //    precision = 1 - (sum_{node}(escaping_edges)) / (number_of_nodes * |S|)
    //
    // A precision of 1.0 indicates that at every node, all allowed transitions are observed (i.e. the model is very restrictive),
    // whereas lower values indicate that the model allows many transitions that were not observed in the log.
    double calculatePrecision(const std::vector<std::vector<Station>>& trajectories) const {
        // First, compute S: the set of all stations observed in the trajectories.
        std::unordered_set<Station> stationSet;
        for (const auto& trajectory : trajectories) {
            for (const auto& s : trajectory)
                stationSet.insert(s);
        }
        size_t S = stationSet.size();
        if (S == 0)
            return 1.0;

        size_t totalNodes = 0;
        size_t totalObservedTransitions = 0;

        // Recursively traverse the trie, considering only nodes with outgoing transitions.
        std::function<void(TrieNode*)> traverse = [&](TrieNode* node) {
            if (!node)
                return;
            if (!node->children.empty()) {
                totalNodes++;
                totalObservedTransitions += node->children.size();
            }
            for (const auto& kv : node->children) {
                traverse(kv.second);
            }
        };

        traverse(root);

        if (totalNodes == 0)
            return 1.0;

        size_t totalPossibleTransitions = totalNodes * S;
        size_t escapingEdges = totalPossibleTransitions - totalObservedTransitions;

        std::cout << totalNodes << " nodes, " << totalObservedTransitions << " observed transitions, "
                  << totalPossibleTransitions << " possible transitions, "
                  << escapingEdges << " escaping edges." << std::endl;

        double precision = static_cast<double>(escapingEdges) / totalPossibleTransitions;
        return precision;
    }

    double calculateF1Score(const std::vector<std::vector<Station>>& trajectories) const {
        double fitness = calculateFitness(trajectories);
        double precision = calculatePrecision(trajectories);
        std::cout << "Fitness: " << fitness << ", Precision: " << precision << std::endl;
        if (fitness + precision == 0)
            return 0.0;
        return 2 * ((fitness * precision) / (fitness + precision));
    }
};

#endif // TRIE_H
