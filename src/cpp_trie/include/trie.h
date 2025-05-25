#ifndef TRIE_H
#define TRIE_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <sstream>
#include <string>
#include <set>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>
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
    // Friend function to create a trie from a triplet map and trajectories.
    // This function is not a member of the Trie class but needs access to its private members.
    // friend Trie create_trie(TripletMap triplet, double epsilon, const std::vector<Trajectory>& trajectories);
private:
    TrieNode* root;
public:
    Trie() : root(new TrieNode()) {}
    ~Trie() { delete root; }

    // delete all nodes and allocate a fresh root
    void reset() {
        delete root;
        root = new TrieNode();
    }

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
                // Update the count for this node.
                // If the node is a leaf (i.e., it has no children), we add the count.
                // Otherwise, we just update the count.
                // This allows for the same station to be part of different sequences.
                // Note: This is a simplification; in a real trie, you might want to handle
                // this differently to avoid over-counting.
                // node->count += countValue;
                 // Only update the count for sequences of length at least 3.
                if (i - start + 1 >= 3)
                    node->count += countValue;
            
            }
        }
    }

    // For backwards compatibility: insert a Triplet (3-gram) by converting it to a vector and calling insertTrajectory.
    void insert(const Triplet& triplet, double countValue) {
        std::vector<Station> sequence = { triplet.s1, triplet.s2, triplet.s3 };
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
    // Neue Methode: calculateF1
    //
    // Berechnet den F₁-Score für 3-Gramme wie folgt:
    //   - trueSet      = alle 3-Gramme, die in den Trajektorien vorkommen
    //   - predictedSet = alle 3-Gramme, die im Trie mit count > 0 stehen
    //   - Precision    = |trueSet ∩ predictedSet| / |predictedSet|
    //   - Recall       = |trueSet ∩ predictedSet| / |trueSet|
    //   - F1           = 2 * (Precision * Recall) / (Precision + Recall)
    //
    // Wird 1 zurückgegeben, wenn sowohl trueSet als auch predictedSet leer sind.
    // ============================================================
    std::array<double,3> calculateF1(const std::vector<std::vector<Station>>& trajectories) const {
        using TripletSet = std::unordered_set<Triplet, TripletHash, TripletEqual>;

        // 1) trueSet aus den Trajektorien aufbauen
        TripletSet trueSet;
        for (const auto& traj : trajectories) {
            if (traj.size() < 3) continue;
            for (size_t i = 0; i + 2 < traj.size(); ++i) {
                trueSet.insert( Triplet{ traj[i], traj[i+1], traj[i+2] } );
            }
        }

        // 2) predictedSet aus dem Trie extrahieren (Tiefe genau 3)
        TripletSet predictedSet;
        for (const auto& kv1 : root->children) {
            Station s1 = kv1.first;
            TrieNode* n1 = kv1.second;
            for (const auto& kv2 : n1->children) {
                Station s2 = kv2.first;
                TrieNode* n2 = kv2.second;
                for (const auto& kv3 : n2->children) {
                    Station s3 = kv3.first;
                    TrieNode* n3 = kv3.second;
                    if (n3->count > 0.0) {
                        predictedSet.insert( Triplet{ s1, s2, s3 } );
                    }
                }
            }
        }

        const size_t TP = [&](){
            size_t cnt = 0;
            for (auto const& t : predictedSet)
                if (trueSet.count(t)) ++cnt;
            return cnt;
        }();

        const size_t P = predictedSet.size();
        const size_t T = trueSet.size();

        // Sonderfälle
        if (P == 0 && T == 0) return {1.0, 1.0, 1.0};   // keine 3-Gramme total → perfekte Übereinstimmung
        if (P == 0 || T == 0) return {0.0, 0.0, 0.0};   // keine Vorhersagen oder keine echten 3-Gramme

        double precision = static_cast<double>(TP) / P;
        double recall    = static_cast<double>(TP) / T;
        double f1 = (precision + recall > 0.0)
             ? 2.0 * (precision * recall) / (precision + recall)
             : 0.0;

        return {f1, precision, recall};
    }

    // returns { TP, FP, FN, TN }
    std::array<double,4> calculateConfusionMatrix(const std::vector<std::vector<Station>>& trajectories) const {
        using TripletSet = std::unordered_set<Triplet, TripletHash, TripletEqual>;

        // 1) Build trueSet from the trajectories
        TripletSet trueSet;
        for (const auto& traj : trajectories) {
            if (traj.size() < 3) continue;
            for (size_t i = 0; i + 2 < traj.size(); ++i) {
                trueSet.insert( Triplet{ traj[i], traj[i+1], traj[i+2] } );
            }
        }

         // 2) Build predictedSet from the trie (depth exactly 3, count > 0)
        TripletSet predictedSet;
        for (auto const& [s1, n1] : root->children) {
            for (auto const& [s2, n2] : n1->children) {
                for (auto const& [s3, n3] : n2->children) {
                    if (n3->count > 0.0)
                        predictedSet.insert( Triplet{ s1, s2, s3 } );
                }
            }
        }

        // 3) Universe = union of true and predicted
        TripletSet allSet = trueSet;
        allSet.insert(predictedSet.begin(), predictedSet.end());

        // 4) Compute confusion‐matrix counts
        double TP = 0.0, FP = 0.0, FN = 0.0, TN = 0.0;
        for (auto const& t : allSet) {
            bool isTrue = !!trueSet.count(t);
            bool isPred = !!predictedSet.count(t);
            if      (isPred && isTrue)  ++TP;
            else if (isPred && !isTrue) ++FP;
            else if (!isPred && isTrue) ++FN;
            else                         ++TN;  // here TN will always stay zero, 
                                                // since universe excludes anything neither
                                                // predicted nor true
        }

        return { TP, FP, FN, TN };
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

    // Estimate prefix count
    double estimatedCount(const std::vector<Station>& prefix) const {
        TrieNode* node = root;
        for (auto const& s: prefix) {
            auto it = node->children.find(s);
            if (it==node->children.end()) return 0.0;
            node = it->second;
        }
        return node->count;
    }

    // True prefix count
    static size_t trueCount(
        const std::vector<std::vector<Station>>& data,
        const std::vector<Station>& prefix
    ) {
        size_t cnt = 0;
        const size_t pL = prefix.size();
        if (pL == 0) return 0;

        // Slide a window of length pL across each traj
        for (auto const& traj : data) {
            if (traj.size() < pL) continue;

            // For every possible alignment i .. i+pL-1
            for (size_t i = 0; i + pL <= traj.size(); ++i) {
                bool match = true;
                for (size_t j = 0; j < pL; ++j) {
                    if (!(traj[i + j] == prefix[j])) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    ++cnt;
                }
            }
        }
        return cnt;
    }

    // Evaluate average relative errors for random count queries
    // Splits into 4 subsets by query length up to maxQL
    std::vector<double> evaluateCountQueries(
        const std::vector<std::vector<Station>>& data,
        size_t numQueries = 10000,
        int maxQL = 20
    ) const {
        static thread_local std::mt19937_64 rng(std::random_device{}());
        // Build universe
        std::unordered_set<Station, StationHash, StationEqual> us;
        for(auto const& t: data) for(auto const& s: t) us.insert(s);
        std::vector<Station> uni(us.begin(), us.end());
        size_t N=data.size(); 
        double s_bd=0.01*N;
        
        std::uniform_int_distribution<size_t> uid(0, uni.size()-1);
        const int k=5;
        std::vector<double> sumE(k,0);
        std::vector<size_t> cntQ(k,0);
        for(size_t qi=0;qi<numQueries;++qi) {
            int idx = qi*k/numQueries;
            int mlen = std::max(1,(idx+1)*maxQL/k);
            std::uniform_int_distribution<int> ld(1,mlen);
            int ql=ld(rng);
            std::vector<Station> q; q.reserve(ql);
            for(int i=0;i<ql;++i) q.push_back(uni[uid(rng)]);
            double tC=trueCount(data,q);
            double eC=estimatedCount(q);
            double denom = std::max(tC, s_bd);
            sumE[idx] += std::abs(eC-tC)/denom;
            cntQ[idx]++;
        }
        std::vector<double> avg(k);
        for(int i=0;i<k;++i) avg[i]=sumE[i]/cntQ[i];
        return avg;
    }

};

#endif // TRIE_H
