#include <iostream>
#include <unordered_map>
#include <string>
#include <map>
#include <vector>
#include <string>
#include <utility>

class TrieNode {
public:
    std::unordered_map<std::string, TrieNode*> children;
    bool isEndOfTrajectory;

    TrieNode() : isEndOfTrajectory(false) {}
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const std::vector<std::pair<double, double>>& trajectory) {
        TrieNode* current = root;
        for (const auto& point : trajectory) {
            std::string key = std::to_string(point.first) + "," + std::to_string(point.second);
            if (current->children.find(key) == current->children.end()) {
                current->children[key] = new TrieNode();
            }
            current = current->children[key];
        }
        current->isEndOfTrajectory = true;
    }

    bool search(const std::vector<std::pair<double, double>>& trajectory) {
        TrieNode* current = root;
        for (const auto& point : trajectory) {
            std::string key = std::to_string(point.first) + "," + std::to_string(point.second);
            if (current->children.find(key) == current->children.end()) {
                return false;
            }
            current = current->children[key];
        }
        return current->isEndOfTrajectory;
    }

    bool startsWith(const std::vector<std::pair<double, double>>& prefix) {
        TrieNode* current = root;
        for (const auto& point : prefix) {
            std::string key = std::to_string(point.first) + "," + std::to_string(point.second);
            if (current->children.find(key) == current->children.end()) {
                return false;
            }
            current = current->children[key];
        }
        return true;
    }
};

int main() {
    Trie trie;
    trie.insert({{37.7749, -122.4194}, {34.0522, -118.2437}});
    trie.insert({{40.7128, -74.0060}, {34.0522, -118.2437}});

    std::cout << std::boolalpha;
    std::cout << "Search for trajectory 1: " << trie.search({{37.7749, -122.4194}, {34.0522, -118.2437}}) << std::endl;
    std::cout << "Search for trajectory 2: " << trie.search({{40.7128, -74.0060}, {34.0522, -118.2437}}) << std::endl;
    std::cout << "Search for partial trajectory: " << trie.search({{37.7749, -122.4194}}) << std::endl;
    std::cout << "Starts with trajectory 1: " << trie.startsWith({{37.7749, -122.4194}}) << std::endl;
    std::cout << "Starts with trajectory 2: " << trie.startsWith({{40.7128, -74.0060}}) << std::endl;

    return 0;
}