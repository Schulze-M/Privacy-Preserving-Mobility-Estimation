// File: gaussian.h
#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <random>
#include <cmath>

class Gaussian {
private:
    double epsilon;
    double sensitivity;
    double delta;
    std::mt19937 generator;
    std::normal_distribution<double> distribution;

    double compute_stddev(double sensitivity, double epsilon, double delta) {
        // Standard deviation based on (ε, δ)-differential privacy Gaussian mechanism
        // stddev ≥ sqrt(2 ln(1.25/δ)) * sensitivity / ε
        return std::sqrt(2 * std::log(1.25 / delta)) * sensitivity / epsilon;
    }

public:
    Gaussian(double _sensitivity, double _epsilon, double _delta, int seed = 42)
        : epsilon(_epsilon), sensitivity(_sensitivity), delta(_delta), generator(seed)
    {
        double stddev = compute_stddev(sensitivity, epsilon, delta);
        distribution = std::normal_distribution<double>(0.0, stddev);
    }

    double sample() {
        return distribution(generator);
    }

    // Optional overload to provide runtime stddev (ignores internal settings)
    double sample(double stddev) {
        std::normal_distribution<double> temp_dist(0.0, stddev);
        return temp_dist(generator);
    }
};

#endif /* GAUSSIAN_H */
