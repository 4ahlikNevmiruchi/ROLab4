/*
 * sequential_sort.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <limits>

/**
 * @brief Sorts a vector of integers in place using the Bubble Sort algorithm.
 * @param arr The vector to be sorted.
 */
void bubbleSort(std::vector<int>& arr) {
    size_t n = arr.size();
    if (n == 0) return;

    bool swapped;
    for (size_t i = 0; i < n - 1; ++i) {
        swapped = false;
        for (size_t j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        // If no two elements were swapped by inner loop, then break
        if (!swapped)
            break;
    }
}

/**
 * @brief Generates a vector of 'n' random integers.
 * @param n The number of integers to generate.
 * @return A vector containing the random integers.
 */
std::vector<int> generateRandomData(int n) {
    // Set up a random number generator
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    );

    std::vector<int> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = distrib(gen);
    }
    return data;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
        std::cerr << "Example: ./sequential_sort 10000" << std::endl;
        return 1;
    }

    int n;
    try {
        n = std::stoi(argv[1]); // Convert argument to integer
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid number '" << argv[1] << "'. Please enter an integer." << std::endl;
        return 1;
    }

    if (n <= 0) {
        std::cerr << "Error: N must be a positive number." << std::endl;
        return 1;
    }

    // Generate Data
    std::cout << "Generating " << n << " random elements..." << std::endl;
    std::vector<int> original_data = generateRandomData(n);
    std::cout << "Data generation complete." << std::endl;

    // Create copies for each sort
    std::vector<int> data_for_bubble = original_data;
    std::vector<int> data_for_std_sort = original_data;

    std::cout << std::fixed << std::setprecision(6); // Set precision for timing output

    // Time Bubble Sort
    std::cout << "\nStarting Bubble Sort..." << std::endl;
    auto start_bubble = std::chrono::high_resolution_clock::now();

    bubbleSort(data_for_bubble);

    auto end_bubble = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_bubble = end_bubble - start_bubble;
    std::cout << "Bubble Sort finished." << std::endl;


    // Time std::sort
    std::cout << "\nStarting std::sort (Library Sort)..." << std::endl;
    auto start_std = std::chrono::high_resolution_clock::now();

    std::sort(data_for_std_sort.begin(), data_for_std_sort.end());

    auto end_std = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_std = end_std - start_std;
    std::cout << "std::sort finished." << std::endl;


    // Compare Timings
    std::cout << "\n--- Sequential Experiment Results ---" << std::endl;
    std::cout << "Dataset size:     " << n << " elements" << std::endl;
    std::cout << "Bubble Sort time: " << duration_bubble.count() << " seconds" << std::endl;
    std::cout << "std::sort time:   " << duration_std.count() << " seconds" << std::endl;

    if(duration_std.count() > 0) {
        std::cout << "\nstd::sort was " << (duration_bubble.count() / duration_std.count())
                  << " times faster than Bubble Sort." << std::endl;
    }

    // Final Verification
    std::cout << "\nVerifying results..." << std::endl;

    // Check if the sorted vectors are identical
    if (data_for_bubble == data_for_std_sort) {
        std::cout << "Verification successful! Both sorts produced the same result." << std::endl;
    } else {
        std::cout << "VERIFICATION FAILED! The sorted arrays do not match." << std::endl;

        // Optional: Print a few elements to show the discrepancy
        std::cout << "Bubble sort [0-5]: ";
        for(int i=0; i<std::min(n, 5); ++i) std::cout << data_for_bubble[i] << " ";
        std::cout << "\nstd::sort [0-5]:   ";
        for(int i=0; i<std::min(n, 5); ++i) std::cout << data_for_std_sort[i] << " ";
        std::cout << std::endl;
    }

    return 0;
}