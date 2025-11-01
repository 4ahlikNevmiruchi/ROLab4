#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>   // Used in dummy init
#include <random>
#include <chrono>    // For high-resolution clock
#include <tuple>
#include <mpi.h>

// mpicxx -std=c++17 -o ParallelSort main.cpp
// mpiexec -n 4 ./ParallelSort


// Global MPI variables
int procNum = 0;
int procRank = -1;

/**
 * @brief Calculates the block size and displacement for each process.
 *
 * This function distributes 'dataSize' items as evenly as possible
 * among 'procNum' processes.
 *
 * @param dataSize The total number of data items.
 * @param procNum The total number of processes.
 * @return A tuple containing two vectors:
 * 1. counts: A vector where counts[i] is the block size for rank i.
 * 2. displs: A vector where displs[i] is the data displacement for rank i.
 */
std::tuple<std::vector<int>, std::vector<int>> calculateBlockParams(int dataSize, int procNum) {
    std::vector<int> counts(procNum);
    std::vector<int> displs(procNum);

    int restData = dataSize;
    int currentDispl = 0;

    for (int i = 0; i < procNum; ++i) {
        int currentSize = restData / (procNum - i);
        counts[i] = currentSize;
        displs[i] = currentDispl;

        restData -= currentSize;
        currentDispl += currentSize;
    }

    return std::make_tuple(counts, displs);
}

/**
 * @brief Initializes data for the sorting process.
 *
 * On rank 0:
 * 1. Prompts user for the total data size.
 * 2. Validates the input.
 * On all ranks:
 * 1. Broadcasts the dataSize.
 * 2. Calculates the local blockSize for the current rank.
 * 3. Resizes the local data vector (pProcData) to its blockSize.
 * On rank 0:
 * 1. Resizes the main data vector (pData) to dataSize.
 * 2. Fills pData with dummy data (sorted in reverse).
 */
void ProcessInitialization(std::vector<double>& pData, int& dataSize,
                           std::vector<double>& pProcData, int& blockSize) {

    if (procRank == 0) {
        std::cout << "Enter the size of data to be sorted: \n";
        while (!(std::cin >> dataSize) || dataSize < procNum) {
            std::cout << "Error: Data size must be an integer greater than or equal to "
                      << "the number of processes (" << procNum << ").\n";
            std::cout << "Enter the size of data to be sorted: \n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        std::cout << "Sorting " << dataSize << " data items...\n";
    }

    // Broadcast the total data size to all processes
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate block sizes and displacements for all processes
    auto [counts, displs] = calculateBlockParams(dataSize, procNum);

    // Get the local block size for this specific rank
    blockSize = counts[procRank];

    // Resize the local data vector to hold this rank's block
    pProcData.resize(blockSize);

    if (procRank == 0) {
        // Resize the main data vector (only on rank 0)
        pData.resize(dataSize);

        // Initialize the data
        // Create a random number generator
        std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> dist(0.0, 1000000.0); // Random doubles between 0 and 1,000,000

        // Fill the vector with random numbers
        for (size_t i = 0; i < dataSize; ++i) {
            pData[i] = dist(gen);
        }
    }
}

/**
 * @brief Distributes the main data vector from rank 0 to all processes.
 *
 * Uses MPI_Scatterv to send variable-sized blocks to each process.
 */
void DataDistribution(std::vector<double>& pData, int dataSize,
                      std::vector<double>& pProcData, int blockSize) {

    auto [counts, displs] = calculateBlockParams(dataSize, procNum);

    MPI_Scatterv(
        pData.data(),         // Send buffer (full data on rank 0)
        counts.data(),        // Array of block sizes
        displs.data(),        // Array of displacements
        MPI_DOUBLE,
        pProcData.data(),     // Receive buffer (local block)
        blockSize,            // Size of local block
        MPI_DOUBLE,
        0,                    // Root process
        MPI_COMM_WORLD
    );
}

/**
 * @brief Collects all sorted data blocks from all processes back to rank 0.
 *
 * Uses MPI_Gatherv to receive variable-sized blocks.
 */
void DataCollection(std::vector<double>& pData, int dataSize,
                    std::vector<double>& pProcData, int blockSize) {

    auto [counts, displs] = calculateBlockParams(dataSize, procNum);

    MPI_Gatherv(
        pProcData.data(),     // Send buffer (local sorted block)
        blockSize,            // Size of local block
        MPI_DOUBLE,
        pData.data(),         // Receive buffer (full data on rank 0)
        counts.data(),        // Array of block sizes
        displs.data(),        // Array of displacements
        MPI_DOUBLE,
        0,                    // Root process
        MPI_COMM_WORLD
    );
}

/**
 * @brief Performs the parallel odd-even transposition sort.
 *
 * 1. Performs a local sort on this process's data block.
 * 2. Iterates 'procNum' times, performing compare-exchange operations
 * with neighboring processes.
 */
void ParallelSort(std::vector<double>& pProcData) {
    int blockSize = pProcData.size();

    // Perform local sort using C++ standard library
    std::sort(pProcData.begin(), pProcData.end());

    enum splitMode { KeepFirstHalf, KeepSecondHalf };

    // Temporary buffers for merging
    std::vector<double> pDualData;
    std::vector<double> pMergedData;

    // Main parallel odd-even sort loop
    for (int i = 0; i < procNum; ++i) {
        int offset;
        splitMode mode;

        // Determine partner and merge-split mode for this iteration
        if ((i % 2) == 1) { // Odd iteration
            if ((procRank % 2) == 1) { // Odd rank
                offset = 1;
                mode = KeepFirstHalf;
            } else { // Even rank
                offset = -1;
                mode = KeepSecondHalf;
            }
        } else { // Even iteration
            if ((procRank % 2) == 1) { // Odd rank
                offset = -1;
                mode = KeepSecondHalf;
            } else { // Even rank
                offset = 1;
                mode = KeepFirstHalf;
            }
        }

        // Handle edge processes that have no partner in this iteration
        if ((procRank == 0 && offset == -1) ||
            (procRank == procNum - 1 && offset == 1)) {
            continue;
        }

        int dualRank = procRank + offset;
        int dualBlockSize;
        MPI_Status status;

        // Exchange block sizes with partner
        MPI_Sendrecv(
            &blockSize, 1, MPI_INT, dualRank, 0,
            &dualBlockSize, 1, MPI_INT, dualRank, 0,
            MPI_COMM_WORLD, &status
        );

        // Resize buffers to hold partner's data and merged data
        pDualData.resize(dualBlockSize);
        pMergedData.resize(blockSize + dualBlockSize);

        // Exchange data blocks with partner
        MPI_Sendrecv(
            pProcData.data(), blockSize, MPI_DOUBLE, dualRank, 0,
            pDualData.data(), dualBlockSize, MPI_DOUBLE, dualRank, 0,
            MPI_COMM_WORLD, &status
        );

        // Merge the two sorted blocks
        if (mode == KeepFirstHalf) {
            // This process is the "lower" rank partner.
            // Merge (mine, theirs) -> pMergedData
            std::merge(pProcData.begin(), pProcData.end(),
                       pDualData.begin(), pDualData.end(),
                       pMergedData.begin());

            // Keep the first 'blockSize' (smallest) elements
            std::copy(pMergedData.begin(), pMergedData.begin() + blockSize,
                      pProcData.begin());
        } else {
            // This process is the "higher" rank partner.
            // Merge (theirs, mine) -> pMergedData
            std::merge(pDualData.begin(), pDualData.end(),
                       pProcData.begin(), pProcData.end(),
                       pMergedData.begin());

            // Keep the last 'blockSize' (largest) elements
            std::copy(pMergedData.begin() + dualBlockSize, pMergedData.end(),
                      pProcData.begin());
        }
    }
}

/**
 * @brief (Optional) Helper function to print a vector's contents.
 */
void PrintData(const std::vector<double>& data, const std::string& title) {
    std::cout << title << " (Size " << data.size() << "):\n";
    if (data.size() > 20) {
        for (int i = 0; i < 10; ++i) std::cout << data[i] << " ";
        std::cout << "... ";
        for (size_t i = data.size() - 10; i < data.size(); ++i) std::cout << data[i] << " ";
    } else {
        for (const auto& val : data) {
            std::cout << val << " ";
        }
    }
    std::cout << "\n";
}

/**
 * @brief Tests the parallel sort result against a serial std::sort.
 *
 * On rank 0:
 * 1. Creates a copy of the unsorted data.
 * 2. Sorts the copy serially using std::sort.
 * 3. Compares the serial result with the parallel result (pData).
 */
void TestResult(const std::vector<double>& pData,
                const std::vector<double>& serialData) {
    if (procRank == 0) {
        // Compare the parallel-sorted vector with the serial-sorted vector
        if (std::equal(pData.begin(), pData.end(), serialData.begin())) {
            std::cout << "\n[SUCCESS] The results of serial and parallel algorithms are identical.\n";
        } else {
            std::cout << "\n[FAILURE] The results of serial and parallel algorithms are NOT identical.\n";
        }
    }
}

int main(int argc, char* argv[]) {
    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    int dataSize = 0;
    int blockSize = 0;
    std::vector<double> pData;       // Holds the full data array (only on rank 0)
    std::vector<double> pProcData;   // Holds the local data block (on all ranks)
    std::vector<double> pUnsortedData; // Copy for testing (only on rank 0)
    std::vector<double> serialData;    // Holds serial sort result (only on rank 0)

    double p_start, p_finish, p_duration; // Parallel timer
    double s_start, s_finish, s_duration; // Serial timer

    // Initialization
    ProcessInitialization(pData, dataSize, pProcData, blockSize);

    if (procRank == 0) {
        pUnsortedData = pData; // Make a copy for testing and serial sort
        serialData = pUnsortedData; // Copy for the serial sort

        std::cout << "Running serial std::sort on Rank 0...\n";
        s_start = MPI_Wtime();

        // Run Serial Sort (on Rank 0 only)
        std::sort(serialData.begin(), serialData.end());

        s_finish = MPI_Wtime();
        s_duration = s_finish - s_start;
        std::cout << "Serial std::sort time: " << s_duration << " seconds\n";
    }

    // Synchronize before starting the parallel timer
    MPI_Barrier(MPI_COMM_WORLD);

    if (procRank == 0) {
        std::cout << "Running parallel sort on " << procNum << " processes...\n";
    }
    p_start = MPI_Wtime();

    // Data Distribution
    DataDistribution(pData, dataSize, pProcData, blockSize);

    // Parallel Sorting
    ParallelSort(pProcData);

    // Data Collection
    DataCollection(pData, dataSize, pProcData, blockSize);

    // Synchronize before stopping the timer
    MPI_Barrier(MPI_COMM_WORLD);
    p_finish = MPI_Wtime();
    p_duration = p_finish - p_start;

    if (procRank == 0) {
        std::cout << "\n--- Results ---\n";
        std::cout << "Serial std::sort time: " << s_duration << " seconds\n";
        std::cout << "Parallel sort time:    " << p_duration << " seconds\n";
    }

    // Test Results
    TestResult(pData, serialData);

    // Finalize
    MPI_Finalize();
    return 0;
}