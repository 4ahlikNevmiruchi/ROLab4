# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# This also sets the default CXX language to C++
project(MSSLab4 LANGUAGES CXX)

# Set a C++ standard for consistency
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This is the most important part!
# Find the MPI package and make it required.
# This will find mpicc, mpiexec, etc.
find_package(MPI REQUIRED)

add_executable(MSSLab4 main.cpp)

# Link your executable against the MPI libraries
# MPI::MPI_CXX is an "imported target" created by find_package.
# It automatically adds the include directories, linker flags,
# and libraries needed for MPI.
target_link_libraries(MSSLab4 PRIVATE MPI::MPI_CXX)

# Optional: Tell CLion (and CTest) how to run your program
# This tells the "Run" button to use mpiexec with 4 processes
set(MPIEXEC_EXECUTABLE ${MPI_CXX_COMPILER_EXECUTABLE} CACHE PATH "mpiexec executable")
set(MPIEXEC_NUMPROC_FLAG "-n" CACHE STRING "Flag for number of processes")
set(MPIEXEC_PREFLAGS "" CACHE STRING "Flags for mpiexec")

add_test(
        NAME main_test
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:MSSLab4>
)
