cmake_minimum_required(VERSION 3.20)
project(
    knn-classifier 
    VERSION 1.0.0 
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "build tests dir" OFF)

if(NOT CMAKE_CXX_COMPILER)
    find_program(CLANG_EXECUTABLE NAMES clang++)
    if(CLANG_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${CLANG_EXECUTABLE})
    endif()
endif()


add_subdirectory(src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(bin)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()