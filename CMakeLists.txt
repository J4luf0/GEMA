cmake_minimum_required(VERSION 3.10)

project(GEMA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/.*")
list(FILTER SOURCES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/CMakeFiles/.*")

add_executable(GEMA ${SOURCES})

target_include_directories(GEMA PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(GEMA PRIVATE ${PROJECT_SOURCE_DIR}/src/core)