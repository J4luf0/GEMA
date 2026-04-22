cmake_minimum_required(VERSION 3.31) #default 3.10

project(GEMA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# find_program(ACPP_COMPILER NAMES acpp acpp.bat PATHS ENV PATH REQUIRED)
# message(STATUS "ACPP: ${ACPP_COMPILER}")
# set(CMAKE_CXX_COMPILER ${ACPP_COMPILER})
# set(CMAKE_CXX_LINKER ${ACPP_COMPILER})

find_package(AdaptiveCpp CONFIG REQUIRED)

enable_testing()

set(GTEST_DIR ${PROJECT_SOURCE_DIR}/lib/googletest)
add_subdirectory(${GTEST_DIR})

file(GLOB_RECURSE SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
#list(FILTER SRC_FILES EXCLUDE REGEX ".*Parallel.*")
file(GLOB_RECURSE TEST_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")
#list(FILTER TEST_FILES EXCLUDE REGEX "Acpp_test.*")

add_executable(GEMA_tests ${SRC_FILES} ${TEST_FILES})

# Compiled release? - no, bad idea
# add_executable(GEMA ${SRC_FILES})
# target_compile_options(GEMA PRIVATE -O3 -march=native)

set(SYCL_FILES
    src/core/TensorParallel.cpp
    test/src/core/Acpp_test.cpp
)

add_sycl_to_target(TARGET GEMA_tests SOURCES ${SYCL_FILES})

target_include_directories(GEMA_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${GTEST_DIR}/include)
target_link_libraries(GEMA_tests PRIVATE gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(GEMA_tests EXTRA_ARGS --gtest_color=yes)