cmake_minimum_required(VERSION 3.31) #default 3.10

project(GEMA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)

enable_testing()

set(GTEST_DIR ${PROJECT_SOURCE_DIR}/lib/googletest)
add_subdirectory(${GTEST_DIR})

file(GLOB_RECURSE SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE TEST_FILES "${PROJECT_SOURCE_DIR}/test/*.cpp")

add_executable(GEMA_tests ${SRC_FILES} ${TEST_FILES})

# Compiled release?
# add_executable(GEMA ${SRC_FILES})
# target_compile_options(GEMA PRIVATE -O3 -march=native)

target_include_directories(GEMA_tests PRIVATE ${PROJECT_SOURCE_DIR}/src ${GTEST_DIR}/include)
target_link_libraries(GEMA_tests PRIVATE gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(GEMA_tests EXTRA_ARGS --gtest_color=yes)