./clean.sh
mkdir -p ../../target
g++ -std=c++20 -O2 -Wall -pedantic ../../src/core/Main.cpp ../../src/core/Tensor.cpp -o ../../target/tenz_build.out