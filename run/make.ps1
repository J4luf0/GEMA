.\clean.ps1
New-Item -Path ../target -ItemType Directory
g++ -std=c++20 -O2 -Wall -pedantic ../src/core/Main.cpp ../src/core/Tensor.cpp -o ../target/tenz_build.exe