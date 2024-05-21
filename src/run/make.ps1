.\clean.ps1
New-Item -Path ../../target -ItemType Directory
g++ -std=c++20 -O2 -Wall -pedantic ../core/Main.cpp ../core/Tensor.cpp -o ../../target/tenz_build.exe