
# ------------- Verze kdy g++ + mingw64 kompiluje vše

#cmake -G "MinGW Makefiles" -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
  # -DCMAKE_EXPORT_COMPILE_COMMANDS=ON may help with intellisense?

# mingw32-make -C build clean

# "" > error.log
#mingw32-make -C build # 2> error.log


# ------------- Verze kdy acpp + windows toolchain kompiluje vše

#Set-Variable ACPP_DEFAULT_TARGETS="cuda:sm_86;omp"
# $env:ACPP_DEFAULT_TARGETS="cuda:sm_86"
# $cmake = "C:\Program Files\CMake\bin\cmake.exe"

# & $cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ `
#   -DAdaptiveCpp_DIR="C:/ManualInstall/AdaptiveCpp/AdaptiveCpp-LLVM20-Win/lib/cmake/AdaptiveCpp" `
#   -DCMAKE_C_COMPILER="C:/ManualInstall/AdaptiveCpp/AdaptiveCpp-LLVM20-Win/bin/clang.exe" `
#   -DCMAKE_CXX_COMPILER="C:/ManualInstall/AdaptiveCpp/AdaptiveCpp-LLVM20-Win/bin/clang++.exe" `
#   -DCMAKE_PREFIX_PATH="C:/ManualInstall/AdaptiveCpp/AdaptiveCpp-LLVM20-Win" `
#   -DCMAKE_BUILD_TYPE=Release `
#   -DWITH_CUDA_BACKEND=ON `
#   -DCMAKE_IGNORE_PATH="C:/msys64;C:/mingw64"

# & $cmake --build build

# # Konec stejný pro všechny případy - samotné spuštění

# $result = $LASTEXITCODE

# if ($result -eq 0) {
#     ./build/GEMA_tests.exe --gtest_color=yes # --gtest_list_tests
# }

export ACPP_DEFAULT_TARGETS="cuda:sm_89"

cmake -B build -G Ninja \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build

# spuštění testů pouze pokud build prošel
if [ $? -eq 0 ]; then
    ./build/GEMA_tests --gtest_color=yes
fi
