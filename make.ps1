cmake -G "MinGW Makefiles" -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON # may help with intellisense?
# mingw32-make -C build clean

# "" > error.log
mingw32-make -C build # 2> error.log

$result = $LASTEXITCODE

if ($result -eq 0) {
    ./build/GEMA_tests.exe --gtest_color=yes # --gtest_list_tests
}