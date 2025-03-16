cmake -G "MinGW Makefiles" -B build 
# mingw32-make -C build clean
mingw32-make -C build

./build/GEMA_tests.exe --gtest_color=yes # --gtest_list_tests