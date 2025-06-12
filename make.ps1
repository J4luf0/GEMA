cmake -G "MinGW Makefiles" -B build
# mingw32-make -C build clean

#"" > error.log
mingw32-make -C build # 2> error.log

./build/GEMA_tests.exe --gtest_color=yes # --gtest_list_tests