#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <vector>
#include <iostream>

//namespace gema{


template<typename I>
void printVector(const std::vector<I>& vector){
    for(auto& item : vector) std::cout << item << ", ";
    std::cout << std::endl;
}

inline void doNotOptimizeAway(const void* p) {
    //asm volatile("" ::: "memory");
    asm volatile("" : : "g"(p) : "memory");
}


//}

#endif