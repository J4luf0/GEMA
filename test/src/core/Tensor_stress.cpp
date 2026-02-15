#include <bitset>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/Tensor.hpp"

using gema::Tensor;

template<typename I>
void printVector(const std::vector<I>& vector){
    for(auto& item : vector) std::cout << item << ", ";
    std::cout << std::endl;
}

inline void doNotOptimizeAway(const void* p) {
    //asm volatile("" ::: "memory");
    asm volatile("" : : "g"(p) : "memory");
}

constexpr double globalMultiplier = 0.5;



constexpr double constructionMultiplier = 1. * globalMultiplier;

TEST(tensor_stress_test, construction_001){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (16 * 16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_001_control){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (16 * 16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_002){

    const std::vector<uint64_t> dimensionSizes{16, 16, 16, 16};

    const uint64_t loopCount = (16 * 16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_002_control){

    const std::vector<uint64_t> dimensionSizes{16, 16, 16, 16};

    const uint64_t loopCount = (16 * 16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_003){

    const std::vector<uint64_t> dimensionSizes{256 * 256};

    const uint64_t loopCount = (16 * 16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_003_control){

    const std::vector<uint64_t> dimensionSizes{256 * 256};

    const uint64_t loopCount = (16 * 16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_004){

    const std::vector<uint64_t> dimensionSizes{4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

    const uint64_t loopCount = (16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_004_control){

    const std::vector<uint64_t> dimensionSizes{4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

    const uint64_t loopCount = (16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_005){

    const std::vector<uint64_t> dimensionSizes{16, 16, 16, 16, 16};

    const uint64_t loopCount = (16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_005_control){

    const std::vector<uint64_t> dimensionSizes{16, 16, 16, 16, 16};

    const uint64_t loopCount = (16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_006){

    const std::vector<uint64_t> dimensionSizes{256 * 256 * 16};

    const uint64_t loopCount = (16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensor_stress_test, construction_006_control){

    const std::vector<uint64_t> dimensionSizes{256 * 256 * 16};

    const uint64_t loopCount = (16 * 16 * constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}



constexpr double getItemMultiplier = 1. * globalMultiplier;

TEST(tensor_stress_test, getItem_001){

    const std::vector<uint64_t> dimensionSizes{256, 256, 256};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = (256 * 256 * 256 * getItemMultiplier);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    std::vector<uint64_t> currentCoords = {0, 0, 0};
    uint64_t i = 0;
    for(; i < loopCount; i++){
        
        tensor.getItem(currentCoords);

        if(i == loopCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}

TEST(tensor_stress_test, getItem_001_control){

    const std::vector<uint64_t> dimensionSizes{256, 256, 256};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t loopCount = (256 * 256 * 256 * getItemMultiplier);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    std::vector<uint64_t> currentCoords = {0, 0, 0};
    for(uint64_t i = 0; i < loopCount; i++){
        
        //tensor.getItem(currentCoords);

        if(i == loopCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}



TEST(tensor_stress_test, getItem_002){

    const std::vector<uint64_t> dimensionSizes{16, 16, 16, 16, 16, 16};

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

}