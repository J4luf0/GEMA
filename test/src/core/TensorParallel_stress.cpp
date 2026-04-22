#include <vector>

#include <gtest/gtest.h>

#include "core/Tensor.hpp"
#include "TestUtils.hpp"

using gema::Tensor;

constexpr uint64_t globalMultiplier = 1;



constexpr uint64_t constructionMultiplier = 16 * 16 * globalMultiplier;

TEST(tensorparallel_stress_test, construction_001){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}

TEST(tensorparallel_stress_test, construction_001_control){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    const uint64_t loopCount = (constructionMultiplier);
    for(uint64_t i = 0; i < loopCount; i++){
        //Tensor<int> tensor = Tensor<int>(dimensionSizes);
        doNotOptimizeAway(&i);
    }
}