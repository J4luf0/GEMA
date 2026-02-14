#include <bitset>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/Tensor.hpp"

using gema::Tensor;

TEST(tensor_stress_test, getItem_001){

    const std::vector<uint64_t> dimensionSizes{256, 256, 256};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t itemCount = (256 * 256 * 256);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    std::vector<uint64_t> currentCoords = {0, 0, 0};
    for(uint64_t i = 0; i < itemCount; i++){
        
        tensor.getItem(currentCoords);

        if(i == itemCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}

TEST(tensor_stress_test, getItem_001_control){

    const std::vector<uint64_t> dimensionSizes{256, 256, 256};
    const uint64_t numberOfDimensions = dimensionSizes.size();
    const uint64_t itemCount = (256 * 256 * 256);

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

    std::vector<uint64_t> currentCoords = {0, 0, 0};
    for(uint64_t i = 0; i < itemCount; i++){
        
        //tensor.getItem(currentCoords);

        if(i == itemCount - 1) break;

        Tensor<int>::incrementCoords(currentCoords, dimensionSizes);
    }
}



TEST(tensor_stress_test, getItem_002){

    const std::vector<uint64_t> dimensionSizes{16, 16, 16, 16, 16, 16};

    Tensor<int> tensor = Tensor<int>(dimensionSizes);

}