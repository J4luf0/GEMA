#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <memory>
#include <format>
#include <thread>

#include "ITensor.hpp"

#define uint64t uint64_t

using std::vector, std::make_unique, std::cout, std::endl;

inline void test_constructor_001(){

    cout << "test_constructor_001\n";
    const vector<int> dimensionSizes{2, 3};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {0, 1});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {0, 2});
    tensor->setItem(7,    {1, 2});


    vector<int> expected = {2, 3};

    assert(tensor->getNumberOfDimensions() == 2);

    for(uint64t i = 0; i < tensor->getNumberOfDimensions(); i++){
        assert(tensor->getDimensionSizes()[i] == expected[i]);
    }
}

inline void test_constructor_002(){

    cout << "test_constructor_002\n";
    const vector<int> dimensionSizes{2, 2, 3};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(7,     {0, 0, 0});
    tensor->setItem(0.2,  {1, 0, 0});
    tensor->setItem(-0,    {0, 1, 0});
    tensor->setItem(-1,    {1, 1, 0});
    tensor->setItem(-100,   {0, 0, 1});
    tensor->setItem(0.1,  {1, 0, 1});
    tensor->setItem(99,    {0, 1, 1});
    tensor->setItem(35,    {1, 1, 1});
    tensor->setItem(-2.56,   {0, 0, 2});
    tensor->setItem(30.62,  {1, 0, 2});
    tensor->setItem(2,  {1, 1, 2});

    vector<int> expected = {2, 2, 3};

    assert(tensor->getNumberOfDimensions() == 3);

    for(uint64t i = 0; i < tensor->getNumberOfDimensions(); i++){
        assert(tensor->getDimensionSizes()[i] == expected[i]);
    }
}

inline void test_constructor_003(){

    cout << "test_constructor_003\n";
    const vector<int> dimensionSizes{2};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(2,     {0});
    tensor->setItem(-0.3,  {1});
    tensor->setItem(4,     {2});

    vector<int> expected = {2};

    assert(tensor->getNumberOfDimensions() == 1);

    for(uint64t i = 0; i < tensor->getNumberOfDimensions(); i++){
        assert(tensor->getDimensionSizes()[i] == expected[i]);
    }
}

inline void test_constructor_004(){

    cout << "test_constructor_004\n";
    const vector<int> dimensionSizes{0};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    assert(tensor->getNumberOfDimensions() == 1);
}


inline void test_setItem_001(){

    cout << "test_setItem_001\n";
    const vector<int> dimensionSizes{2, 3};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {0, 1});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {0, 2});
    tensor->setItem(7,    {1, 2});

    assert(tensor->getItem({0, 0}) == 5);
    assert(tensor->getItem({1, 0}) == 0.55);
    assert(tensor->getItem({0, 1}) == 0);
    assert(tensor->getItem({1, 1}) == -2);
    assert(tensor->getItem({0, 2}) == 4.5);
    assert(tensor->getItem({1, 2}) == 7);
}

inline void test_setItem_002(){

    cout << "test_setItem_002\n";
    const vector<int> dimensionSizes{3, 2};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {2, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});
    tensor->setItem(7,    {2, 1});

    tensor2->setItems({5, 0.55, -0, -2, 4.5, 7});

    /*tensor->forEach([](double& item){
        cout << item << "; ";
    });
    cout << endl;*/
    assert(*tensor == *tensor2);
}

//Our main <3 -----------------------------------------------------------------------------------------------------------------
int main(){

    cout << "\nStart testing.\n";
    cout << "Testing Tensor.\n\n";


    test_constructor_001();
    test_constructor_002();
    test_constructor_003();
    test_constructor_004();
    test_setItem_001();
    test_setItem_002();

    cout << "\nAll test done.\n";
    
/*
    const vector<int> dimensionSizes = {2, 3};
    auto tensor1 = make_unique<Tensor<double>>(dimensionSizes);

    tensor1->setItem(5,     {0, 0});
    tensor1->setItem(0.55,    {1, 0});
    tensor1->setItem(-0,     {0, 1});
    tensor1->setItem(-2,      {1, 1});
    tensor1->setItem(4.5,   {0, 2});
    tensor1->setItem(7,    {1, 2});

    tensor1->showTensor();
    
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);

    const vector<double> tensor2Items = {2, -5, 10, 5.89647, 3, 4};
    tensor2->setItems(tensor2Items);

    tensor2->showTensor();


    cout << "Addition now.\n";

    auto tensorAdd = *tensor1 + *tensor2;
    tensorAdd->showTensor();


    cout << '\n';

    cout << "Transposition now.\n";
    auto tensorTransposed = tensor1->transposition();
    cout << "Tensor allocated.\n";
    tensorTransposed->showTensor();

    cout << "Is equilateral? / no\n";
    cout << tensorTransposed->isTensorEquilateral() << '\n';

    vector<int> tensorEquiDimensionSizes = {3, 3};
    auto tensorEqui = make_unique<Tensor<double>>(tensorEquiDimensionSizes);
    cout << "Is equilateral? / yes\n";
    cout << tensorEqui->isTensorEquilateral() << '\n';

    assert(tensorEqui->isTensorEquilateral() == 1);
    assert(tensorTransposed->isTensorEquilateral() == 0);*/



    return 0;
}

