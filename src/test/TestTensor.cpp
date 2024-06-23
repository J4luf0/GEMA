#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <memory>
#include <format>
#include <thread>

#include "../core/ITensor.hpp"

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

inline void test_constructor_005(){

    cout << "test_constructor_005\n";
    const vector<int> dimensionSizes{2};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    auto result = make_unique<Tensor<double>>(*tensor);

    assert(*tensor == *result);
}

inline void test_operatorEquals_001(){

    cout << "test_operatorEquals_001\n";
    const vector<int> dimensionSizes{1, 2, 2};
    
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000});
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({5.1, -0, -0.000001, 500000});

    bool expected = true;
    assert(expected == (*tensor == *tensor));
}

inline void test_operatorEquals_002(){

    cout << "test_operatorEquals_002\n";
    const vector<int> dimensionSizes{3, 2, 1};
    
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000, 1, -1});
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({5.1, 0, -0, 500000, 1, -1});

    bool expected = false;
    assert(expected == (*tensor == *tensor2));
}

inline void test_operatorEquals_003(){

    cout << "test_operatorEquals_003\n";
    const vector<int> dimensionSizes{1};
    
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);

    bool expected = true;
    assert(expected == (*tensor == *tensor2));
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

inline void test_setItem_003(){

    cout << "test_setItem_003\n";
    const vector<int> dimensionSizes{2, 2};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({2, 0, -1, 6.4});

    tensor->setItem(3.3, {1, 0});

    auto expected = make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({2, 3.3, -1, 6.4});
    
    assert(*tensor == *expected);
}

inline void test_isTensorEquilateral_001(){

    cout << "test_isTensorEquilateral_001\n";

    const vector<int> dimensionSizes{2, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100});

    assert(tensor->isTensorEquilateral() == true);
}

inline void test_isTensorEquilateral_002(){

    cout << "test_isTensorEquilateral_002\n";

    const vector<int> dimensionSizes{2, 2, 1};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, 3, -2});

    assert(tensor->isTensorEquilateral() == false);
}

inline void test_isTensorEquilateral_003(){

    cout << "test_isTensorEquilateral_003\n";

    const vector<int> dimensionSizes{2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({200});

    assert(tensor->isTensorEquilateral() == true);
}

inline void test_isTensorEquilateral_004(){

    cout << "test_isTensorEquilateral_004\n";

    const vector<int> dimensionSizes{2, 2, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100, 24, -24, 5, 45});

    assert(tensor->isTensorEquilateral() == true);
}

inline void test_transposition_001(){

    cout << "test_transposition_001\n";

    const vector<int> dimensionSizes{2, 3};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100, -2, -16});

    Tensor<int>* result = tensor->transposition();
    
    const vector<int> expectedDimensionSizes{3, 2};
    auto expected = make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({0, -1, -2, 5, 100, -16});

    assert(*result == *expected);
    delete result;
}

inline void test_transposition_002(){

    cout << "test_transposition_002\n";

    const vector<int> dimensionSizes{1, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5});

    Tensor<int>* result = tensor->transposition();
    
    const vector<int> expectedDimensionSizes{2, 1};
    auto expected = make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({0, 5});

    assert(*result == *expected);
    delete result;
}

inline void test_operatorAssign_001(){

    cout << "test_operatorAssign_001\n";

    const vector<int> dimensionSizes{3, 2};

    auto tensor = new Tensor<double>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {2, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});
    tensor->setItem(7,    {2, 1});

    Tensor<double> tensor2;
    tensor2 = *tensor;

    tensor->setItem(8, {0, 0});

    auto expected = new Tensor<double>(dimensionSizes);
    expected->setItems({5, 0.55, -0, -2, 4.5, 7});
    //cout << "ex " << expected->getItem({0, 0}) << " " << tensor2.getItem({0, 0}) << endl;
    assert(*expected == tensor2);

    delete expected;
    delete tensor;
}

//Our main <3 -----------------------------------------------------------------------------------------------------------------
int main(){

    cout << "\nStart testing.\n";
    cout << "Testing Tensor.\n\n";

    test_constructor_001();
    test_constructor_002();
    test_constructor_003();
    test_constructor_004();
    test_constructor_005();
    test_operatorEquals_001();
    test_operatorEquals_002();
    test_operatorEquals_003();
    test_setItem_001();
    test_setItem_002();
    test_setItem_003();
    test_isTensorEquilateral_001();
    test_isTensorEquilateral_002();
    test_isTensorEquilateral_003();
    test_isTensorEquilateral_004();
    test_transposition_001();
    test_transposition_002();
    test_operatorAssign_001();

    cout << "\nAll test done.\n";
    
    return 0;
}

