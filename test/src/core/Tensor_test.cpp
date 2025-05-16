#include <iostream>
#include <vector>
#include <memory>

#include <gtest/gtest.h>

#include "core/Tensor.hpp"

using GeMa::Tensor;
using std::vector, std::make_unique, std::cout, std::endl, std::string;

TEST(tensor_test, constructor_001){

    const vector<int> dimensionSizes{2, 3};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {0, 1});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {0, 2});
    tensor->setItem(7,    {1, 2});

    vector<int> expected = {2, 3};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 2);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_002){

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

    EXPECT_EQ(tensor->getNumberOfDimensions(), 3);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_003){

    const vector<int> dimensionSizes{2};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(2,     {0});
    tensor->setItem(-0.3,  {1});
    //tensor->setItem(4,     {2}); // proc?, dyt to unese jen 2 prvky

    vector<int> expected = {2};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 1);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_004){

    const vector<int> dimensionSizes{2};
    auto tensor = make_unique<Tensor<Tensor<int>>>(dimensionSizes);

    auto tensorIn1 = make_unique<Tensor<int>>(dimensionSizes);
    tensorIn1->setItems({1, -2});
    auto tensorIn2 = make_unique<Tensor<int>>(dimensionSizes);
    tensorIn2->setItems({0, 5});

    tensor->setItem(*tensorIn1,  {0});
    tensor->setItem(*tensorIn2,  {1});

    vector<int> expected = {2};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 1);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_005){

    const vector<int> dimensionSizes{0};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    EXPECT_EQ(tensor->getNumberOfDimensions(), 1);
}

TEST(tensor_test, constructor_006){

    const vector<int> dimensionSizes{2};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({2, -3});

    auto result = make_unique<Tensor<double>>(*tensor);

    EXPECT_EQ(*tensor, *result);
}

TEST(tensor_test, constructor_007){

    const vector<int> dimensionSizes{2};

    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({2, -3});

    auto result = make_unique<Tensor<double>>(*tensor);
    result->setItem(5, {1});

    EXPECT_NE(*tensor, *result);
}

TEST(tensor_test, constructor_008){

    const vector<int> dimensionSizes{2};

    auto tensor = make_unique<Tensor<Tensor<int>>>(dimensionSizes);

    Tensor<int> tenVal1({2});
    tenVal1.setItems({1, -2});
    Tensor<int> tenVal2({2});
    tenVal2.setItems({0, 4});

    tensor->setItems({tenVal1, tenVal2});

    auto result = make_unique<Tensor<Tensor<int>>>(*tensor);

    Tensor<int> tenVal3({2});
    tenVal3.setItems({1, 2});

    result->setItem(tenVal3, {0});

    EXPECT_NE(*tensor, *result);
}

TEST(tensor_test, constructor_009){

    const vector<int> dimensionSizes{2};

    auto tensor = make_unique<Tensor<Tensor<int>*>>(dimensionSizes);

    Tensor<int>* tenVal1 = new Tensor<int>(dimensionSizes);
    tenVal1->setItems({1, -2});
    Tensor<int>* tenVal2 = new Tensor<int>(dimensionSizes);
    tenVal2->setItems({0, 4});

    tensor->setItems({tenVal1, tenVal2});

    auto tensor2 = make_unique<Tensor<Tensor<int>*>>(*tensor);

    tensor2->getItem({0})->setItem(6, {0});

    EXPECT_EQ(*tensor, *tensor2);

    delete tenVal1;
    delete tenVal2;
}

TEST(tensor_test, operatorEquals_001){

    const vector<int> dimensionSizes{1, 2, 2};
    
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000});
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({5.1, -0, -0.000001, 500000});

    bool expected = true;
    EXPECT_EQ(expected, (*tensor == *tensor));
}

TEST(tensor_test, operatorEquals_002){

    const vector<int> dimensionSizes{3, 2, 1};
    
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000, 1, -1});
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({5.1, 0, -0, 500000, 1, -1});

    bool expected = false;
    EXPECT_EQ(expected, (*tensor == *tensor2));
}

TEST(tensor_test, operatorEquals_003){

    const vector<int> dimensionSizes{1};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);

    bool expected = true;
    EXPECT_EQ(expected, (*tensor == *tensor2));
}

TEST(tensor_test, setItem_001){

    const vector<int> dimensionSizes{2, 3};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {0, 1});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {0, 2});
    tensor->setItem(7,    {1, 2});

    EXPECT_EQ(tensor->getItem({0, 0}), 5);
    EXPECT_EQ(tensor->getItem({1, 0}), 0.55);
    EXPECT_EQ(tensor->getItem({0, 1}), 0);
    EXPECT_EQ(tensor->getItem({1, 1}), -2);
    EXPECT_EQ(tensor->getItem({0, 2}), 4.5);
    EXPECT_EQ(tensor->getItem({1, 2}), 7);
}

TEST(tensor_test, setItem_002){

    const vector<int> dimensionSizes{3, 2};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);

    // Little endian
    /*tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {2, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});
    tensor->setItem(7,    {2, 1});*/

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {0, 1});
    tensor->setItem(-0,     {1, 0});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {2, 0});
    tensor->setItem(7,    {2, 1});

    tensor2->setItems({5, 0.55, -0, -2, 4.5, 7});

    /*tensor->forEach([](double& item){
        cout << item << "; ";
    });
    cout << endl;*/
    EXPECT_EQ(*tensor, *tensor2);
}

TEST(tensor_test, setItem_003){

    const vector<int> dimensionSizes{2, 2};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItems({2, 0, -1, 6.4});
    tensor->setItem(3.3, {0, 1});// swtich coords for little endian

    auto expected = make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({2, 3.3, -1, 6.4});
    
    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, isEquilateral_001){

    const vector<int> dimensionSizes{2, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, isEquilateral_002){

    const vector<int> dimensionSizes{2, 2, 1};
    auto tensor = make_unique<Tensor<bool>>(dimensionSizes);

    tensor->setItems({false, true, false, false});

    EXPECT_EQ(tensor->isEquilateral(), false);
}

TEST(tensor_test, isEquilateral_003){

    const vector<int> dimensionSizes{2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({200});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, isEquilateral_004){

    const vector<int> dimensionSizes{2, 2, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100, 24, -24, 5, 45});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, fillWith_001){

    const vector<int> dimensionSizes{2, 3};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->fillWith(69);
    
    auto expected = make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({69, 69, 69, 69, 69, 69});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, fillWith_002){

    const vector<int> dimensionSizes{2, 3};
    auto tensor = make_unique<Tensor<bool>>(dimensionSizes);

    tensor->fillWith(true);
    
    auto expected = make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true, true, true, true, true, true});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, transposition_001){

    const vector<int> dimensionSizes{2, 3};

    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    Tensor<int>* result = tensor->transposition();
    
    const vector<int> expectedDimensionSizes{3, 2};
    auto expected = make_unique<Tensor<int>>(expectedDimensionSizes);
    //expected->setItems({0, -1, -2, 5, 100, -16}); //little endian
    expected->setItems({0, 100, 5, -2, -1, -16});

    EXPECT_EQ(*result, *expected);
    delete result;
}

TEST(tensor_test, transposition_002){

    const vector<int> dimensionSizes{1, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5});

    Tensor<int>* result = tensor->transposition();
    
    const vector<int> expectedDimensionSizes{2, 1};
    auto expected = make_unique<Tensor<int>>(expectedDimensionSizes);
    expected->setItems({0, 5});

    EXPECT_EQ(*result, *expected);
    delete result;
}

TEST(tensor_test, operatorAssign_001){

    const vector<int> dimensionSizes{3, 2};
    auto tensor = new Tensor<double>(dimensionSizes);

    // Little endian
    /*tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {2, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});
    tensor->setItem(7,    {2, 1});*/

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {0, 1});
    tensor->setItem(-0,     {1, 0});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {2, 0});
    tensor->setItem(7,    {2, 1});

    Tensor<double> tensor2;
    tensor2 = *tensor;

    tensor->setItem(8, {0, 0});

    auto expected = new Tensor<double>(dimensionSizes);
    expected->setItems({5, 0.55, -0, -2, 4.5, 7});
    EXPECT_EQ(*expected, tensor2);

    delete expected;
    delete tensor;
}

TEST(tensor_test, operatorAssign_002){

    const vector<int> dimensionSizes{2, 2};
    auto tensor = new Tensor<double>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});

    Tensor<double> tensor2;
    tensor2 = *tensor;

    tensor->setItem(8, {0, 0});

    EXPECT_NE(*tensor, tensor2);

    delete tensor;
}

TEST(tensor_test, operatorAssign_003){

    const vector<int> dimensionSizes{2, 2};
    Tensor<double>* tensor = new Tensor<double>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});
    
    Tensor<double>* tensor2;
    tensor2 = tensor;

    tensor->setItem(1.1, {0, 0});

    Tensor<double>* expected = new Tensor<double>(dimensionSizes);
    expected->setItems({5, 0.55, -2, 4.5});
    EXPECT_NE(*expected, *tensor2);

    delete expected;
    delete tensor;
}

TEST(tensor_test, toString_001){

    const vector<int> dimensionSizes{2};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    
    string expected = "{0, 0}";

    EXPECT_EQ(tensor->toString(), expected);

}

TEST(tensor_test, toString_002){

    const vector<int> dimensionSizes{1, 2, 2};
    auto tensor = make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000});
    
    string expected = "{{{5.1, 0}, {-1e-06, 5e+05}}}";

    EXPECT_EQ(tensor->toString(), expected);
}

TEST(tensor_test, toString_003){

    const vector<int> dimensionSizes{2, 2, 2};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, 24, -24, 5, 45});
    
    string expected = "{{{0, 5}, {-1, 100}}, {{24, -24}, {5, 45}}}";

    EXPECT_EQ(tensor->toString(), expected);
}

TEST(tensor_test, toString_004){

    const vector<int> dimensionSizes{3, 2};

    auto tensor = new Tensor<double>(dimensionSizes);

    // Little endian approach
    /*tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {2, 0});
    tensor->setItem(-2,      {0, 1});
    tensor->setItem(4.5,   {1, 1});
    tensor->setItem(7,    {2, 1});*/

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {0, 1});
    tensor->setItem(-0,     {1, 0});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {2, 0});
    tensor->setItem(7,    {2, 1});
    
    //string expected = "{{5, 0.55, 0}, {-2, 4.5, 7}}";
    string expected = "{{5, 0.55}, {0, -2}, {4.5, 7}}";

    EXPECT_EQ(tensor->toString(), expected);
}

TEST(tensor_test, operatorCompare_001){

    const vector<int> dimensionSizes{2, 3};

    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({0, 5, -1, 100, -2, -16});

    bool result = (*tensor == *tensor2);
    bool expected = true;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorCompare_002){

    const vector<int> dimensionSizes{2, 2};

    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    const vector<int> dimensionSizes2{1, 4};

    auto tensor2 = make_unique<Tensor<int>>(dimensionSizes2);
    tensor2->setItems({0, 5, -1, 100, -2, -16});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorCompare_003){

    const vector<int> dimensionSizes{2, 1};

    auto tensor = make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true, false});

    auto tensor2 = make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({true, true});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorCompare_004){

    const vector<int> dimensionSizes{1};

    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({5});

    const vector<int> dimensionSizes2{1, 1};

    auto tensor2 = make_unique<Tensor<int>>(dimensionSizes2);
    tensor2->setItems({5});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorAdd_001){

    const vector<int> dimensionSizes{2, 3};

    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({3, -3, -3, 0, -7, -16});

    auto result = *tensor + *tensor2;

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAdd_002){

    const vector<int> dimensionSizes{1, 2};

    auto tensor = make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0.1, 5.8});

    auto tensor2 = make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({0.2, -8.3});

    auto expected = make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({0.3, -2.5});

    auto result = *tensor + *tensor2;

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAdd_003){

    const vector<int> dimensionSizes{1, 1};

    auto tensor = make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true});

    auto tensor2 = make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({false});

    auto expected = make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true});
    
    auto result = *tensor + *tensor2;

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAddAssign_001){

    const vector<int> dimensionSizes{2, 3};

    auto tensor = make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({3, -3, -3, 0, -7, -16});

    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssign_002){

    const vector<int> dimensionSizes{1, 1};

    auto tensor = make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true});

    auto tensor2 = make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({false});

    auto expected = make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true});
    
    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, showDebug){

    const vector<int> dimensionSizes{2, 3};
    auto tensor = make_unique<Tensor<int>>(dimensionSizes);

}

/*//Our main <3 -----------------------------------------------------------------------------------------------------------------
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
    test_isEquilateral_001();
    test_isEquilateral_002();
    test_isEquilateral_003();
    test_isEquilateral_004();
    test_transposition_001();
    test_transposition_002();
    test_operatorAssign_001();
    test_operatorAssign_002();
    test_operatorAssign_003();
    test_toString_001();
    test_toString_002();
    test_toString_003();
    test_toString_004();

    cout << "\nAll test done.\n";
    
    return 0;
}*/
