#include <bitset>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "core/Tensor.hpp"

using gema::Tensor;

// Formatter specializations for certain used types
namespace std{
    template <size_t N>
    struct formatter<bitset<N>, char> {

        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();
        }

        auto format(const bitset<N>& bitset, format_context& ctx) const {
            return format_to(ctx.out(), "{}", bitset.to_string());
        }
    };
}

#define DEBUG(text)\
    std::cout << text << std::endl;

TEST(tensor_test, constructor_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(5,     {0, 0});
    tensor->setItem(0.55,    {1, 0});
    tensor->setItem(-0,     {0, 1});
    tensor->setItem(-2,      {1, 1});
    tensor->setItem(4.5,   {0, 2});
    tensor->setItem(7,    {1, 2});

    std::vector<int> expected = {2, 3};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 2);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_002){

    const std::vector<uint64_t> dimensionSizes{2, 2, 3};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

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

    std::vector<int> expected = {2, 2, 3};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 3);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_003){

    const std::vector<uint64_t> dimensionSizes{2};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItem(2,     {0});
    tensor->setItem(-0.3,  {1});
    //tensor->setItem(4,     {2}); // proc?, dyt to unese jen 2 prvky

    std::vector<int> expected = {2};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 1);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_004){

    const std::vector<uint64_t> dimensionSizes{2};
    auto tensor = std::make_unique<Tensor<Tensor<int>>>(dimensionSizes);

    auto tensorIn1 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensorIn1->setItems({1, -2});
    auto tensorIn2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensorIn2->setItems({0, 5});

    tensor->setItem(*tensorIn1,  {0});
    tensor->setItem(*tensorIn2,  {1});

    std::vector<int> expected = {2};

    EXPECT_EQ(tensor->getNumberOfDimensions(), 1);

    for(uint64_t i = 0; i < tensor->getNumberOfDimensions(); i++){
        EXPECT_EQ(tensor->getDimensionSizes()[i], expected[i]);
    }
}

TEST(tensor_test, constructor_005){

    const std::vector<uint64_t> dimensionSizes{0};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

    EXPECT_EQ(tensor->getNumberOfDimensions(), 1);
}

TEST(tensor_test, constructor_006){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({2, -3});

    auto result = std::make_unique<Tensor<double>>(*tensor);

    EXPECT_EQ(*tensor, *result);
}

TEST(tensor_test, constructor_007){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({2, -3});

    auto result = std::make_unique<Tensor<double>>(*tensor);
    result->setItem(5, {1});

    EXPECT_NE(*tensor, *result);
}

TEST(tensor_test, constructor_008){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<int>>>(dimensionSizes);

    Tensor<int> tenVal1({2});
    tenVal1.setItems({1, -2});
    Tensor<int> tenVal2({2});
    tenVal2.setItems({0, 4});

    tensor->setItems({tenVal1, tenVal2});

    auto result = std::make_unique<Tensor<Tensor<int>>>(*tensor);

    Tensor<int> tenVal3({2});
    tenVal3.setItems({1, 2});

    result->setItem(tenVal3, {0});

    EXPECT_NE(*tensor, *result);
}

TEST(tensor_test, constructor_009){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<int>*>>(dimensionSizes);

    Tensor<int>* tenVal1 = new Tensor<int>(dimensionSizes);
    tenVal1->setItems({1, -2});
    Tensor<int>* tenVal2 = new Tensor<int>(dimensionSizes);
    tenVal2->setItems({0, 4});

    tensor->setItems({tenVal1, tenVal2});

    auto tensor2 = std::make_unique<Tensor<Tensor<int>*>>(*tensor);

    tensor2->getItem({0})->setItem(6, {0});

    EXPECT_EQ(*tensor, *tensor2);

    delete tenVal1;
    delete tenVal2;
}

TEST(tensor_test, setItem_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

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

    const std::vector<uint64_t> dimensionSizes{3, 2};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);

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

    const std::vector<uint64_t> dimensionSizes{2, 2};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

    tensor->setItems({2, 0, -1, 6.4});
    tensor->setItem(3.3, {0, 1});// swtich coords for little endian

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({2, 3.3, -1, 6.4});
    
    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, isEquilateral_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, isEquilateral_002){

    const std::vector<uint64_t> dimensionSizes{2, 2, 1};
    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);

    tensor->setItems({false, true, false, false});

    EXPECT_EQ(tensor->isEquilateral(), false);
}

TEST(tensor_test, isEquilateral_003){

    const std::vector<uint64_t> dimensionSizes{2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({200});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, isEquilateral_004){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5, -1, 100, 24, -24, 5, 45});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, fillWith_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->fillWith(69);
    
    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({69, 69, 69, 69, 69, 69});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, fillWith_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);

    tensor->fillWith(true);
    
    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true, true, true, true, true, true});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, transposition_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    std::unique_ptr<Tensor<int>> result(tensor->transposition());
    
    const std::vector<uint64_t> expectedDimensionSizes{3, 2};
    auto expected = std::make_unique<Tensor<int>>(expectedDimensionSizes);
    //expected->setItems({0, -1, -2, 5, 100, -16}); //little endian
    expected->setItems({0, 100, 5, -2, -1, -16});

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, transposition_002){

    const std::vector<uint64_t> dimensionSizes{1, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setItems({0, 5});

    std::unique_ptr<Tensor<int>> result(tensor->transposition());
    
    const std::vector<uint64_t> expectedDimensionSizes{2, 1};
    auto expected = std::make_unique<Tensor<int>>(expectedDimensionSizes);
    expected->setItems({0, 5});

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAssign_001){

    const std::vector<uint64_t> dimensionSizes{3, 2};
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

    const std::vector<uint64_t> dimensionSizes{2, 2};
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

    const std::vector<uint64_t> dimensionSizes{2, 2};
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

    const std::vector<uint64_t> dimensionSizes{2};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    
    std::string expected = "{0, 0}";

    EXPECT_EQ(tensor->toString(), expected);

}

TEST(tensor_test, toString_002){

    const std::vector<uint64_t> dimensionSizes{1, 2, 2};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000});
    
    std::string expected = "{{{5.1, 0}, {-1e-06, 5e+05}}}";

    EXPECT_EQ(tensor->toString(), expected);
}

TEST(tensor_test, toString_003){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, 24, -24, 5, 45});
    
    std::string expected = "{{{0, 5}, {-1, 100}}, {{24, -24}, {5, 45}}}";

    EXPECT_EQ(tensor->toString(), expected);
}

TEST(tensor_test, toString_004){

    const std::vector<uint64_t> dimensionSizes{3, 2};

    Tensor<double>* tensor = new Tensor<double>(dimensionSizes);

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
    std::string expected = "{{5, 0.55}, {0, -2}, {4.5, 7}}";

    EXPECT_EQ(tensor->toString(), expected);

    delete tensor;
}



// OPERATOR OVERLOAD TESTS ----------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorEquals_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({0, 5, -1, 100, -2, -16});

    bool result = (*tensor == *tensor2);
    bool expected = true;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_002){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    const std::vector<uint64_t> dimensionSizes2{1, 4};

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes2);
    tensor2->setItems({0, 5, -1, 100, -2, -16});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_003){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true, false});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({true, true});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_004){

    const std::vector<uint64_t> dimensionSizes{1};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({5});

    const std::vector<uint64_t> dimensionSizes2{1, 1};

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes2);
    tensor2->setItems({5});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_005){

    const std::vector<uint64_t> dimensionSizes{1, 2, 2};
    
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000});
    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({5.1, -0, -0.000001, 500000});

    bool expected = true;
    EXPECT_EQ(expected, (*tensor == *tensor));
}

TEST(tensor_test, operatorEquals_006){

    const std::vector<uint64_t> dimensionSizes{3, 2, 1};
    
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5.1, 0, -0.000001, 500000, 1, -1});
    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({5.1, 0, -0, 500000, 1, -1});

    bool expected = false;
    EXPECT_EQ(expected, (*tensor == *tensor2));
}

TEST(tensor_test, operatorEquals_007){

    const std::vector<uint64_t> dimensionSizes{1};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);

    bool expected = true;
    EXPECT_EQ(expected, (*tensor == *tensor2));
}

// (+) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorAdd_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({3, -3, -3, 0, -7, -16});

    std::unique_ptr<Tensor<int>> result(*tensor + *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAdd_002){

    const std::vector<uint64_t> dimensionSizes{1, 2};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0.1, 5.8});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({0.2, -8.3});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({0.3, -2.5});

    std::unique_ptr<Tensor<float>> result(*tensor + *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAdd_003){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({false});

    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true});
    
    std::unique_ptr<Tensor<bool>> result(*tensor + *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAdd_004){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor->setItems({"yee"});

    auto tensor2 = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor2->setItems({"haw"});

    auto expected = std::make_unique<Tensor<std::string>>(dimensionSizes);
    expected->setItems({"yeehaw"});
    
    std::unique_ptr<Tensor<std::string>> result(*tensor + *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAddValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    int value = -4;

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({-4, 1, -5, 96, -6, -20});

    std::unique_ptr<Tensor<int>> result(*tensor + value);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAddValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    int value = -4;

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({-4, 1, -5, 96, -6, -20});

    std::unique_ptr<Tensor<int>> result(value + *tensor);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorAddAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({3, -3, -3, 0, -7, -16});

    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssign_002){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({false});

    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true});
    
    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssign_003){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor->setItems({"yee"});

    auto tensor2 = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor2->setItems({"haw"});

    auto expected = std::make_unique<Tensor<std::string>>(dimensionSizes);
    expected->setItems({"yeehaw"});
    
    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5., -1.});

    double value = 3.;

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({8., 2.});
    
    *tensor += value;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssignValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor->setItems({"drg", "fsef", "sdsc", "_", "", "\n$"});

    std::string value = "_red";

    auto expected = std::make_unique<Tensor<std::string>>(dimensionSizes);
    expected->setItems({"drg_red", "fsef_red", "sdsc_red", "__red", "_red", "\n$_red"});
    
    *tensor += value;

    EXPECT_EQ(*tensor, *expected);
}

// (-) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorSubstract_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({-3, 13, 1, 200, 3, -16});

    std::unique_ptr<Tensor<int>> result(*tensor - *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorSubstract_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true, true, false});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({true, false, false});

    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({false, true, false});

    std::unique_ptr<Tensor<bool>> result(*tensor - *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorSubstractValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    int value = -4;

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({4, 9, 3, 104, 2, -12});

    std::unique_ptr<Tensor<int>> result(*tensor - value);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorSubstractValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({-4, -9, -3, -104, -2, 12});

    std::unique_ptr<Tensor<int>> result(-4 - *tensor);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorSubstractAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({-3, 13, 1, 200, 3, -16});

    *tensor -= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorSubstractAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0.2, 5.01});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({3e+2, -8.});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({-299.8, 13.01});

    *tensor -= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorSubstractAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({5., -1.});

    double value = 3.;

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({2., -4.});
    
    *tensor -= value;

    EXPECT_EQ(*tensor, *expected);
}

// (*) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorMultiply_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({0, -40, 2, -10000, 10, 0});

    std::unique_ptr<Tensor<int>> result(*tensor * *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorMultiply_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({0., 5., -1., 100., -2., -16.});

    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({3., -8., -2., -100., -5., 0.});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({0., -40., 2., -10000., 10., 0.});

    std::unique_ptr<Tensor<double>> result(*tensor * *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorMultiplyValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({0., 5., -1., 100., -2., -16.});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({0., -2.5, 0.5, -50., 1., 8.});

    std::unique_ptr<Tensor<double>> result(*tensor * -0.5);

    EXPECT_EQ(*result, *expected);
}

// (|) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseOr_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({3, -3, -1, -4, -1, -16});

    std::unique_ptr<Tensor<int>> result(*tensor | *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseOr_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0., std::bit_cast<float>(0x40140000)});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({3., std::bit_cast<float>(0xC0200000)});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({3., std::bit_cast<float>(0xC0340000)});

    std::unique_ptr<Tensor<float>> result(*tensor | *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseOr_003){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setItems({0., std::bit_cast<double>(0x4014000000000000)});

    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setItems({3., std::bit_cast<double>(0xC020000000000000)});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setItems({3., std::bit_cast<double>(0xC034000000000000)});

    std::unique_ptr<Tensor<double>> result(*tensor | *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseOrAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setItems({3, -3, -1, -4, -1, -16});

    *tensor |= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseOrAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0., std::bit_cast<float>(0x40140000)});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({3., std::bit_cast<float>(0xC0200000)});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({3., std::bit_cast<float>(0xC0340000)});

    *tensor |= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

// (&) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseAnd_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({0, 0, -2, 4, -6, 0});

    std::unique_ptr<Tensor<int>> result(*tensor & *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseAnd_002){

    const std::vector<uint64_t> dimensionSizes{3, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setItems({true, false, false});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setItems({true, true, false});

    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setItems({true, false, false});

    std::unique_ptr<Tensor<bool>> result(*tensor & *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseAndAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setItems({0, 0, -2, 4, -6, 0});

    *tensor &= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseAndAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0., std::bit_cast<float>(0x40140000)});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({3., std::bit_cast<float>(0xC0200000)});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({0., std::bit_cast<float>(0x40000000)});

    *tensor &= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

// (^) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseXor_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setItems({3, -3, 1, -8, 5, -16});

    std::unique_ptr<Tensor<int>> result(*tensor ^ *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseXor_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    tensor->setItems({0b1100, 0b1010});

    auto tensor2 = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    tensor2->setItems({0b1010, 0b0011});

    auto expected = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    expected->setItems({0b0110, 0b1001});

    std::unique_ptr<Tensor<std::bitset<4>>> result(*tensor ^ *tensor2);

    EXPECT_EQ(*result, *expected);
}

TEST(tensor_test, operatorBitwiseXorAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setItems({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setItems({3, -3, 1, -8, 5, -16});

    *tensor ^= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseXorAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({0., std::bit_cast<float>(0x40140000)});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setItems({3., std::bit_cast<float>(0xC0200000)});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({3., std::bit_cast<float>(0x80340000)});

    *tensor ^= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

// (~) --------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorNegation_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setItems({0, 5, -1, 100, -2, -16});

    std::unique_ptr<Tensor<int64_t>> tensor2(~*tensor);

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setItems({-1, -6, 0, -101, 1, 15});

    EXPECT_EQ(*tensor2, *expected);
}

TEST(tensor_test, operatorNegation_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setItems({std::bit_cast<float>(0x40140000), std::numeric_limits<float>::infinity(), 2.520999908447265625, 2.51});

    std::unique_ptr<Tensor<float>> tensor2(~*tensor);

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setItems({std::bit_cast<float>(0xBFEBFFFF), std::bit_cast<float>(0x807FFFFF), -1.73949992656707763671875, -1.7449999});

    EXPECT_EQ(*tensor2, *expected);
}

// OPERATOR OVERLOAD TESTS END



TEST(tensor_test, showDebug){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    auto expected = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    expected->setItems({0b0110, 0b1001, 0b0101, 0b1010, 0b1110, 0b0111});

    //std::cout << "is formattable bitset: " << std::boolalpha << gema::is_formattable<std::bitset<4>> << std::endl;
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
