#include <bitset>
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
    tensorIn1->setData({1, -2});
    auto tensorIn2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensorIn2->setData({0, 5});

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
    tensor->setData({2, -3});

    auto result = std::make_unique<Tensor<double>>(*tensor);

    EXPECT_EQ(*tensor, *result);
}

TEST(tensor_test, constructor_007){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({2, -3});

    auto result = std::make_unique<Tensor<double>>(*tensor);
    result->setItem(5, {1});

    EXPECT_NE(*tensor, *result);
}

TEST(tensor_test, constructor_008){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<int>>>(dimensionSizes);

    Tensor<int> tenVal1({2});
    tenVal1.setData({1, -2});
    Tensor<int> tenVal2({2});
    tenVal2.setData({0, 4});

    tensor->setData({tenVal1, tenVal2});

    auto result = std::make_unique<Tensor<Tensor<int>>>(*tensor);

    Tensor<int> tenVal3({2});
    tenVal3.setData({1, 2});

    result->setItem(tenVal3, {0});

    EXPECT_NE(*tensor, *result);
}

TEST(tensor_test, constructor_009){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<int>*>>(dimensionSizes);

    Tensor<int>* tenVal1 = new Tensor<int>(dimensionSizes);
    tenVal1->setData({1, -2});
    Tensor<int>* tenVal2 = new Tensor<int>(dimensionSizes);
    tenVal2->setData({0, 4});

    tensor->setData({tenVal1, tenVal2});

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

    tensor2->setData({5, 0.55, -0, -2, 4.5, 7});

    /*tensor->forEach([](double& item){
        cout << item << "; ";
    });
    cout << endl;*/
    EXPECT_EQ(*tensor, *tensor2);
}

TEST(tensor_test, setItem_003){

    const std::vector<uint64_t> dimensionSizes{2, 2};
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);

    tensor->setData({2, 0, -1, 6.4});
    tensor->setItem(3.3, {0, 1});// swtich coords for little endian

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({2, 3.3, -1, 6.4});
    
    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, isEquilateral_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setData({0, 5, -1, 100});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, isEquilateral_002){

    const std::vector<uint64_t> dimensionSizes{2, 2, 1};
    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);

    tensor->setData({false, true, false, false});

    EXPECT_EQ(tensor->isEquilateral(), false);
}

TEST(tensor_test, isEquilateral_003){

    const std::vector<uint64_t> dimensionSizes{2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setData({200});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, isEquilateral_004){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setData({0, 5, -1, 100, 24, -24, 5, 45});

    EXPECT_EQ(tensor->isEquilateral(), true);
}

TEST(tensor_test, fillWith_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->fillWith(69);
    
    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({69, 69, 69, 69, 69, 69});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, fillWith_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);

    tensor->fillWith(true);
    
    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setData({true, true, true, true, true, true});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, transposition_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    Tensor<int> result(tensor->transposition());
    
    const std::vector<uint64_t> expectedDimensionSizes{3, 2};
    auto expected = std::make_unique<Tensor<int>>(expectedDimensionSizes);
    //expected->setItems({0, -1, -2, 5, 100, -16}); //little endian
    expected->setData({0, 100, 5, -2, -1, -16});

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, transposition_002){

    const std::vector<uint64_t> dimensionSizes{1, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    tensor->setData({0, 5});

    Tensor<int> result(tensor->transposition());
    
    const std::vector<uint64_t> expectedDimensionSizes{2, 1};
    auto expected = std::make_unique<Tensor<int>>(expectedDimensionSizes);
    expected->setData({0, 5});

    EXPECT_EQ(result, *expected);
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
    expected->setData({5, 0.55, -0, -2, 4.5, 7});
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
    expected->setData({5, 0.55, -2, 4.5});
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
    tensor->setData({5.1, 0, -0.000001, 500000});
    
    std::string expected = "{{{5.1, 0}, {-1e-06, 5e+05}}}";

    EXPECT_EQ(tensor->toString(), expected);
}

TEST(tensor_test, toString_003){

    const std::vector<uint64_t> dimensionSizes{2, 2, 2};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, 24, -24, 5, 45});
    
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
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({0, 5, -1, 100, -2, -16});

    bool result = (*tensor == *tensor2);
    bool expected = true;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_002){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    const std::vector<uint64_t> dimensionSizes2{1, 4};

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes2);
    tensor2->setData({0, 5, -1, 100, -2, -16});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_003){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true, false});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setData({true, true});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_004){

    const std::vector<uint64_t> dimensionSizes{1};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({5});

    const std::vector<uint64_t> dimensionSizes2{1, 1};

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes2);
    tensor2->setData({5});

    bool result = (*tensor == *tensor2);
    bool expected = false;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorEquals_005){

    const std::vector<uint64_t> dimensionSizes{1, 2, 2};
    
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({5.1, 0, -0.000001, 500000});
    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setData({5.1, -0, -0.000001, 500000});

    bool expected = true;
    EXPECT_EQ(expected, (*tensor == *tensor));
}

TEST(tensor_test, operatorEquals_006){

    const std::vector<uint64_t> dimensionSizes{3, 2, 1};
    
    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({5.1, 0, -0.000001, 500000, 1, -1});
    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setData({5.1, 0, -0, 500000, 1, -1});

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

// (+) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorAdd_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = Tensor<int>(dimensionSizes);
    tensor2.setData({3, -8, -2, -100, -5, 0});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({3, -3, -3, 0, -7, -16});

    Tensor<int> result = tensor + tensor2;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorAdd_002){

    const std::vector<uint64_t> dimensionSizes{1, 2};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setData({0.1, 5.8});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setData({0.2, -8.3});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setData({0.3, -2.5});

    Tensor<float> result(*tensor + *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorAdd_003){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setData({false});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({1});
    
    Tensor<int> result(*tensor + *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorAdd_004){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor->setData({"yee"});

    auto tensor2 = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor2->setData({"haw"});

    auto expected = std::make_unique<Tensor<std::string>>(dimensionSizes);
    expected->setData({"yeehaw"});
    
    Tensor<std::string> result(*tensor + *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorAddValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    int value = -4;

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({-4, 1, -5, 96, -6, -20});

    Tensor<int> result(*tensor + value);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorAddValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    int value = -4;

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({-4, 1, -5, 96, -6, -20});

    Tensor<int> result(value + *tensor);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorAddAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({3, -3, -3, 0, -7, -16});

    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssign_002){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setData({false});

    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setData({true});
    
    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssign_003){

    const std::vector<uint64_t> dimensionSizes{1, 1};

    auto tensor = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor->setData({"yee"});

    auto tensor2 = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor2->setData({"haw"});

    auto expected = std::make_unique<Tensor<std::string>>(dimensionSizes);
    expected->setData({"yeehaw"});
    
    *tensor += *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({5., -1.});

    double value = 3.;

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({8., 2.});
    
    *tensor += value;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorAddAssignValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<std::string>>(dimensionSizes);
    tensor->setData({"drg", "fsef", "sdsc", "_", "", "\n$"});

    std::string value = "_red";

    auto expected = std::make_unique<Tensor<std::string>>(dimensionSizes);
    expected->setData({"drg_red", "fsef_red", "sdsc_red", "__red", "_red", "\n$_red"});
    
    *tensor += value;

    EXPECT_EQ(*tensor, *expected);
}

// (-) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorSubstract_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({-3, 13, 1, 200, 3, -16});

    Tensor<int> result(*tensor - *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorSubstract_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true, true, false});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setData({true, false, false});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({0, 1, 0});

    Tensor<int> result(*tensor - *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorSubstractValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    int value = -4;

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({4, 9, 3, 104, 2, -12});

    Tensor<int> result(*tensor - value);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorSubstractValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({-4, -9, -3, -104, -2, 12});

    Tensor<int> result = -4 - *tensor;

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorSubstractAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({-3, 13, 1, 200, 3, -16});

    *tensor -= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorSubstractAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor->setData({0.2, 5.01});

    auto tensor2 = std::make_unique<Tensor<float>>(dimensionSizes);
    tensor2->setData({3e+2, -8.});

    auto expected = std::make_unique<Tensor<float>>(dimensionSizes);
    expected->setData({-299.8, 13.01});

    *tensor -= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorSubstractAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({5., -1.});

    double value = 3.;

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({2., -4.});
    
    *tensor -= value;

    EXPECT_EQ(*tensor, *expected);
}

// (*) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorMultiply_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = Tensor<int>(dimensionSizes);
    tensor2.setData({3, -8, -2, -100, -5, 0});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0, -40, 2, -10000, 10, 0});

    Tensor<int> result = tensor * tensor2;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorMultiply_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({0., 5., -1., 100., -2., -16.});

    auto tensor2 = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor2->setData({3., -8., -2., -100., -5., 0.});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({0., -40., 2., -10000., 10., 0.});

    Tensor<double> result(*tensor * *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorMultiplyValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({0., 5., -1., 100., -2., -16.});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({0., -2.5, 0.5, -50., 1., 8.});

    Tensor<double> result(*tensor * -0.5);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorMultiplyAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({0, -40, 2, -10000, 10, 0});

    *tensor *= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorMultiplyAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({5., -1.});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({15., -3.});
    
    *tensor *= 3.;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorMultiplyAssignValue_002){

    const std::vector<uint64_t> dimensionSizes{2};
    const std::vector<uint64_t> dimensionSizesInner{1};

    auto tensor = std::make_unique<Tensor<Tensor<int>>>(dimensionSizes);
    auto tensorInner = Tensor<int>(dimensionSizesInner);
    tensorInner.setData({-2});
    auto tensorInner2 = Tensor<int>(dimensionSizesInner);
    tensorInner2.setData({1});
    tensor->setData({tensorInner, tensorInner2});

    auto expected = std::make_unique<Tensor<Tensor<int>>>(dimensionSizes);
    auto expectedInner = Tensor<int>(dimensionSizesInner);
    expectedInner.setData({-6});
    auto expectedInner2 = Tensor<int>(dimensionSizesInner);
    expectedInner2.setData({3});
    expected->setData({expectedInner, expectedInner2});

    auto valueTensor = std::make_unique<Tensor<int>>(dimensionSizesInner);
    valueTensor->setData({3});
    
    *tensor *= *valueTensor;

    EXPECT_EQ(*tensor, *expected);
}

// (/) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorDivide_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -1, -2, 100});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({0, -5, 0, 1});

    Tensor<int> result(*tensor / *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorDivide_002){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<double>(dimensionSizes);
    tensor.setData({1., 5., -1., 100.});

    auto tensor2 = Tensor<double>(dimensionSizes);
    tensor2.setData({3., -2., -2., 100.});

    auto expected = Tensor<double>(dimensionSizes);
    expected.setData({1./3., 5./(-2.), 0.5, 1.});

    Tensor<double> result = (tensor / tensor2);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorDivideValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<double>>(dimensionSizes);
    tensor->setData({0., 5., -1., 100., -2., -16.});

    auto expected = std::make_unique<Tensor<double>>(dimensionSizes);
    expected->setData({0., 5./-0.5, 2., -200., 4., 32.});

    Tensor<double> result(*tensor / -0.5);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorDivideValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 1, 1};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({-1, 5});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({(-2)/(-1), (-2)/5});

    Tensor<int> result = (-2) / *tensor;

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorDivideAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<Tensor<char>>>(dimensionSizes);
    tensor->setData({Tensor<char>({2}).setData({0x00, 0x01}), Tensor<char>({2}).setData({-0x10, 0x78})});

    auto tensor2 = std::make_unique<Tensor<Tensor<char>>>(dimensionSizes);
    tensor2->setData({Tensor<char>({2}).setData({-0x10, 0x01}), Tensor<char>({2}).setData({0x08, 0x08})});

    auto expected = std::make_unique<Tensor<Tensor<char>>>(dimensionSizes);
    expected->setData({Tensor<char>({2}).setData({0x00, 0x01}), Tensor<char>({2}).setData({-0x02, 0x0f})});

    *tensor /= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorDivideAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setData({4, -1});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setData({4/3, -(1)/(-3)});
    
    *tensor /= 3;

    EXPECT_EQ(*tensor, *expected);
}

// (%) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorModulo_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -1, -2, 3});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({0, 0, -1, 1});

    Tensor<int> result(*tensor % *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorModulo_002){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<double>(dimensionSizes);
    tensor.setData({0., 5.1, -1., -0.0});

    auto tensor2 = Tensor<double>(dimensionSizes);
    tensor2.setData({3.5, -3.0, -2., 1.0});

    auto expected = Tensor<double>(dimensionSizes);
    expected.setData({0., 2.1, -1., -0.});

    Tensor<double> result = tensor % tensor2;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorModuloValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<short>(dimensionSizes);
    tensor.setData({0, 5, -1, 100, -24, -16});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0, 0, -1, 0, -4, -1});

    Tensor<int> result(tensor % (short)-5);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorModuloValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<char>(dimensionSizes);
    tensor.setData({1, 5, -1, 100, -2, -16});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0, 3, 0, 3, 1, 3});

    Tensor<int> result((char)3 % tensor);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorModuloAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = Tensor<Tensor<char>>(dimensionSizes);
    tensor.setData({Tensor<char>({2}).setData({0x00, 0x01}), Tensor<char>({2}).setData({-0x10, 0x78})});

    auto tensor2 = Tensor<Tensor<char>>(dimensionSizes);
    tensor2.setData({Tensor<char>({2}).setData({-0x10, 0x01}), Tensor<char>({2}).setData({0x09, 0x08})});

    auto expected = Tensor<Tensor<char>>(dimensionSizes);
    expected.setData({Tensor<char>({2}).setData({0x00, 0x00}), Tensor<char>({2}).setData({-0x07, 0x00})});

    tensor %= tensor2;

    EXPECT_EQ(tensor, expected);
}

TEST(tensor_test, operatorModuloAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = Tensor<int64_t>(dimensionSizes);
    tensor.setData({4, -1});

    auto expected = Tensor<int64_t>(dimensionSizes);
    expected.setData({1, -1});
    
    tensor %= 3;

    EXPECT_EQ(tensor, expected);
}

// (&&) -----------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorAnd_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<bool>(dimensionSizes);
    tensor.setData({false, false, true, true});

    auto tensor2 = Tensor<bool>(dimensionSizes);
    tensor2.setData({false, true, false, true});

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({false, false, false, true});

    Tensor<bool> result(tensor && tensor2);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorAnd_002){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = Tensor<Tensor<bool>>(dimensionSizes);
    tensor.setData({Tensor<bool>({1}).setData({true}), Tensor<bool>({1}).setData({false})});

    auto tensor2 = Tensor<Tensor<bool>>(dimensionSizes);
    tensor2.setData({Tensor<bool>({1}).setData({true}), Tensor<bool>({1}).setData({true})});

    auto expected = Tensor<Tensor<bool>>(dimensionSizes);
    expected.setData({Tensor<bool>({1}).setData({true}), Tensor<bool>({1}).setData({false})});

    Tensor<Tensor<bool>> result(tensor && tensor2);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorAndValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<short>(dimensionSizes);
    tensor.setData({0, 5, -1, 100, 0, -16});

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({false, true, true, true, false, true});

    Tensor<bool> result(tensor && (short)-5);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorAndValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<float>(dimensionSizes);
    tensor.setData({1., 0., -1., 100., -2., -0.});

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({true, false, true, true, true, false});

    Tensor<bool> result((float)3 && tensor);

    EXPECT_EQ(result, expected);
}

// (||) -----------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorOr_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<bool>(dimensionSizes);
    tensor.setData({false, false, true, true});

    auto tensor2 = Tensor<bool>(dimensionSizes);
    tensor2.setData({false, true, false, true});

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({false, true, true, true});

    Tensor<bool> result(tensor || tensor2);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorOrValue_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<short>(dimensionSizes);
    tensor.setData({0, 5, -1, 100, 0, -16});

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({true, true, true, true, true, true});

    Tensor<bool> result(tensor || (short)-5);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorOrValue_002){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<float>(dimensionSizes);
    tensor.setData({1., 0., -1., 100., -2., -0.});

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({true, false, true, true, true, false});

    Tensor<bool> result((float)0 || tensor);

    EXPECT_EQ(result, expected);
}

// (|) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseOr_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({3, -3, -1, -4, -1, -16});

    Tensor<int> result(*tensor | *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseOr_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<unsigned int>>(dimensionSizes);
    tensor->setData({0, std::bit_cast<unsigned int>(0x40140000)});

    auto tensor2 = std::make_unique<Tensor<unsigned int>>(dimensionSizes);
    tensor2->setData({3, std::bit_cast<unsigned int>(0xC0200000)});

    auto expected = std::make_unique<Tensor<unsigned int>>(dimensionSizes);
    expected->setData({3, std::bit_cast<unsigned int>(0xC0340000)});

    Tensor<unsigned int> result(*tensor | *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseOr_003){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setData({0, std::bit_cast<int64_t>(0x4014000000000000)});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setData({3, std::bit_cast<int64_t>(0xC020000000000000)});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setData({3, std::bit_cast<int64_t>(0xC034000000000000)});

    Tensor<int64_t> result(*tensor | *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseOrValue_001){

    const std::vector<uint64_t> dimensionSizes{1, 3};

    auto tensor = std::make_unique<Tensor<uint64_t>>(dimensionSizes);
    tensor->setData({std::bit_cast<uint64_t>(0.), 
                        std::bit_cast<uint64_t>(4000000.211), 
                        std::bit_cast<uint64_t>(-1.)});

    auto expected = std::make_unique<Tensor<uint64_t>>(dimensionSizes);
    expected->setData({std::bit_cast<uint64_t>(-0.5), 
                        std::bit_cast<uint64_t>(0xFFEE84801B020C4A), 
                        std::bit_cast<uint64_t>(0xBFF0000000000000)});

    Tensor<uint64_t> result(*tensor | std::bit_cast<uint64_t>(-0.5));

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseOrValue_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = std::make_unique<Tensor<char>>(dimensionSizes);
    tensor->setData({(char)0b00000000, (char)0b00110101, (char)0b11010010});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({(int)0b00101110, (int)0b00111111, -2/*(int)0b11111110*/});

    Tensor<int> result((char)0b00101110 | *tensor);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseOrAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setData({3, -3, -1, -4, -1, -16});

    *tensor |= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseOrAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = Tensor<Tensor<int>>(dimensionSizes);
    tensor.setData({Tensor<int>({1}, {0}), Tensor<int>({1}, {0x40140000})});

    auto tensor2 = Tensor<Tensor<int>>(dimensionSizes);
    tensor2.setData({Tensor<int>({1}, {3}), Tensor<int>({1}, {(int)0xC0200000})});

    auto expected = Tensor<Tensor<int>>(dimensionSizes);
    expected.setData({Tensor<int>({1}, {3}), Tensor<int>({1}, {(int)0xC0340000})});

    tensor |= tensor2;

    EXPECT_EQ(tensor, expected);
}

TEST(tensor_test, operatorBitwiseOrAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<uint8_t>>>(dimensionSizes);
    tensor->setData({Tensor<uint8_t>({1}, {0b10101100}), Tensor<uint8_t>({1}, {0b00001001})});

    auto expected = std::make_unique<Tensor<Tensor<uint8_t>>>(dimensionSizes);
    expected->setData({Tensor<uint8_t>({1}, {0b11111110}), Tensor<uint8_t>({1}, {0b01011011})});
    
    *tensor |= Tensor<uint8_t>({1}).setData({0b01010010});

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseOrAssignValue_002){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = Tensor<std::bitset<32>>(dimensionSizes);
    tensor.setData({0x00000000, 0xf0454ac0});

    auto expected = Tensor<std::bitset<32>>(dimensionSizes);
    expected.setData({0x0A1B80ED, 0xFA5FCAED});
    
    tensor |= 0x0a1b80ed;

    EXPECT_EQ(tensor, expected);
}

// (&) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseAnd_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({0, 0, -2, 4, -6, 0});

    Tensor<int> result(*tensor & *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseAnd_002){

    const std::vector<uint64_t> dimensionSizes{3, 1};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true, false, false});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setData({true, true, false});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({1, 0, 0});

    Tensor<int> result(*tensor & *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseAndValue_001){

    const std::vector<uint64_t> dimensionSizes{1, 3};

    auto tensor = std::make_unique<Tensor<uint64_t>>(dimensionSizes);
    tensor->setData({0, std::bit_cast<uint64_t>(4000000.211), std::bit_cast<uint64_t>(-1.)});

    auto expected = std::make_unique<Tensor<uint64_t>>(dimensionSizes);
    expected->setData({0, 0x140000000000000, 0xBFE0000000000000});

    Tensor<uint64_t> result(*tensor & std::bit_cast<uint64_t>(-0.5));

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseAndValue_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = std::make_unique<Tensor<char>>(dimensionSizes);
    tensor->setData({(char)0b00010001, (char)0b00110101, (char)0b11010010});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({(int)0b00000000, (int)0b00100100, (int)0b00000010});

    Tensor<int> result((char)0b00101110 & *tensor);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseAndAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<int64_t>(dimensionSizes);
    tensor.setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = Tensor<int64_t>(dimensionSizes);
    tensor2.setData({3, -8, -2, -100, -5, 0});

    auto expected = Tensor<int64_t>(dimensionSizes);
    expected.setData({0, 0, -2, 4, -6, 0});

    tensor &= tensor2;

    EXPECT_EQ(tensor, expected);
}

TEST(tensor_test, operatorBitwiseAndAssign_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({false, true, true});

    auto tensor2 = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor2->setData({true, true, 2});

    auto expected = std::make_unique<Tensor<bool>>(dimensionSizes);
    expected->setData({false, true, true});

    *tensor &= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseAndAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<uint8_t>>>(dimensionSizes);
    tensor->setData({Tensor<uint8_t>({1}).setData({0b10101100}), Tensor<uint8_t>({1}).setData({0b00001011})});

    auto expected = std::make_unique<Tensor<Tensor<uint8_t>>>(dimensionSizes);
    expected->setData({Tensor<uint8_t>({1}).setData({0b00000000}), Tensor<uint8_t>({1}).setData({0b00000010})});
    
    *tensor &= Tensor<uint8_t>({1}).setData({0b01010010});

    EXPECT_EQ(*tensor, *expected);
}

// (^) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseXor_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({3, -3, 1, -8, 5, -16});

    Tensor<int> result(*tensor ^ *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseXor_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    tensor->setData({0b1100, 0b1010});

    auto tensor2 = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    tensor2->setData({0b1010, 0b0011});

    auto expected = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    expected->setData({0b0110, 0b1001});

    Tensor<std::bitset<4>> result(*tensor ^ *tensor2);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseXorValue_001){

    const std::vector<uint64_t> dimensionSizes{1, 3};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true, false, true});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({false, true, false});

    Tensor<int> result(*tensor ^ true);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseXorValue_002){

    const std::vector<uint64_t> dimensionSizes{4};

    auto tensor = std::make_unique<Tensor<bool>>(dimensionSizes);
    tensor->setData({true, false, true, false});

    auto expected = std::make_unique<Tensor<int>>(dimensionSizes);
    expected->setData({false, true, false, true});

    Tensor<int> result(true ^ *tensor);

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseXorAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setData({3, -8, -2, -100, -5, 0});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setData({3, -3, 1, -8, 5, -16});

    *tensor ^= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseXorAssign_002){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setData({0, 0x40140000});

    auto tensor2 = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor2->setData({3, 0xC0200000});

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setData({3, 0x80340000});

    *tensor ^= *tensor2;

    EXPECT_EQ(*tensor, *expected);
}

TEST(tensor_test, operatorBitwiseXorAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<Tensor<uint8_t>>>(dimensionSizes);
    tensor->setData({Tensor<uint8_t>({1}, {0b10101100}), Tensor<uint8_t>({1}, {0b00001011})});

    auto expected = std::make_unique<Tensor<Tensor<uint8_t>>>(dimensionSizes);
    expected->setData({Tensor<uint8_t>({1}, {0b11111110}), Tensor<uint8_t>({1}, {0b01011001})});
    
    *tensor ^= Tensor<uint8_t>({1}).setData({0b01010010});

    EXPECT_EQ(*tensor, *expected);
}

// (<<) -----------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitshiftLeft_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({0, 5, -1, (int)0x80000000, 3, -16});

    auto tensor2 = Tensor<int>(dimensionSizes);
    tensor2.setData({3, 1, 3, 1, 1, 3});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0, 10, -8, 0, 6, -128});

    Tensor<int> result(tensor << tensor2);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorBitshiftLeftValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = Tensor<uint64_t>(dimensionSizes);
    tensor.setData({1, std::bit_cast<uint64_t>(4000000.211)});

    auto expected = Tensor<uint64_t>(dimensionSizes);
    expected.setData({0x00000010, 0x14E84801B020C4A0});

    Tensor<uint64_t> result(tensor << (uint64_t)0x00000004);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorBitshiftLeftValue_002){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = Tensor<short>(dimensionSizes);
    tensor.setData({1, 0});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0b10000, 0});

    Tensor<int> result(tensor << (short)0x00000004);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorBitshiftLeftAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = Tensor<int64_t>(dimensionSizes);
    tensor.setData({0, 0x40140000});

    auto tensor2 = Tensor<int64_t>(dimensionSizes);
    tensor2.setData({3, 0x00000020});

    auto expected = Tensor<int64_t>(dimensionSizes);
    expected.setData({0, 0x4014000000000000});

    tensor <<= tensor2;

    EXPECT_EQ(tensor, expected);
}

TEST(tensor_test, operatorBitshiftLeftAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<short>>(dimensionSizes);
    tensor->setData({0x7000, 69});

    auto expected = std::make_unique<Tensor<short>>(dimensionSizes);
    expected->setData({(short)0xC000, 276});
    
    *tensor <<= 0x0002;

    EXPECT_EQ(*tensor, *expected);
}

// (>>) -----------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitshiftRight_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({0, 5, -1, (int)0x80000000});

    auto tensor2 = Tensor<int>(dimensionSizes);
    tensor2.setData({3, 1, 3, 1});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0, 2, -1, (int)-0x40000000});

    Tensor<int> result(tensor >> tensor2);

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorBitshiftRightValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = Tensor<uint8_t>(dimensionSizes);
    tensor.setData({0b10000001, 0b01110010});

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({0b00100000, 0b00011100});

    Tensor<int> result = tensor >> (uint8_t)2;

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorBitshiftRightAssign_001){

    const std::vector<uint64_t> dimensionSizes{2, 1};

    auto tensor = Tensor<int64_t>(dimensionSizes);
    tensor.setData({0, 0x40140000});

    auto tensor2 = Tensor<int64_t>(dimensionSizes);
    tensor2.setData({3, 0x00000010});

    auto expected = Tensor<int64_t>(dimensionSizes);
    expected.setData({0, 0x4014});

    tensor >>= tensor2;

    EXPECT_EQ(tensor, expected);
}

TEST(tensor_test, operatorBitshiftRightAssignValue_001){

    const std::vector<uint64_t> dimensionSizes{2};

    auto tensor = std::make_unique<Tensor<short>>(dimensionSizes);
    tensor->setData({0x7000, 69});

    auto expected = std::make_unique<Tensor<short>>(dimensionSizes);
    expected->setData({(short)0x1C00, 0x11});
    
    *tensor >>= 0x0002;

    EXPECT_EQ(*tensor, *expected);
}

// (~) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorBitwiseNegation_001){

    const std::vector<uint64_t> dimensionSizes{2, 3};

    auto tensor = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    tensor->setData({0, 5, -1, 100, -2, -16});

    Tensor<int64_t> result(~*tensor);

    auto expected = std::make_unique<Tensor<int64_t>>(dimensionSizes);
    expected->setData({-1, -6, 0, -101, 1, 15});

    EXPECT_EQ(result, *expected);
}

TEST(tensor_test, operatorBitwiseNegation_002){

    const std::vector<uint64_t> dimensionSizes{4};

    auto tensor = Tensor<std::bitset<9>>(dimensionSizes);
    tensor.setData({0b100101110, 0b111111111, 0b000000000, 0b011001001});

    auto expected = Tensor<std::bitset<9>>(dimensionSizes);
    expected.setData({0b011010001, 0b000000000, 0b111111111, 0b100110110});

    Tensor<std::bitset<9>> result = ~tensor;

    EXPECT_EQ(result, expected);
}

// (!) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorLogicalNegation_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<bool>(dimensionSizes);
    tensor.setData({false, true, false, true});

    Tensor<bool> result = !tensor;

    auto expected = Tensor<bool>(dimensionSizes);
    expected.setData({true, false, true, false});

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorLogicalNegation_002){

    const std::vector<uint64_t> dimensionSizes{3};

    auto tensor = Tensor<Tensor<short>>(dimensionSizes);
    tensor.setData({Tensor<short>({2}, {152, 0}), Tensor<short>({2}, {-1, -10000}), Tensor<short>({1}, {0})});

    auto expected = Tensor<Tensor<bool>>(dimensionSizes);
    expected.setData({Tensor<bool>({2}, {0, 1}), Tensor<bool>({2}, {0, 0}), Tensor<bool>({1}, {1})});

    Tensor<Tensor<bool>> result = !tensor;

    EXPECT_EQ(result, expected);
}

// (+) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorUnaryPlus_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({1, 69, 0, -152});

    Tensor<int> result = +tensor;

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({1, 69, 0, -152});

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorUnaryPlus_002){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({1, 69, 0, -152});

    Tensor<int> result = +tensor;

    result.getItem({1, 0})++;

    EXPECT_NE(tensor, result);
}

TEST(tensor_test, operatorUnaryPlus_003){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({1, 69, 0, -152});

    Tensor<int> result = +tensor;
    Tensor<int> result2 = +tensor;

    result.getItem({1, 0})++;

    EXPECT_NE(result, result2);
}

// (-) ------------------------------------------------------------------------------------------------------------------------

TEST(tensor_test, operatorUnaryMinus_001){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<int>(dimensionSizes);
    tensor.setData({1, 69, 0, -152});

    Tensor<int> result = -tensor;

    auto expected = Tensor<int>(dimensionSizes);
    expected.setData({-1, -69, -0, 152});

    EXPECT_EQ(result, expected);
}

TEST(tensor_test, operatorUnaryMinus_002){

    const std::vector<uint64_t> dimensionSizes{2, 2};

    auto tensor = Tensor<float>(dimensionSizes);
    tensor.setData({1.5, 69.001, 0., -152.65202});

    Tensor<float> result = -tensor;

    auto expected = Tensor<float>(dimensionSizes);
    expected.setData({-1.5, -69.001, -0., 152.65202});

    EXPECT_EQ(result, expected);
}

// OPERATOR OVERLOAD TESTS END



TEST(tensor_test, showDebug){

    const std::vector<uint64_t> dimensionSizes{2, 3};
    auto tensor = std::make_unique<Tensor<int>>(dimensionSizes);

    auto expected = std::make_unique<Tensor<std::bitset<4>>>(dimensionSizes);
    expected->setData({0b0110, 0b1001, 0b0101, 0b1010, 0b1110, 0b0111});

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
