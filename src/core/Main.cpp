#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <memory>

#include <format>

#include "ITensor.hpp"

using std::vector, std::make_unique, std::cout, std::endl;

//Our main <3 -----------------------------------------------------------------------------------------------------------------
int main(){

    cout << "\nStart testing.\n";
    cout << "Testing Tensor.\n\n";

    const vector<int> dimensionSizes = {2, 3};

    cout << "Allocation:\n\n";

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
    assert(tensorTransposed->isTensorEquilateral() == 0);



    return 0;
}