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

    vector<int> dimensionSizes = {2, 3};

    //Tensor<double>* tensor1 = new Tensor<double>(dimensionSizes);
    //unique_ptr<Tensor<double>> tensor1(new Tensor<double>(dimensionSizes));
    auto tensor1 = make_unique<Tensor<double>>(dimensionSizes);

    tensor1->assign(5,     {0, 0});
    tensor1->assign(0.55,    {1, 0});
    tensor1->assign(1,     {0, 1});
    tensor1->assign(-2,      {1, 1});
    tensor1->assign(4.5,   {0, 2});
    tensor1->assign(7,    {1, 2});
    
    auto tensor2 = make_unique<Tensor<double>>(dimensionSizes);

    double tensor2Items[6] = {2, -5, 10, 5.89647, 3, 4};
    tensor2->setItems(tensor2Items);

    tensor1->showTensor();
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