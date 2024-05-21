#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

#include <format>

#include "ITensor.hpp"

//Our main <3 -----------------------------------------------------------------------------------------------------------------
int main(){

    std::cout << "\nStart testing.\n";
    std::cout << "Testing Tensor.\n\n";

    int dimension = 2;
    std::vector<int> dimensionSizes = {2, 3};

    Tensor<double>* tensor1 = new Tensor<double>(dimension, dimensionSizes);

    tensor1->assign(5,     {0, 0});
    tensor1->assign(0.55,    {1, 0});
    tensor1->assign(1,     {0, 1});
    tensor1->assign(-2,      {1, 1});
    tensor1->assign(4.5,   {0, 2});
    tensor1->assign(7,    {1, 2});
    
    Tensor<double>* tensor2 = new Tensor<double>({2}, dimensionSizes);

    double tensor2Items[6] = {2, -5, 10, 5.89647, 3, 4};
    tensor2->setTensor(tensor2Items);

    tensor1->showTensor();
    tensor2->showTensor();


    std::cout << "Addition now.\n";

    Tensor<double>* tensorAdd = *tensor1 + *tensor2;
    tensorAdd->showTensor();


    std::cout << "\n\n";

    std::cout << "Transposition now.\n";
    Tensor<double>* tensorTransposed = tensor1->transposition();
    std::cout << "Tensor allocated.\n";
    tensorTransposed->showTensor();

    std::cout << "Is equilateral? / no\n";
    std::cout << tensorTransposed->isTensorEquilateral() << "\n";

    std::vector<int> tensorEquiDimensionSizes = {3, 3};
    Tensor<double>* tensorEqui = new Tensor<double>(dimension, tensorEquiDimensionSizes);
    std::cout << "Is equilateral? / yes\n";
    std::cout << tensorEqui->isTensorEquilateral() << "\n";

    assert(tensorEqui->isTensorEquilateral() == 1);
    assert(tensorTransposed->isTensorEquilateral() == 0);



    return 0;
}