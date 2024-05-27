#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <memory>

#include <format>

#include "ITensor.hpp"

using std::vector, std::unique_ptr, std::cout, std::endl;

//Our main <3 -----------------------------------------------------------------------------------------------------------------
int main(){

    cout << "\nStart testing.\n";
    cout << "Testing Tensor.\n\n";

    vector<int> dimensionSizes = {2, 3};

    //Tensor<double>* tensor1 = new Tensor<double>(dimensionSizes);
    unique_ptr<Tensor<double>> tensor1(new Tensor<double>(dimensionSizes));

    tensor1->assign(5,     {0, 0});
    tensor1->assign(0.55,    {1, 0});
    tensor1->assign(1,     {0, 1});
    tensor1->assign(-2,      {1, 1});
    tensor1->assign(4.5,   {0, 2});
    tensor1->assign(7,    {1, 2});
    
    unique_ptr<Tensor<double>> tensor2(new Tensor<double>(dimensionSizes));

    double tensor2Items[6] = {2, -5, 10, 5.89647, 3, 4};
    tensor2->setItems(tensor2Items);

    tensor1->showTensor();
    tensor2->showTensor();


    cout << "Addition now.\n";

    unique_ptr<Tensor<double>> tensorAdd(*tensor1 + *tensor2);
    tensorAdd->showTensor();


    cout << "\n\n";

    cout << "Transposition now.\n";
    unique_ptr<Tensor<double>> tensorTransposed(tensor1->transposition());
    cout << "Tensor allocated.\n";
    tensorTransposed->showTensor();

    cout << "Is equilateral? / no\n";
    cout << tensorTransposed->isTensorEquilateral() << "\n";

    vector<int> tensorEquiDimensionSizes = {3, 3};
    unique_ptr<Tensor<double>> tensorEqui(new Tensor<double>(tensorEquiDimensionSizes));
    cout << "Is equilateral? / yes\n";
    cout << tensorEqui->isTensorEquilateral() << "\n";

    assert(tensorEqui->isTensorEquilateral() == 1);
    assert(tensorTransposed->isTensorEquilateral() == 0);



    return 0;
}