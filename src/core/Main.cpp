#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

#include <format>

#include "ITensor.hpp"

using namespace std;

//Our main <3 -----------------------------------------------------------------------------------------------------------------
int main(){

    cout << "\nStart testing." << endl;
    cout << "Testing Tensor." << endl;
    cout << endl;

    int dimension = 2;
    vector<int> dimensionSizes = {2, 3};

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


    cout << "Addition now." << endl;

    Tensor<double>* tensorAdd = *tensor1 + *tensor2;
    tensorAdd->showTensor();


    cout << endl << endl;

    cout << "Transposition now." << endl;
    Tensor<double>* tensorTransposed = tensor1->transposition();
    cout << "Tensor allocated." << endl;
    tensorTransposed->showTensor();

    cout << "Is equilateral? / no" << endl;
    cout << tensorTransposed->isTensorEquilateral() << endl;

    vector<int> tensorEquiDimensionSizes = {3, 3};
    Tensor<double>* tensorEqui = new Tensor<double>(dimension, tensorEquiDimensionSizes);
    cout << "Is equilateral? / yes" << endl;
    cout << tensorEqui->isTensorEquilateral() << endl;


    // Testing - will be moved to separate file shortly
    assert(tensorEqui->isTensorEquilateral() == 1);
    assert(tensorTransposed->isTensorEquilateral() == 0);



    return 0;
}