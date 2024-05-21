#include <iostream>
#include <cmath>
#include <vector>

#include "complex.h"

using namespace std;

#define MAX_LOOP_COUNT 1024;

#ifndef TENSOR_HPP
#define TENSOR_HPP

/** ===========================================================================================================================
 * Class for work with general tensors
 * 
 * Note: Tensor is represented in class as one dimensional array with calculated number of items and 
 * methods getItem and getCoords that calculate the virtual coordinated of the tensor and vice versa
 */
template<class T> class Tensor{

    private:

    vector<T> tensor;               // The tensor itself, represented by one-dimensional vector containing all the elements

    int dimension;                  // Number of tensor dimensions
    vector<int> dimensionSizes;     // Size od each tensor dimension
    int itemCount;                  // Total number of items

    public:

    /** -----------------------------------------------------------------------------------------------------------------------
     * TensorClass() constructor - Writes into tensor attributes dimension and dimension sizes and with that calculates
     * number of items
     * 
     * @param newTensorDimension - number of tensor dimensions
     * @param newTensorDImensionSizes - a vector filled with sizes of dimensions
     * 
     * Note: It can also calculate advanced characteristics of tensor like if a tensor is equilateral
     * Note: to-do how to deal with message and maybe it should not always compute advanced characteristics
    */
    Tensor(const int newTensorDimension, const vector<int> newTensorDimensionSizes);

    /** -----------------------------------------------------------------------------------------------------------------------
     * getDimensionSizes() - Public method to get all dimension sizes
     * 
     * @return - vector containing one int per dimension with value of its size
    */
    vector<int> getDimensionSizes() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getNumberOfDimensions() - Public method to get the number of dimensions
     * 
     * @return - a number representing a number of dimensions
    */
    int getNumberOfDimensions() const;

    void setTensor(const T* tensorItems);

    /** -----------------------------------------------------------------------------------------------------------------------
     * isTensorEquilateral() - Public method to decide, if the tensor dimensions have the same sizes
     * 
     * @return - boolean true if the tensor is equilateral and false if not
    */
    bool isTensorEquilateral() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * assign() - Public method to assign value into tensor on the desired coordinates
     *  
     * @param value - a value of custom type that will be stored in the tensor
     * @param coordinates - a vector of coordinates that the value will be assigned to
    */
    void assign(T value, vector<int> coordinates);

    /** -----------------------------------------------------------------------------------------------------------------------
     * showTensor() - Public method to output the whole tensor into std::cout
     * 
     * Note: Is currently working only for 1d, 2d and maybe 3d tensors
    */
    void showTensor() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * showItem() - Public method to write into cout the item index based on the coordinates input
     * 
     * @param coordinates - address of coordinates in a tensor
     * 
     * Note: Tensor is actually stored as one-dimensional array so any item can be represented either in artificial coordinate system 
     * or by just index in the actual array
    */
    void showItem(vector<int> coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * showCoords() - Public method to write into cout tensor coordinates based from item index input
     * 
     * @param itemNumber - item index in a tensor
     * 
     * Note: Tensor is actually stored as one-dimensional array so any item can be represented either in artificial coordinate system 
     * or by just index in the actual array
    */
    void showCoords(const int itemNumber) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * transposition() - Public method to swap two dimensions in a tensor
     * 
     * @param dim1 - first dimension to swap, default value is 0
     * @param dim2 - second dimension to swap, default value is 1
     * 
     * @return - a pointer to new allocated tensor, that has got transposed two dimensions
     * 
     * Example: TensorClass* tensorTransposed = tensor1->transposition();
    */
    Tensor<T>* transposition(const int dim1 = 0, const int dim2 = 1);

    /** -----------------------------------------------------------------------------------------------------------------------
     * TensorClass* operator+() - Publicly overload to add two tensors of the same size. Both by reference.
     * 
     * @param tensor2 - a second tensor to be added as reference (the same as the first)
     * 
     * @return - a pointer to new allocated tensor that is the sum of the both
     * 
     * Example: TensorClass* tensorAdd = *tensor1 + *tensor2;
    */
    Tensor<T>* operator+(const Tensor<T>& tensor2) const;

    // Our destructor -------------------------------------------------------------------------------------------------------
    ~Tensor();

    private:

    /** -----------------------------------------------------------------------------------------------------------------------
     * getCoords() - Private method to get coordinates from itemNumber in tensor, this is inverse method of "getItem()" method
     * 
     * @param itemNumber - it is the index of item that is stored in the tensor
     * 
     * @return - coordinates of the item in the tensor
     * 
     * Note: Tensor is actually stored as one-dimensional array so any item can be represented either in artificial coordinate system 
     * or by just index in the actual array
    */
    vector<int> getCoords(int itemNumber) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getItem() - Private method to get items index number in tensor - this is inverse method of "getCoords()"
     * 
     * @param coordinates - an array of coordinates of one item in tensor
     * 
     * @return - index of one number in tensor when represented in one dimension
     * 
     * Note: Tensor is actually stored as one-dimensional array so any item can be represented either in artificial coordinate system 
     * or by just index in the actual array
    */
    int getItem(const vector<int>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * constructorMessage() - Private method to output message to console about the object creation
     * 
     * @param itemCount - number of item in a tensor to be output
     * @param dimension - number of dimensions to be output
     * @param dimensionSizes - the dimension sizes to be output
    */
    void constructorMessage(int itemCount, int dimension, vector<int>& dimensionSizes) const;
};

#endif