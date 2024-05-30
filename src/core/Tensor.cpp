#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <stdint.h>

#include "ITensor.hpp"

#define MAX_LOOP_COUNT 1024;
#define __uint64 uint64_t

    template class Tensor<bool>;
    template class Tensor<char>;
    template class Tensor<short>;
    template class Tensor<int>;
    template class Tensor<long long int>;
    template class Tensor<float>;
    template class Tensor<double>;

    template <class T>
    Tensor<T>::Tensor(const std::vector<int>& newTensorDimensionSizes) 
    : dimensionSizes(newTensorDimensionSizes) {

        // Calculate number of items
        int itemCounting = 1;
        for(const auto& newTensorDimensionSize : newTensorDimensionSizes){
            itemCounting *= newTensorDimensionSize;
        }

        // Allocate space for the tensor
        tensor.resize(itemCounting);

        constructorMessage(dimensionSizes);
    }

    template <class T>
    const std::vector<int>* Tensor<T>::getDimensionSizes() const{
        return &dimensionSizes;
    }

    template <class T>
    int Tensor<T>::getNumberOfDimensions() const{
        return dimensionSizes.size();
    }

    template <class T>
    void Tensor<T>::setItems(const std::vector<T>& tensorItems){

        int copyLength = fmin(tensorItems.size(), tensor.size());

        for(int i = 0; i < copyLength; i++){
            tensor[i] = tensorItems[i];
        }
    }

    template <class T>
    void Tensor<T>::setItems(const T* tensorItems){

        if(tensorItems == nullptr || tensorItems == 0){
            fillWith(0);
            return;
        }
        
        for(__uint64 i = 0; i < tensor.size(); i++){
            tensor[i] = tensorItems[i];
        }
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill){

        // Since the type parameter is allowed to be bool, this cannot be a ranged based for loop
        for(__uint64 i = 0; i < tensor.size(); i++){
            tensor[i] = fill;
        }
    }

    template <class T>
    bool Tensor<T>::isTensorEquilateral() const{
        return std::adjacent_find(dimensionSizes.begin(), dimensionSizes.end(), std::not_equal_to<int>()) == dimensionSizes.end();
    }

    template <class T>
    void Tensor<T>::assign(const T& value, const std::vector<int>& coordinates){

        T itemNumber = getIndex(coordinates);
        
        tensor[itemNumber] = value;
    }

    template <class T>
    void Tensor<T>::showTensor() const{

        std::cout << "Tensor is as follows:\n\n";

        for(__uint64 i = 0; i < tensor.size(); i++){

            if((i % dimensionSizes[0] == 0) && 
                i > 0){
                std::cout << '\n';
                if(i % dimensionSizes[1] == 0){
                    std::cout << '\n';
                }
            }

            std::cout << "[" << tensor[i] << "] ";
        }

        std::cout << "\n\n";
    }

    template <class T>
    void Tensor<T>::showItem(const std::vector<int>& coordinates) const{
        
        int itemNumber = getIndex(coordinates);
        std::cout << "Item: " << tensor[itemNumber] << '\n';
    }

    template <class T>
    void Tensor<T>::showCoords(const int itemNumber) const{

        std::vector<int> coords = getCoords(itemNumber);

        std::cout << "Coords: ";
        for(const auto& coord : coords){
            std::cout << coord << " ";
        }
        std::cout << '\n';
    }

    template <class T>
    Tensor<T>* Tensor<T>::transposition(const int dim1, const int dim2) const{

        // Copying the dimensionSizes
        std::vector<int> transposedDimensionSizes = dimensionSizes; 

        // Swapping the dimension sizes
        transposedDimensionSizes[dim1] = dimensionSizes[dim2]; 
        transposedDimensionSizes[dim2] = dimensionSizes[dim1];
        
        // Initializing the new tensor
        Tensor* tensorTransposed = new Tensor(transposedDimensionSizes);

        std::vector<int> temp, switched;
        temp.reserve(dimensionSizes.size());
        switched.reserve(dimensionSizes.size());

        // Looping thru elements in tensor and swapping the desired coordinates
        for(__uint64 i = 0; i < tensor.size(); i++){
            
            // Switching the two coordinated corresponding to the two dimensions we want to switch
            std::vector<int> temp(getCoords(i));

            // Deep copy of coords before swap
            for(const int& value : temp) switched.push_back(value);
            
            // The swap of two desired coordinates
            switched[dim1] = temp[dim2];
            switched[dim2] = temp[dim1];

            // Works until now, check the getIndex function if it actually works properly
            tensorTransposed->tensor[tensorTransposed->getIndex(switched)] = tensor[getIndex(temp)];
        }

        return tensorTransposed;
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator+(const Tensor<T>& tensor2) const{

        //Allocation of new tensor. Since tensor addition doesnt change the size, we can get right to allocation
        Tensor* tensorOut = new Tensor(dimensionSizes);
        tensorOut->tensor.reserve(tensor.size());

        for(__uint64 i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = tensor[i] + tensor2.tensor[i];
        }

        return tensorOut;
    }

    template <class T>
    Tensor<T>::~Tensor(){
            //
    }

    template <class T>
    std::vector<int> Tensor<T>::getCoords(int itemNumber) const{

        std::vector<int> coordinates;
        __uint64 dimension = dimensionSizes.size();
        coordinates.reserve(dimension);

        int dimensionProduct = 1;

        for(__uint64 i = 0; i < dimension; i++){
            for(__uint64 j = 0; j < (dimension - i - 1); j++){
                dimensionProduct *= dimensionSizes[j];
            }

            coordinates[dimension - i - 1] = itemNumber / dimensionProduct;
            itemNumber -= coordinates[dimension - i - 1] * dimensionProduct;
            dimensionProduct = 1;
        }

        return coordinates;
    }

    template <class T>
    int Tensor<T>::getIndex(const std::vector<int>& coordinates) const{
        
        int itemNumber = 0;

       for(__uint64 i = 0; i < dimensionSizes.size(); i++){

            if(i == 0){
                itemNumber += coordinates[i];
                continue;
            }
            
            int dimensionProduct = 1;

            for(__uint64 j = 0; j < i; j++){
                dimensionProduct *= dimensionSizes[j];
            }

            itemNumber += coordinates[i] * dimensionProduct;
       }

        return itemNumber;
    }

    template <class T>
    void Tensor<T>::constructorMessage(const std::vector<int>& dimensionSizes) const{

        // Message
        std::cout << "A tensor of " << tensor.size() << " items and " << dimensionSizes.size() << " dimensions been allocated.\n";
        std::cout << "A tensor dimensions are as follows: ";
        for(const auto& dimensionSize : dimensionSizes){
            std::cout << dimensionSize << " ";
        }

        std::cout << "\n\n";
    }
