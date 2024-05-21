#include <iostream>
#include <cmath>
#include <vector>

#include "ITensor.hpp"

using namespace std;

#define MAX_LOOP_COUNT 1024;
#define TENSOR_TYPE double

    template class Tensor<int>;
    template class Tensor<long long int>;
    template class Tensor<float>;
    template class Tensor<double>;

    template <class T>
    Tensor<T>::Tensor(const int newTensorDimension, const vector<int> newTensorDimensionSizes) 
    : dimension(newTensorDimension), dimensionSizes(newTensorDimensionSizes){

        // Calculate number of items
        int itemCounting = 1;
        for(int i = 0; i < newTensorDimension; i++){
            itemCounting *= newTensorDimensionSizes[i];
        }

        itemCount = itemCounting;

        // Allocate space for the tensor
        tensor.reserve(itemCount);

        constructorMessage(itemCount, dimension, dimensionSizes);

        cout << endl;
        cout << endl;
    }

    template <class T>
    vector<int> Tensor<T>::getDimensionSizes() const{
        return dimensionSizes;
    }

    template <class T>
    int Tensor<T>::getNumberOfDimensions() const{
        return dimension;
    }

    template <class T>
    void Tensor<T>::setTensor(const T* tensorItems){
        
        for(int i = 0; i < itemCount; i++){
            tensor[i] = tensorItems[i];
        }
    }

    template <class T>
    bool Tensor<T>::isTensorEquilateral() const{

        bool isEquilateral = false;

        for(int i = 0; i < dimension - 1; i++){

            isEquilateral = true;
            if(dimensionSizes[i] != dimensionSizes[i + 1]){
                isEquilateral = false;
                break;
            }
        }

        return isEquilateral;
    }

    template <class T>
    void Tensor<T>::assign(T value, vector<int> coordinates){

        TENSOR_TYPE itemNumber = getItem(coordinates);
        
        tensor[itemNumber] = value;
    }

    template <class T>
    void Tensor<T>::showTensor() const{

        cout << "Tensor is as follows:" << "\n\n";

        for(int i = 0; i < itemCount; i++){

            if((i % dimensionSizes[0] == 0) && 
                i > 0){
                cout << "\n";
                if(i % dimensionSizes[1] == 0){
                    cout << "\n";
                }
            }

            cout << "[" << tensor[i] << "] ";
        }

        cout << "\n\n";
    }

    template <class T>
    void Tensor<T>::showItem(vector<int> coordinates) const{
        
        int itemNumber = getItem(coordinates);
        cout << "Item: " << tensor[itemNumber] << endl;
    }

    template <class T>
    void Tensor<T>::showCoords(const int itemNumber) const{

        vector<int> coords = getCoords(itemNumber);

        cout << "Coords: ";
        for(int i = 0; i < dimension; i++){
            cout << coords[i] << " ";
        }
        cout << endl;
    }

    template <class T>
    Tensor<T>* Tensor<T>::transposition(const int dim1, const int dim2){

        // Copying the dimensionSizes
        vector<int> transposedDimensionSizes = dimensionSizes; 

        // Swapping the dimension sizes
        transposedDimensionSizes[dim1] = dimensionSizes[dim2]; 
        transposedDimensionSizes[dim2] = dimensionSizes[dim1];
        
        // Initializing the new tensor
        Tensor* tensorTransposed = new Tensor(dimension, transposedDimensionSizes);

        vector<int> temp, switched;
        temp.reserve(dimension);
        switched.reserve(dimension);

        // Looping thru elements in tensor and swapping the desired coordinates
        for(int i = 0; i < itemCount; i++){
            
            // Switching the two coordinated corresponding to the two dimensions we want to switch
            vector<int> temp(getCoords(i));

            // Deep copy of coords before swap
            for(const int& value : temp) switched.push_back(value);
            
            // The swap of two desired coordinates
            switched[dim1] = temp[dim2];
            switched[dim2] = temp[dim1];

            //cout << " tensor[" << i << "]: " << tensor[i] << " temp: " << temp[0] << " " << temp[1] 
            //<< " switched: " << switched[0] << " " << switched[1] << " | getItem(temp): " << getItem(temp) << " | tr.getItem(switched): " << tensorTransposed->getItem(switched) << endl;

            // Works until now, check the getItem function if it actually works properly
            tensorTransposed->tensor[tensorTransposed->getItem(switched)] = tensor[getItem(temp)];
        }

        return tensorTransposed;
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator+(const Tensor<T>& tensor2) const{

        //Allocation of new tensor. Since tensor addition doesnt change the size, we can get right to allocation
        Tensor* tensorOut = new Tensor(dimension, dimensionSizes);
        tensorOut->tensor.reserve(itemCount);

        for(int i = 0; i < itemCount; i++){
            tensorOut->tensor[i] = tensor[i] + tensor2.tensor[i];
        }

        return tensorOut;
    }

    template <class T>
    Tensor<T>::~Tensor(){
            //
    }

    template <class T>
    vector<int> Tensor<T>::getCoords(int itemNumber) const{

        vector<int> coordinates;
        coordinates.reserve(dimension);

        int dimensionProduct = 1;

        for(int i = 0; i < dimension; i++){
            for(int j = 0; j < (dimension - i - 1); j++){
                dimensionProduct *= dimensionSizes[j];
            }

            coordinates[dimension - i - 1] = itemNumber / dimensionProduct;
            itemNumber -= coordinates[dimension - i - 1] * dimensionProduct;
            dimensionProduct = 1;
        }

        return coordinates;
    }

    template <class T>
    int Tensor<T>::getItem(const vector<int>& coordinates) const{
        
        int itemNumber = 0;

       for(int i = 0; i < dimension; i++){

            if(i == 0){
                itemNumber += coordinates[i];
                continue;
            }
            
            int dimensionProduct = 1;

            for(int j = 0; j < i; j++){
                dimensionProduct *= dimensionSizes[j];
            }

            itemNumber += coordinates[i] * dimensionProduct;
            
       }

        return itemNumber;
    }

    template <class T>
    void Tensor<T>::constructorMessage(int itemCount, int dimension, vector<int>& dimensionSizes) const{

        // Message
        cout << "A tensor of " << itemCount << " items and " << dimension << " dimensions been allocated." << endl;
        cout << "A tensor dimensions are as follows: ";
        for(int i = 0; i < dimension; i++){
            cout << dimensionSizes[i] << " ";
        }
    }
