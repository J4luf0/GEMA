#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
//#include <typeinfo>

#include "ITensor.hpp"

#define MAX_LOOP_COUNT 1024;
#define uint64t uint64_t

    // Primitives and simple types
    template class Tensor<char>;
    template class Tensor<short>;
    template class Tensor<int>;
    template class Tensor<long long int>;
    template class Tensor<float>;
    template class Tensor<double>;
    template class Tensor<std::string>;
    // The tensor itself and basically any object, O meaning Object
    template <typename T> class Tensor<Tensor<T>*>;
    template <typename O> class Tensor<O&>;
    template <typename O> class Tensor<O*>;
    // Others
    template <typename O> class Tensor<std::vector<O>>;
    template <typename O> class Tensor<std::unique_ptr<O>>;

    // public

    template <class T>
    Tensor<T>::Tensor(const std::vector<int>& newTensorDimensionSizes) 
    : dimensionSizes(newTensorDimensionSizes) {

        int itemCounting = getNumberOfItems(newTensorDimensionSizes);

        tensor.resize(itemCounting);

        constructorMessage(dimensionSizes);
    }

    template <class T>
    const std::vector<int>* Tensor<T>::getDimensionSizes() const{
        return &dimensionSizes;
    }

    template <class T>
    uint64t Tensor<T>::getNumberOfDimensions() const{
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
    void Tensor<T>::setTensorOutput(const std::function<void(const T&)> tensorOutput){
        this->tensorOutput = tensorOutput;
    }

    template <class T>
    bool Tensor<T>::isTensorEquilateral() const{
        return std::adjacent_find(dimensionSizes.begin(), dimensionSizes.end(), std::not_equal_to<int>()) == dimensionSizes.end();
    }

    template <class T>
    void Tensor<T>::assign(const T& value, const std::vector<int>& coordinates){

        int itemNumber = getIndex(coordinates);
        
        tensor[itemNumber] = value;
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill){

        for(T& item : tensor){
            item = fill;
        }
    }

    template <class T>
    void Tensor<T>::showTensor() const{

        std::cout << "Tensor is as follows:\n\n";

        for(uint64t i = 0; i < tensor.size(); i++){

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
        Tensor<T>* tensorTransposed = new Tensor<T>(transposedDimensionSizes);

        std::vector<int> original, switched;
        original.resize(dimensionSizes.size());
        switched.resize(dimensionSizes.size());

        // Looping thru elements in tensor and swapping the desired coordinates
        for(uint64t i = 0; i < tensor.size(); i++){
            
            // Switching the two coordinated corresponding to the two dimensions we want to switch
            original = getCoords(i);

            // The swap of two desired coordinates
            switched = original;
            switched[dim1] = original[dim2];
            switched[dim2] = original[dim1];

            // Works until now, check the getIndex function if it actually works properly
            tensorTransposed->tensor[tensorTransposed->getIndex(switched)] = tensor[getIndex(original)];
        }

        return tensorTransposed;
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator+(const Tensor<T>& tensor2) const{

        Tensor* tensorOut = new Tensor(dimensionSizes);
        tensorOut->tensor.resize(tensor.size());

        for(uint64t i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = tensor[i] + tensor2.tensor[i];
        }

        return tensorOut;
    }

    template <class T>
    void Tensor<T>::operator+=(const Tensor<T>& tensor2){
        
        for(uint64t i = 0; i < tensor.size(); i++){
            tensor[i] += tensor2.tensor[i];
        }
    }

    template <class T>
    constexpr Tensor<T>& Tensor<T>::operator=(const Tensor<T>& assigner) const{
        return *copy();
    }

    template <class T>
    constexpr Tensor<T>* Tensor<T>::operator=(const Tensor<T>* assigner) const{
        return copy();
    }

    template <class T>
    void Tensor<T>::deleteTensor(){
        
        for(T& item : tensor){
            delete item;
        }
        
    }

    template <class T>
    Tensor<T>::~Tensor(){
        //
    }

    // private

    template <class T>
    std::vector<int> Tensor<T>::getCoords(int itemNumber) const{

        std::vector<int> coordinates;
        uint64t dimension = dimensionSizes.size();
        coordinates.reserve(dimension);

        int dimensionProduct = 1;

        for(uint64t i = 0; i < dimension; i++){
            for(uint64t j = 0; j < (dimension - i - 1); j++){
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

       for(uint64t i = 0; i < dimensionSizes.size(); i++){

            if(i == 0){
                itemNumber += coordinates[i];
                continue;
            }
            
            int dimensionProduct = 1;

            for(uint64t j = 0; j < i; j++){
                dimensionProduct *= dimensionSizes[j];
            }

            itemNumber += coordinates[i] * dimensionProduct;
       }

        return itemNumber;
    }

    template <class T>
    int Tensor<T>::getNumberOfItems(const std::vector<int>& dimensionSizes) const{

        int itemCounting = 1;
        for(const auto& dimensionSize : dimensionSizes){
            itemCounting *= dimensionSize;
        }
        return itemCounting;
    }

    template <class T>
    inline constexpr Tensor<T>* Tensor<T>::copy() const{

        Tensor<T>* newTensor = new Tensor<T>(dimensionSizes);

        for(uint64t i = 0; i < tensor.size(); i++){
            newTensor->tensor[i] = tensor[i];
        }

        return newTensor;
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