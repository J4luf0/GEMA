#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <bit>
#include <format>

#include "ITensor.hpp"

#define MAX_LOOP_COUNT 1024;
#define uint64t uint64_t
#define uint32t uint32_t

    // Primitives and simple types
    //template class Tensor<bool>; //TODO: specialize toString()
    template class Tensor<char>;
    template class Tensor<short>;
    template class Tensor<int>;
    template class Tensor<long long int>;
    template class Tensor<float>;
    template class Tensor<double>;
    // The tensor itself and basically any object, O meaning Object
    template <typename T> class Tensor<Tensor<T>*>;
    template <typename O> class Tensor<O&>;
    template <typename O> class Tensor<O*>;
    // Others
    template <typename O> class Tensor<std::vector<O>>;
    template <typename O> class Tensor<std::unique_ptr<O>>;



    // public methods

    template <class T>
    Tensor<T>::Tensor(const std::vector<int>& newTensorDimensionSizes) noexcept
    : dimensionSizes(newTensorDimensionSizes) {

        int itemCounting = getNumberOfItems(newTensorDimensionSizes);

        tensor.resize(itemCounting);

        //constructorMessage(dimensionSizes);
    }

    template <class T>
    Tensor<T>::Tensor(const Tensor<T>& otherTensor) noexcept{
        *this = otherTensor;
    }

    template <class T>
    Tensor<T>::Tensor() noexcept{
        
    }

    template <class T>
    const std::vector<int>& Tensor<T>::getDimensionSizes() const noexcept{
        return dimensionSizes;
    }

    template <class T>
    uint64t Tensor<T>::getNumberOfDimensions() const noexcept{
        return dimensionSizes.size();
    }

    template <class T>
    T Tensor<T>::getItem(const std::vector<int>& coordinates) const noexcept{

        return tensor[getIndex(coordinates)];
    }

    template <class T>
    inline void* Tensor<T>::getPointer(const std::vector<int>& coordinates) const noexcept{
        
        return (void*)&tensor[getIndex(coordinates)];
    }

    template <>
    inline void* Tensor<bool>::getPointer(const std::vector<int>& coordinates) const noexcept{
        
        return (void*)&tensor;
    }
    
    template <class T>
    void Tensor<T>::setItem(const T& value, const std::vector<int>& coordinates) noexcept{

        int itemNumber = getIndex(coordinates);
        tensor[itemNumber] = value;
    }

    template <class T>
    void Tensor<T>::setItems(const std::vector<T>& tensorItems) noexcept{

        //int copyLength = fmin(tensorItems.size(), tensor.size()); //readd in safety wrapper

        for(uint64t i = 0; i < tensor.size(); i++){
            tensor[i] = tensorItems[i];
        }
    }

    template <class T>
    void Tensor<T>::setTensorOutput(const std::function<void(const T&)> tensorOutput) noexcept{
        this->tensorOutput = tensorOutput;
    }

    template <class T>
    void Tensor<T>::setItemOutput(const std::function<void(const T&)> itemOutput) noexcept{
        this->itemOutput = itemOutput;
    }

    template <class T>
    bool Tensor<T>::isTensorEquilateral() const noexcept{
        return std::adjacent_find(dimensionSizes.begin(), dimensionSizes.end(), std::not_equal_to<int>()) == dimensionSizes.end();
    }

    template <class T>
    std::string Tensor<T>::toString() const{

        std::string output = "";

        for(uint64t i = 0; i < tensor.size(); i++){

            const std::vector<int> itemCoords = getCoords(i);
            
            std::cout << "item: " << tensor[i] << " item coords: " << std::endl;
            for(uint64t j = 0; j < itemCoords.size(); j++){
                std::cout << itemCoords[j] << ", ";
            }
            std::cout << std::endl;

            std::string startBrackets = "";
            std::string endBrackets = "";

            // Get number of opening/closing brackets by looping through item coordinates and detecting presence of lowest/highest coordinate
            for(uint64t j = 0; j < dimensionSizes.size(); j++){

                if(itemCoords[j] <= 0){
                    startBrackets += '{';
                }

                if(itemCoords[j] >= dimensionSizes[j] - 1){
                    endBrackets += '}';
                }
            }

            output += std::format("{}{}, {}", startBrackets, tensor[i], endBrackets);
        }

        return output;
    }

    template <class T>
    inline constexpr Tensor<T>* Tensor<T>::copy() const noexcept{

        Tensor<T>* newTensor = new Tensor<T>(dimensionSizes);

        for(uint64t i = 0; i < tensor.size(); i++){
            newTensor->tensor[i] = *(new T(tensor[i]));
        }

        return newTensor;
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill) noexcept{

        for(T& item : tensor){
            item = fill;
        }
    }

    template <>
    void Tensor<bool>::fillWith(const bool& fill) noexcept{

        for(uint64t i = 0; i < tensor.size(); i++){
            tensor.at(i) = fill;
        }
    }

    template <class T>
    Tensor<T>* Tensor<T>::transposition(const int dim1, const int dim2) const noexcept{

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
    bool Tensor<T>::operator==(const Tensor<T>& tensor2) const noexcept{

        if(this->tensor.size() != tensor2.tensor.size()){
            return false;
        }

        // Check if all items are equal
        return std::equal(this->tensor.begin(), this->tensor.end(), tensor2.tensor.begin(), [&](const auto& a, const auto& b){

            return compareItems(a, b);
        });

        // This one line does the same thing and gets rid of branching, yet is unreadable and the performance difference may be negligible
        //return !(this->tensor.size() - tensor2.tensor.size()) && std::equal(this->tensor.begin(), this->tensor.end(), tensor2.tensor.begin());
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator+(const Tensor<T>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem + tensor2Item;
        });
    }

    template <class T>
    void Tensor<T>::operator+=(const Tensor<T>& tensor2) noexcept{

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem += tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator-(const Tensor<T>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem - tensor2Item;
        });
    }

    template <class T>
    void Tensor<T>::operator-=(const Tensor<T>& tensor2) noexcept{
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem -= tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator|(const Tensor<T>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){

            return tensorItem | tensor2Item;
        });
    }

    template <>
    Tensor<double>* Tensor<double>::operator|(const Tensor<double>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const double& tensorItem, const double& tensor2Item){
            return std::bit_cast<uint64t>(tensorItem) | std::bit_cast<uint64t>(tensor2Item);
        });
    }

    template <>
    Tensor<float>* Tensor<float>::operator|(const Tensor<float>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const float& tensorItem, const float& tensor2Item){
            return std::bit_cast<uint32t>(tensorItem) | std::bit_cast<uint32t>(tensor2Item);
        });
    }

    template <class T>
    void Tensor<T>::operator|=(const Tensor<T>& tensor2) noexcept{
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem |= tensor2Item;
        });
    }

    template <>
    void Tensor<double>::operator|=(const Tensor<double>& tensor2) noexcept{
        
        apply(tensor2, [](double& tensorItem, const double& tensor2Item){
            auto newItem = std::bit_cast<uint64t>(tensorItem);
            tensorItem = newItem | std::bit_cast<uint64t>(tensor2Item);
            return tensorItem;
        });
    }

    template <>
    void Tensor<float>::operator|=(const Tensor<float>& tensor2) noexcept{
        
        apply(tensor2, [](float& tensorItem, const float& tensor2Item){
            auto newItem = std::bit_cast<uint32t>(tensorItem);
            tensorItem = newItem | std::bit_cast<uint32t>(tensor2Item);
            return tensorItem;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator&(const Tensor<T>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem & tensor2Item;
        });
    }

    template <>
    Tensor<double>* Tensor<double>::operator&(const Tensor<double>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const double& tensorItem, const double& tensor2Item){
            return std::bit_cast<uint64t>(tensorItem) & std::bit_cast<uint64t>(tensor2Item);
        });
    }

    template <>
    Tensor<float>* Tensor<float>::operator&(const Tensor<float>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const float& tensorItem, const float& tensor2Item){
            return std::bit_cast<uint32t>(tensorItem) & std::bit_cast<uint32t>(tensor2Item);
        });
    }

    template <class T>
    void Tensor<T>::operator&=(const Tensor<T>& tensor2) noexcept{
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem &= tensor2Item;
        });
    }

    template <>
    void Tensor<double>::operator&=(const Tensor<double>& tensor2) noexcept{
        
        apply(tensor2, [](double& tensorItem, const double& tensor2Item){
            auto newItem = std::bit_cast<uint64t>(tensorItem);
            tensorItem = newItem & std::bit_cast<uint64t>(tensor2Item);
            return tensorItem;
        });
    }

    template <>
    void Tensor<float>::operator&=(const Tensor<float>& tensor2) noexcept{
        
        apply(tensor2, [](float& tensorItem, const float& tensor2Item){
            auto newItem = std::bit_cast<uint32t>(tensorItem);
            tensorItem = newItem & std::bit_cast<uint32t>(tensor2Item);
            return tensorItem;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator^(const Tensor<T>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem ^ tensor2Item;
        });
    }

    template <>
    Tensor<double>* Tensor<double>::operator^(const Tensor<double>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const double& tensorItem, const double& tensor2Item){
            return std::bit_cast<uint64t>(tensorItem) ^ std::bit_cast<uint64t>(tensor2Item);
        });
    }

    template <>
    Tensor<float>* Tensor<float>::operator^(const Tensor<float>& tensor2) const noexcept{

        return applyAndReturn(tensor2, [](const float& tensorItem, const float& tensor2Item){
            return std::bit_cast<uint32t>(tensorItem) ^ std::bit_cast<uint32t>(tensor2Item);
        });
    }

    template <class T>
    void Tensor<T>::operator^=(const Tensor<T>& tensor2) noexcept{
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem ^= tensor2Item;
        });
    }

    template <>
    void Tensor<double>::operator^=(const Tensor<double>& tensor2) noexcept{
        
        apply(tensor2, [](double& tensorItem, const double& tensor2Item){
            auto newItem = std::bit_cast<uint64t>(tensorItem);
            tensorItem = newItem ^ std::bit_cast<uint64t>(tensor2Item);
            return tensorItem;
        });
    }

    template <>
    void Tensor<float>::operator^=(const Tensor<float>& tensor2) noexcept{
        
        apply(tensor2, [](float& tensorItem, const float& tensor2Item){
            auto newItem = std::bit_cast<uint32t>(tensorItem);
            tensorItem = newItem ^ std::bit_cast<uint32t>(tensor2Item);
            return tensorItem;
        });
    }

    template <class T>
    void Tensor<T>::operator~() noexcept{
        
        forEach([](T& item){
            item = ~item;
        });
    }

    template <>
    void Tensor<bool>::operator~() noexcept{
        
        forEach([](bool& item){
            item = !item;
        });
    }

    template <>
    void Tensor<double>::operator~() noexcept{
        
        forEach([](double& item){
            item = std::bit_cast<double>(~std::bit_cast<uint64t>(item));
        });
    }

    template <>
    void Tensor<float>::operator~() noexcept{
        
        forEach([](float& item){
            item = std::bit_cast<float>(~std::bit_cast<uint32t>(item));
        });
    }

    template <class T>
    inline Tensor<T>* Tensor<T>::applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation) const noexcept{

        Tensor<T>* tensorOut = new Tensor<T>(dimensionSizes);
        tensorOut->tensor.resize(tensor.size());

        for(uint64t i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = operation(tensor[i], tensor2.tensor[i]);
        }

        return tensorOut;
    }

    template <>
    inline Tensor<double>* 
    Tensor<double>::applyAndReturn(const Tensor<double>& tensor2, const std::function<double(const double&, const double&)>& operation) const noexcept{

        Tensor<double>* tensorOut = new Tensor<double>(dimensionSizes);
        tensorOut->tensor.resize(tensor.size());

        for(uint64t i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = operation(tensor[i], tensor2.tensor[i]);
        }

        return tensorOut;
    }

    template <>
    inline Tensor<float>* 
    Tensor<float>::applyAndReturn(const Tensor<float>& tensor2, const std::function<float(const float&, const float&)>& operation) const noexcept{

        Tensor<float>* tensorOut = new Tensor<float>(dimensionSizes);
        tensorOut->tensor.resize(tensor.size());

        for(uint64t i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = operation(tensor[i], tensor2.tensor[i]);
        }

        return tensorOut;
    }

    template <class T>
    inline void Tensor<T>::apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation) noexcept{

        for(uint64t i = 0; i < tensor.size(); i++){
            operation(tensor[i], tensor2.tensor[i]);
        }
    }

    template <>
    inline void Tensor<bool>::apply(const Tensor<bool>& tensor2, const std::function<void(bool&, const bool&)>& operation) noexcept{

        for(uint64t i = 0; i < tensor.size(); i++){
            bool tensorItemValue = tensor.at(i);
            bool tensor2ItemValue = tensor2.tensor.at(i);
            operation(tensorItemValue, tensor2ItemValue);
        }
    }

    template <class T>
    void Tensor<T>::forEach(const std::function<void(T&)>& apply) noexcept{
        for(T& item : tensor){
            apply(item);
        }
    }

    template <>
    void Tensor<bool>::forEach(const std::function<void(bool&)>& apply) noexcept{
        
        for(uint64t i = 0; i < tensor.size(); i++){
            bool value = tensor.at(i);
            apply(value);
            tensor.at(i) = value;
        }
    }

    template <class T>
    void Tensor<T>::showTensor() const noexcept{

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
    void Tensor<T>::showItem(const std::vector<int>& coordinates) const noexcept{
        
        int itemNumber = getIndex(coordinates);
        std::cout << "Item: " << tensor[itemNumber] << '\n';
    }

    template <class T>
    Tensor<T>::~Tensor() noexcept{
        //
    }



    // private methods

    template <class T>
    std::vector<int> Tensor<T>::getCoords(int itemNumber) const noexcept{

        std::vector<int> coordinates;
        uint64t dimension = dimensionSizes.size();
        coordinates.resize(dimension);

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
    int Tensor<T>::getIndex(const std::vector<int>& coordinates) const noexcept{
        
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
    int Tensor<T>::getNumberOfItems(const std::vector<int>& dimensionSizes) const noexcept{

        int itemCounting = 1;
        for(const auto& dimensionSize : dimensionSizes){
            itemCounting *= dimensionSize;
        }
        return itemCounting;
    }
/*
    template<class T>
    inline bool Tensor<T>::compareItems(const double a, const double b) noexcept{

        //std::cout << "comparing: " <<  a << " vs " << b << std::endl;
        
        double veightedEpsilon = std::numeric_limits<double>::epsilon();
        return std::fabs(a - b) < veightedEpsilon;
    }*/

    template<>
    inline bool Tensor<double>::compareItems(const double a, const double b) noexcept{
        
        double veightedEpsilon = std::numeric_limits<double>::epsilon();
        return std::fabs(a - b) < veightedEpsilon;
    }

    template<>
    inline bool Tensor<float>::compareItems(const float a, const float b) noexcept{
        
        float veightedEpsilon = std::numeric_limits<float>::epsilon();
        return std::fabs(a - b) < veightedEpsilon;
    }

    template <class T>
    inline bool Tensor<T>::compareItems(const T& a, const T& b) const noexcept{
        
        return a == b;
    }

    template <class T>
    void Tensor<T>::constructorMessage(const std::vector<int>& dimensionSizes) const noexcept{

        // Message
        std::cout << "A tensor of " << tensor.size() << " items and " << dimensionSizes.size() << " dimensions been allocated.\n";
        std::cout << "A tensor dimensions are as follows: ";
        for(const auto& dimensionSize : dimensionSizes){
            std::cout << dimensionSize << " ";
        }

        std::cout << "\n\n";
    }
