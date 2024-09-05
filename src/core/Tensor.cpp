#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <bit>
#include <format>

#include "Tensor.hpp"

#define MAX_LOOP_COUNT 1024;
#define uint64t uint64_t
#define uint32t uint32_t

namespace GeMa{

    // Helper converter from floating types to integral
    template<typename F>
    struct float_to_integral{
        using type = std::conditional_t<sizeof(F) == 4, uint32_t, 
                     std::conditional_t<sizeof(F) == 8, uint64_t,
                     void>>;
    };

    // Primitives and simple types
    template class Tensor<bool>;
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
    inline void* Tensor<T>::getPointer(const std::vector<int>& coordinates) const noexcept requires(!std::is_same<T, bool>::value){
        
        return (void*)&tensor[getIndex(coordinates)];
    }

    template <class T>
    inline void* Tensor<T>::getPointer(const std::vector<int>& coordinates) const noexcept requires(std::is_same<T, bool>::value){
        
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
    std::string Tensor<T>::toString() const noexcept{

        std::string output = "";

        // Is used for checking if there is a new dimension coming
        std::vector<int> itemsFromBegin(dimensionSizes.size());
        std::fill(itemsFromBegin.begin(), itemsFromBegin.end(), -1);

        // Is used for checking if there is a new dimension coming
        std::vector<int> itemsFromEnd(dimensionSizes.size());
        std::fill(itemsFromEnd.begin(), itemsFromEnd.end(), -1);

        // Saves all opening brackets to one item in a tensor
        std::vector<std::string> beginBracketsOfItem(tensor.size());
        std::fill(beginBracketsOfItem.begin(), beginBracketsOfItem.end(), "");

        // Saves all closing brackets to one item in a tensor
        std::vector<std::string> endBracketsOfItem(tensor.size());
        std::fill(endBracketsOfItem.begin(), endBracketsOfItem.end(), "");

        // Looping through items
        for(uint64t i = 0; i < tensor.size(); i++){
            
            /*std::cout << "item: " << tensor[i] << " item coords: " << std::endl;
            for(uint64t j = 0; j < itemCoords.size(); j++){
                std::cout << itemCoords[j] << ", ";
            }
            std::cout << std::endl;*/
           getItemsOpeningBrackets(i, itemsFromBegin, beginBracketsOfItem);
           getItemsClosingBrackets(i, itemsFromEnd, endBracketsOfItem);
        }

        for(uint64t i = 0; i < tensor.size(); i++){

            if(endBracketsOfItem[i].empty()){
                output += std::format("{}{}, {}", beginBracketsOfItem[i], tensor[i], endBracketsOfItem[i]);
            }else{
                output += std::format("{}{}{}", beginBracketsOfItem[i], tensor[i], endBracketsOfItem[i]);
            }
        }


        return output;
    }

    template <class T>
    void Tensor<T>::getItemsOpeningBrackets(const uint64t i, std::vector<int>& itemsFromBegin, std::vector<std::string>& beginBracketsOfItem) const noexcept{
        
        std::vector<int> itemCoords = getCoords(i); // This doesnt have to be here if put right into the toString()
        
        for(uint64t j = 0; j < dimensionSizes.size(); j++){

            if(itemCoords[j] <= 0){

                if(itemsFromBegin[j] == itemCoords[j]){
                    continue;
                }

                beginBracketsOfItem[i] += '{';
            }

            itemsFromBegin[j] = itemCoords[j];
        }
    }

    template <class T>
    void Tensor<T>::getItemsClosingBrackets(const uint64t i, std::vector<int>& itemsFromEnd, std::vector<std::string>& endBracketsOfItem) const noexcept{

        // Inverting the index to simulate looping backwards through items
        uint64t inverseIndex = tensor.size() - i - 1;
        std::vector<int> itemCoords = getCoords(inverseIndex); // This can be just assigment without declaration if put right into toString()

        for(uint64t j = 0; j < dimensionSizes.size(); j++){

            if(itemCoords[j] >= dimensionSizes[j] - 1){

                if(itemsFromEnd[j] == itemCoords[j]){
                    continue;
                }

                endBracketsOfItem[inverseIndex] += '}';
            }
            
            itemsFromEnd[j] = itemCoords[j];
        }
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
    void Tensor<T>::fillWith(const T& fill) noexcept requires(!std::is_same<T, bool>::value){

        for(T& item : tensor){
            item = fill;
        }
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill) noexcept  requires(std::is_same<T, bool>::value){

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

    // TODO: also compare dimension sizes
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

    template<typename T> //template<typename F, typename std::enable_if<!std::is_floating_point<F>::value, double>::type>
    Tensor<T>* Tensor<T>::operator|(const Tensor<T>& tensor2) const noexcept requires(!std::is_floating_point<T>::value){

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){

            return tensorItem | tensor2Item;
        });
    }

    template<typename T> //template<typename F, typename std::enable_if<std::is_floating_point<F>::value, double>::type>
    Tensor<T>* Tensor<T>::operator|(const Tensor<T>& tensor2) const noexcept requires(std::is_floating_point<T>::value){

        return applyAndReturn(tensor2, [](const double& tensorItem, const double& tensor2Item){
            return std::bit_cast<uint64t>(tensorItem) | std::bit_cast<uint64t>(tensor2Item);
        });
    }

    template <class T>
    void Tensor<T>::operator|=(const Tensor<T>& tensor2) noexcept requires(!std::is_floating_point<T>::value){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem |= tensor2Item;
        });
    }

    template <class T>
    void Tensor<T>::operator|=(const Tensor<T>& tensor2) noexcept requires(std::is_floating_point<T>::value){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            auto newItem = std::bit_cast<typename float_to_integral<T>::type>(tensorItem);
            tensorItem = newItem | std::bit_cast<typename float_to_integral<T>::type>(tensor2Item);
            return tensorItem;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator&(const Tensor<T>& tensor2) const noexcept requires(!std::is_floating_point<T>::value){

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem & tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator&(const Tensor<T>& tensor2) const noexcept requires(std::is_floating_point<T>::value){

        return applyAndReturn(tensor2, [](const double& tensorItem, const double& tensor2Item){
            return std::bit_cast<uint64t>(tensorItem) & std::bit_cast<uint64t>(tensor2Item);
        });
    }

    template <class T>
    void Tensor<T>::operator&=(const Tensor<T>& tensor2) noexcept requires(!std::is_floating_point<T>::value){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem &= tensor2Item;
        });
    }

    template <class T>
    void Tensor<T>::operator&=(const Tensor<T>& tensor2) noexcept requires(std::is_floating_point<T>::value){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            auto newItem = std::bit_cast<typename float_to_integral<T>::type>(tensorItem);
            tensorItem = newItem & std::bit_cast<typename float_to_integral<T>::type>(tensor2Item);
            return tensorItem;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator^(const Tensor<T>& tensor2) const noexcept requires(!std::is_floating_point<T>::value){

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem ^ tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator^(const Tensor<T>& tensor2) const noexcept requires(std::is_floating_point<T>::value){

        return applyAndReturn(tensor2, [](const double& tensorItem, const double& tensor2Item){
            return std::bit_cast<uint64t>(tensorItem) ^ std::bit_cast<uint64t>(tensor2Item);
        });
    }

    template <class T>
    void Tensor<T>::operator^=(const Tensor<T>& tensor2) noexcept requires(!std::is_floating_point<T>::value){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem ^= tensor2Item;
        });
    }

    template <class T>
    void Tensor<T>::operator^=(const Tensor<T>& tensor2) noexcept requires(std::is_floating_point<T>::value){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            auto newItem = std::bit_cast<typename float_to_integral<T>::type>(tensorItem);
            tensorItem = newItem ^ std::bit_cast<typename float_to_integral<T>::type>(tensor2Item);
            return tensorItem;
        });
    }

    template <class T>
    void Tensor<T>::operator~() noexcept requires(!std::is_floating_point<T>::value && !std::is_same<T, bool>::value){
        
        forEach([](T& item){
            item = ~item;
        });
    }

    template <class T>
    void Tensor<T>::operator~() noexcept requires(std::is_same<T, bool>::value){
        
        forEach([](T& item){
            item = !item;
        });
    }

    template <class T>
    void Tensor<T>::operator~() noexcept requires(std::is_floating_point<T>::value){
        
        forEach([](T& item){
            item = std::bit_cast<T>(~std::bit_cast<typename float_to_integral<T>::type>(item));
        });
    }

    template <class T>
    inline Tensor<T>* Tensor<T>::applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation)
    const noexcept requires(!std::is_floating_point<T>::value){

        Tensor<T>* tensorOut = new Tensor<T>(dimensionSizes);
        tensorOut->tensor.resize(tensor.size());

        for(uint64t i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = operation(tensor[i], tensor2.tensor[i]);
        }

        return tensorOut;
    }

    template <class T>
    inline Tensor<T>* Tensor<T>::applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation)
    const noexcept requires(std::is_floating_point<T>::value){

        Tensor<T>* tensorOut = new Tensor<T>(dimensionSizes);
        tensorOut->tensor.resize(tensor.size());

        for(uint64t i = 0; i < tensor.size(); i++){
            tensorOut->tensor[i] = operation(tensor[i], tensor2.tensor[i]);
        }

        return tensorOut;
    }

    template <class T>
    inline void Tensor<T>::apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    noexcept requires(!std::is_same<T, bool>::value){

        for(uint64t i = 0; i < tensor.size(); i++){
            operation(tensor[i], tensor2.tensor[i]);
        }
    }

    template <class T>
    inline void Tensor<T>::apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    noexcept requires(std::is_same<T, bool>::value){

        for(uint64t i = 0; i < tensor.size(); i++){
            bool tensorItemValue = tensor.at(i);
            bool tensor2ItemValue = tensor2.tensor.at(i);
            operation(tensorItemValue, tensor2ItemValue);
        }
    }

    template <class T>
    void Tensor<T>::forEach(const std::function<void(T&)>& apply) noexcept requires(!std::is_same<T, bool>::value){
        for(T& item : tensor){
            apply(item);
        }
    }

    template <class T>
    void Tensor<T>::forEach(const std::function<void(T&)>& apply) noexcept requires(std::is_same<T, bool>::value){
        
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
        for(const int dimensionSize : dimensionSizes){
            itemCounting *= dimensionSize;
        }
        return itemCounting;
    }

    template <class T>
    inline bool Tensor<T>::compareItems(const T& a, const T& b) const noexcept requires(!std::is_floating_point<T>::value){
        
        return a == b;
    }

    template <class T>
    inline bool Tensor<T>::compareItems(const T a, const T b) const noexcept requires(std::is_floating_point<T>::value){
        
        T veightedEpsilon = std::numeric_limits<T>::epsilon();
        return std::fabs(a - b) < veightedEpsilon;
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

}
