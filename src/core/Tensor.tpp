#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "Tensor.hpp" // Not needed, just keep it for intelisense

namespace GeMa{

    const int maxLoopCount = 65536; // Will be probably unused.

    // Helper converter from floating types to integral
    template<typename F>
    struct float_to_integral{
        using type = std::conditional_t<sizeof(F) == 4, uint32_t, 
                     std::conditional_t<sizeof(F) == 8, uint64_t,
                     void>>;
    };

    // Primitives and simple types
    //template <typename T> class Tensor<T>;
    //template class Tensor<T>;
    /*template class Tensor<bool>;
    template class Tensor<char>;
    template class Tensor<short>;
    template class Tensor<int>;
    template class Tensor<long long int>;
    template class Tensor<float>;
    template class Tensor<double>;*/
    
    //template <typename T> class Tensor<Tensor<T>*>;
    //template <typename O> class Tensor<O&>;
    //template <typename O> class Tensor<O*>;
    // Others
    //template <typename O> class Tensor<std::vector<O>>;
    //template <typename O> class Tensor<std::unique_ptr<O>>;



    // public methods

    template <class T>
    Tensor<T>::Tensor(const std::vector<int>& newTensorDimensionSizes) noexcept
    : dimensionSizes_(newTensorDimensionSizes) {

        int itemCounting = calculateNumberOfItems(newTensorDimensionSizes);

        tensor_.resize(itemCounting);

        defaultFunctions();
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
        return dimensionSizes_;
    }

    template <class T>
    uint64_t Tensor<T>::getNumberOfDimensions() const noexcept{
        return dimensionSizes_.size();
    }

    template <class T>
    uint64_t Tensor<T>::getNumberOfItems() const noexcept{
        return tensor_.size();
    }

    template <class T>
    T Tensor<T>::getItem(const std::vector<int>& coordinates) const noexcept{

        return tensor_[getIndex(coordinates)];
    }
    
    template <class T>
    void Tensor<T>::setItem(const T& value, const std::vector<int>& coordinates) noexcept{

        int itemIndex = getIndex(coordinates);
        tensor_[itemIndex] = value;
    }

    // Secure version will need to check for correct tensorItems size
    template <class T>
    void Tensor<T>::setItems(const std::vector<T>& tensorItems) noexcept{

        //int copyLength = fmin(tensorItems.size(), tensor.size()); //readd in safety wrapper

        // why to use this?
        /*for(uint64_t i = 0; i < tensor_.size(); i++){
            tensor_[i] = tensorItems[i];
        }*/

        tensor_ = tensorItems;
    }

    template <class T>
    void Tensor<T>::setTensorOutput(const std::function<void(const T&)>& tensorOutput) noexcept{
        this->tensorOutput_ = tensorOutput;
    }

    template <class T>
    void Tensor<T>::setItemOutput(const std::function<void(const T&, const std::vector<int>&)>& itemOutput) noexcept{
        this->itemOutput_ = itemOutput;
    }

    template <class T>
    bool Tensor<T>::isEquilateral() const noexcept{
        return std::adjacent_find(dimensionSizes_.begin(), dimensionSizes_.end(), std::not_equal_to<int>()) == dimensionSizes_.end();
    }

    template <class T>
    std::string Tensor<T>::toString() const noexcept{

        std::vector<std::string> openingBrackets(tensor_.size());
        std::fill(openingBrackets.begin(), openingBrackets.end(), "");

        std::vector<std::string> closingBrackets(tensor_.size());
        std::fill(closingBrackets.begin(), closingBrackets.end(), "");

        uint64_t dimensionProduct = tensor_.size();

        //for(uint64_t i = dimensionSizes_.size() - 1; (i >= 0) && (i < dimensionSizes_.size()); i--){ // Opposite endianness
        for(uint64_t i = 0; i < dimensionSizes_.size(); ++i){ // Identity endianness

            for(uint64_t j = 0; j < tensor_.size(); j++){

                if(j % dimensionProduct == 0){
                    openingBrackets[j] += "{";
                }

                if(j % dimensionProduct == (dimensionProduct - 1)){
                    closingBrackets[j] += "}";
                }
            }

            dimensionProduct /= dimensionSizes_[i];
        }

        std::string output = "";

        for(uint64_t i = 0; i < tensor_.size(); i++){
            output += 
                std::format("{}{}{}{}", openingBrackets[i], tensor_[i], closingBrackets[i], (((i + 1) >= tensor_.size()) ? "" : ", "));
        }

        return output; 
    }

    template <class T>
    void Tensor<T>::parse(const std::string& tensorString, const std::function<const T(const std::string&)>& parseItem) noexcept{
        
        uint64_t i;
        for(i = 0; tensorString[i] == '{'; ++i){}
        std::vector<int> parsedDimensionSizes(i);

    }

    // TODO: check is this is actually needed
    template <class T>
    inline constexpr Tensor<T>* Tensor<T>::copy() const noexcept{

        Tensor<T>* newTensor = new Tensor<T>(dimensionSizes_);

        for(uint64_t i = 0; i < tensor_.size(); i++){
            newTensor->tensor_[i] = *(new T(tensor_[i]));
        }

        return newTensor;
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill) noexcept requires(!std::is_same<T, bool>::value){

        // could use assign method on everything thus dodging the specialization but std::fill is probably faster
        tensor_.assign(tensor_.size(), fill); 

        /*for(T& item : tensor_){
            item = fill;
        }*/
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill) noexcept requires(std::is_same<T, bool>::value){

        std::fill(tensor_.begin(), tensor_.end(), fill); // probably better optimalized

        // 
        /*for(uint64_t i = 0; i < tensor_.size(); i++){
            tensor_.at(i) = fill;
        }*/
    }

    template <class T>
    Tensor<T>* Tensor<T>::transposition(const int dim1, const int dim2) const noexcept{

        // Copying the dimensionSizes
        // Change assigment to just construction of correct size
        std::vector<int> transposedDimensionSizes = dimensionSizes_; 

        // Swapping the dimension sizes
        transposedDimensionSizes[dim1] = dimensionSizes_[dim2]; 
        transposedDimensionSizes[dim2] = dimensionSizes_[dim1];
        
        // Initializing the new tensor
        Tensor<T>* tensorTransposed = new Tensor<T>(transposedDimensionSizes);

        std::vector<int> original, switched;
        original.resize(dimensionSizes_.size());
        switched.resize(dimensionSizes_.size());

        // Looping thru elements in tensor and swapping the desired coordinates
        for(uint64_t i = 0; i < tensor_.size(); i++){
            
            // Switching the two coordinated corresponding to the two dimensions we want to switch
            original = getCoords(i);

            // The swap of two desired coordinates
            switched = original;
            switched[dim1] = original[dim2];
            switched[dim2] = original[dim1];

            // Works until now, check the getIndex function if it actually works properly
            tensorTransposed->tensor_[tensorTransposed->getIndex(switched)] = tensor_[getIndex(original)];
        }

        return tensorTransposed;
    }

    template <class T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor2) noexcept{
        
        this->tensor_ = tensor2.tensor_;
        this->dimensionSizes_ = tensor2.dimensionSizes_;

        // TODO: decide if to actually copy this
        this->equals_ = tensor2.equals_;
        this->order_ = tensor2.order_;

        // TODO: decide what to do with tensorOutput and itemOutput
        return *this;
    }

    // Do not simplify
    template <class T>
    bool Tensor<T>::operator==(const Tensor<T>& tensor2) const noexcept{

        // Values should be compared first, as tensors of same dimensions are more likely to be compared
        //return (this->tensor_ == tensor2.tensor_) && (this->dimensionSizes_ == tensor2.dimensionSizes_);

        // Too complicated
        /*if(this->tensor_.size() != tensor2.tensor_.size()){
            return false;
        }

        // Check if all items are equal
        return std::equal(this->tensor_.begin(), this->tensor_.end(), tensor2.tensor_.begin(), [&](const auto& a, const auto& b){

            return compareItems(a, b);
        });*/

        // This is the implementation similar to std::vector::oprator== workings, but with possibility of custom comparison function
        return std::equal(this->tensor_.begin(), this->tensor_.end(), tensor2.tensor_.begin(), [this](const auto& a, const auto& b){

            return equals_(a, b); //was: compareItems

        }) && (this->dimensionSizes_ == tensor2.dimensionSizes_); // Could be also: !(this->tensor_.size() - tensor2.tensor_.size())
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
            return std::bit_cast<uint64_t>(tensorItem) | std::bit_cast<uint64_t>(tensor2Item);
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
            return std::bit_cast<uint64_t>(tensorItem) & std::bit_cast<uint64_t>(tensor2Item);
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
            return std::bit_cast<uint64_t>(tensorItem) ^ std::bit_cast<uint64_t>(tensor2Item);
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

        Tensor<T>* tensorOut = new Tensor<T>(dimensionSizes_);
        tensorOut->tensor_.resize(tensor_.size());

        for(uint64_t i = 0; i < tensor_.size(); i++){
            tensorOut->tensor_[i] = operation(tensor_[i], tensor2.tensor_[i]);
        }

        return tensorOut;
    }

    template <class T>
    inline Tensor<T>* Tensor<T>::applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation)
    const noexcept requires(std::is_floating_point<T>::value){

        Tensor<T>* tensorOut = new Tensor<T>(dimensionSizes_);
        tensorOut->tensor_.resize(tensor_.size());

        for(uint64_t i = 0; i < tensor_.size(); i++){
            tensorOut->tensor_[i] = operation(tensor_[i], tensor2.tensor_[i]);
        }

        return tensorOut;
    }

    template <class T>
    inline void Tensor<T>::apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    noexcept requires(!std::is_same<T, bool>::value){

        for(uint64_t i = 0; i < tensor_.size(); i++){
            operation(tensor_[i], tensor2.tensor_[i]);
        }
    }

    template <class T>
    inline void Tensor<T>::apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    noexcept requires(std::is_same<T, bool>::value){

        for(uint64_t i = 0; i < tensor_.size(); i++){
            bool tensorItemValue = tensor_.at(i);
            bool tensor2ItemValue = tensor2.tensor_.at(i);
            operation(tensorItemValue, tensor2ItemValue);
        }
    }

    template <class T>
    void Tensor<T>::forEach(const std::function<void(T&)>& apply) noexcept requires(!std::is_same<T, bool>::value){
        for(T& item : tensor_){
            apply(item);
        }
    }

    template <class T>
    void Tensor<T>::forEach(const std::function<void(T&)>& apply) noexcept requires(std::is_same<T, bool>::value){
        
        for(uint64_t i = 0; i < tensor_.size(); i++){
            bool value = tensor_.at(i);
            apply(value);
            tensor_.at(i) = value;
        }
    }

    template <class T>
    Tensor<T>::~Tensor() noexcept{
        //
    }



    // private methods

    template <class T>
    std::vector<int> Tensor<T>::getCoords(int itemIndex) const noexcept{

        std::vector<int> coordinates;
        coordinates.resize(dimensionSizes_.size());

        uint64_t divisor = tensor_.size();
        
        for(uint64_t i = 0; i < dimensionSizes_.size(); ++i){

            divisor /= dimensionSizes_[i];
            coordinates[i] = itemIndex / divisor;
            itemIndex %= divisor;
        }

        return coordinates;
    }

    template <class T>
    int Tensor<T>::getIndex(const std::vector<int>& coordinates) const noexcept{
        
        int itemIndex = 0;
        int dimensionProduct = 1;

       for(uint64_t i = dimensionSizes_.size() - 1; i < dimensionSizes_.size(); --i){

            itemIndex += coordinates[i] * dimensionProduct;
            dimensionProduct *= dimensionSizes_[i];
       }

        return itemIndex;
    }

    template <class T>
    std::vector<int> Tensor<T>::littleGetCoords(int itemIndex) const noexcept{

        std::vector<int> coordinates;
        coordinates.resize(dimensionSizes_.size());
        uint64_t divisor = tensor_.size();
        
        for(uint64_t i = dimensionSizes_.size() - 1; i < dimensionSizes_.size(); --i){

            divisor /= dimensionSizes_[i];
            coordinates[i] = itemIndex / divisor;
            itemIndex %= divisor;
        }

        return coordinates;
    }

    template <class T>
    int Tensor<T>::littleGetIndex(const std::vector<int>& coordinates) const noexcept{
        
        int itemIndex = 0;
        int dimensionProduct = 1;

       for(uint64_t i = 0; i < dimensionSizes_.size(); ++i){

            itemIndex += coordinates[i] * dimensionProduct;
            dimensionProduct *= dimensionSizes_[i];
       }

        return itemIndex;
    }

    template <class T>
    int Tensor<T>::calculateNumberOfItems(const std::vector<int>& dimensionSizes) const noexcept{

        return std::accumulate(dimensionSizes_.begin(), dimensionSizes_.end(), 1, std::multiplies<int>());
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
    void Tensor<T>::defaultFunctions() noexcept{

        if constexpr (std::is_floating_point<T>::value){

            equals_ = [](const T a, const T b){
                T epsilon = std::numeric_limits<T>::epsilon();
                return std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)));
                
                // this does not work: std::fabs(a - b) < (epsilon * std::max(std::fabs(a), std::fabs(b)));
                // but this does: std::fabs(a - b) < epsilon;
                // luckily this works: std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)));
            };

        }else{

            equals_ = [](const T& a, const T& b){
                return a == b;
            };
        }
    }
}
