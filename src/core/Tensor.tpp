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

namespace gema{

    template<class T>
    concept has_to_string = requires(const T& t) {
        { t.to_string() } -> std::convertible_to<std::string>;
    };

    template<typename T>
    concept is_formattable = requires(T t, std::format_context& ctx) {
        std::formatter<T, char>{};                         // constructible
        std::formatter<T, char>{}.format(t, ctx);          // callable
        //std::format("{}", t);
    };
}

namespace std{

    template <class T>
    struct formatter<gema::Tensor<T>, char> {
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();  // No custom format specs supported
        }

        auto format(const gema::Tensor<T>& tensor, format_context& ctx) const {
            // Use tensor.toString() to get the string representation
            return format_to(ctx.out(), "{}", tensor.toString());
        }
    };

    template <class T>
    struct formatter<gema::Tensor<T>*, char> {
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();  // No custom format specs supported
        }

        auto format(const gema::Tensor<T>* tensor, format_context& ctx) const {
            // Use tensor.toString() to get the string representation
            return format_to(ctx.out(), "{}", tensor->toString());
        }
    };
/*
    template <size_t N>
    struct formatter<bitset<N>, char> {
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();
        }

        auto format(const bitset<N>& bs, format_context& ctx) const {
            return format_to(ctx.out(), "{}", bs.to_string());
        }
    };*/

    template <class T>
    requires (!gema::is_formattable<T> && gema::has_to_string<T>)
    struct formatter<T, char>{
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();
        }

        auto format(const T& t, format_context& ctx) const {
            return format_to(ctx.out(), "{}", t.to_string());
        }
    };
}

namespace gema{

    constexpr int maxLoopCount = 65536; // Will be probably unused.

    // Meta helpers:
    
    // Converts from floating types to integral of same size
    template<typename F>
    struct to_integral {
        using type = std::conditional_t<sizeof(F) == 1, uint8_t,
                     std::conditional_t<sizeof(F) == 2, uint16_t,
                     std::conditional_t<sizeof(F) == 4, uint32_t, 
                     std::conditional_t<sizeof(F) == 8, uint64_t,
                     void>>>>;
    };

    // If F is floating point, it will get converted to integral and if not, then just returned
    template<typename F>
    struct integral_if_float{
        using type = std::conditional_t<std::is_floating_point_v<F>, to_integral<F>, F>;
    };

    // If argument is floating type, it will get bitcasted to integral of same size, otherwise just returned
    template<typename T>
    inline constexpr integral_if_float<T> bitcast_if_float(const T& value){

        // use this if the bitcast wont get optimized away to nonfloating types
        integral_if_float<T> valueBits;
        if constexpr(std::is_floating_point<T>::value){
            valueBits = std::bit_cast<typename to_integral<T>::type>(value);
        }else{
            valueBits = value;
        }

        return valueBits;
    }

    // Takes two parameters and returns length of tensor_ of the first one to be type Tensor<T>
    template<typename T, typename A, typename B>
    consteval uint64_t tensor_size(const A& operand1, const B& operand2){

        if constexpr (std::is_same_v<A, Tensor<T>>){
            return operand1.tensor_.size();
        } else if constexpr (std::is_same_v<B, Tensor<T>>){
            return operand2.tensor_.size();
        }else{
            return 1;
        }
    }

    // Returns first argument of two arguments that matches given type
    template<typename X, typename T, typename A, typename B>
    consteval Tensor<T>* type_pick_b(const A& operand1, const B& operand2){
        
        if constexpr (std::is_same_v<A, X>){
            return &operand1;
        } else if constexpr (std::is_same_v<B, X>){
            return &operand2;
        } else {
            return nullptr;
        }
    }

    // Returns first argument that matches given type
    template<typename X, typename F, typename... R>
    inline const X* type_pick(const F& first, const R&... rest) { 

        if constexpr (std::is_same_v<std::decay_t<F>, X>) {
            return (&first);
        } else if constexpr (sizeof...(rest) > 0) {
            return type_pick<X>(rest...);
        } else {
            return nullptr; // No match found
        }
    }
    
    // Returns first type of two type arguments that matches given type
    template<typename X, typename A, typename B, typename T>
    struct first_of_specified{
        using type =    std::conditional<std::is_same_v<A, X>, A,
                        std::conditional<std::is_same_v<B, X>, B, 
                        void>>;
    };

    

    



    // static private default values:

    template <class T>
    inline std::function<bool(const T&, const T&)> Tensor<T>::defaultEquals_ = [] (const T& a, const T& b) {
        if constexpr (std::is_floating_point<T>::value) {
            T epsilon = std::numeric_limits<T>::epsilon();
            return std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)));
        } else {
            return a == b;
        }
    };

    template <class T>
    inline std::function<int(const T&, const T&)> Tensor<T>::defaultOrder_ = [] (const T& a, const T& b) {

        if constexpr (std::is_floating_point<T>::value) {
            T epsilon = std::numeric_limits<T>::epsilon();
            if (std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)))){
                return 0;
            }else if(a > b){
                return 1;
            }else{
                return -1;
            }

        } else if constexpr (std::is_integral<T>::value){
            return (a != b) * ((a > b) + -(a < b));
        }else{
            return 0;
        }
    };



    // public methods:

    template <class T>
    Tensor<T>::Tensor(const std::vector<uint64_t>& newTensorDimensionSizes) : dimensionSizes_(newTensorDimensionSizes) {
    
        int itemCounting = calculateNumberOfItems(newTensorDimensionSizes);

        tensor_.resize(itemCounting);
    }

    template <class T>
    Tensor<T>::Tensor(const Tensor<T>& otherTensor){
        *this = otherTensor;
    }

    template <class T>
    Tensor<T>::Tensor(const Tensor<T>* otherTensor){
        tensor_.resize(otherTensor->tensor_.size());
        dimensionSizes_ = otherTensor->dimensionSizes_;
    }

    template <class T>
    Tensor<T>::Tensor(){

    }

    template <class T>
    const std::vector<uint64_t>& Tensor<T>::getDimensionSizes() const{
        return dimensionSizes_;
    }

    template <class T>
    uint64_t Tensor<T>::getNumberOfDimensions() const{
        return dimensionSizes_.size();
    }

    template <class T>
    uint64_t Tensor<T>::getNumberOfItems() const{
        return tensor_.size();
    }

    template <class T>
    T Tensor<T>::getItem(const std::vector<uint64_t>& coordinates) const{

        return tensor_[getIndex(coordinates)];
    }
    
    template <class T>
    void Tensor<T>::setItem(const T& value, const std::vector<uint64_t>& coordinates){

        int itemIndex = getIndex(coordinates);
        tensor_[itemIndex] = value;
    }

    // Secure version will need to check for correct tensorItems size
    template <class T>
    void Tensor<T>::setItems(const std::vector<T>& tensorItems){
        tensor_ = tensorItems;
    }

    template <class T>
    void Tensor<T>::setEquals(const std::function<bool(const T&, const T&)>& equals){
        
        userEquals_ = equals;
        equals_ = &userEquals_;
    }

    template <class T>
    void Tensor<T>::setOrder(const std::function<int(const T&, const T&)>& order){

        userOrder_ = order;
        order_ = &userOrder_;
    }

    template <class T>
    void Tensor<T>::setTensorOutput(const std::function<void(const T&)>& tensorOutput){
        this->tensorOutput_ = tensorOutput;
    }

    template <class T>
    void Tensor<T>::setItemOutput(const std::function<void(const T&, const std::vector<uint64_t>&)>& itemOutput){
        this->itemOutput_ = itemOutput;
    }

    template <class T>
    bool Tensor<T>::isEquilateral() const{
        return std::adjacent_find(dimensionSizes_.begin(), dimensionSizes_.end(), std::not_equal_to<int>()) == dimensionSizes_.end();
    }

    template <class T>
    std::string Tensor<T>::toString() const{

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
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){
        return os << tensor.toString();
    }

    template <class T>
    void Tensor<T>::parse(const std::string& tensorString, const std::function<const T(const std::string&)>& parseItem){
        
        uint64_t i;
        for(i = 0; tensorString[i] == '{'; ++i){}
        std::vector<int> parsedDimensionSizes(i);

    }

    // TODO: check is this is actually needed
    template <class T>
    inline constexpr Tensor<T>* Tensor<T>::copy() const{

        Tensor<T>* newTensor = new Tensor<T>(dimensionSizes_);

        for(uint64_t i = 0; i < tensor_.size(); i++){
            newTensor->tensor_[i] = *(new T(tensor_[i]));
        }

        return newTensor;
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill) requires(!std::is_same<T, bool>::value){

        // could use assign method on everything thus dodging the specialization but std::fill is probably faster
        tensor_.assign(tensor_.size(), fill); 

        /*for(T& item : tensor_){
            item = fill;
        }*/
    }

    template <class T>
    void Tensor<T>::fillWith(const T& fill) requires(std::is_same<T, bool>::value){

        std::fill(tensor_.begin(), tensor_.end(), fill); // probably better optimalized

        // 
        /*for(uint64_t i = 0; i < tensor_.size(); i++){
            tensor_.at(i) = fill;
        }*/
    }

    template <class T>
    Tensor<T>* Tensor<T>::transposition(const int dim1, const int dim2) const{

        // Copying the dimensionSizes
        // Change assigment to just construction of correct size
        std::vector<uint64_t> transposedDimensionSizes = dimensionSizes_; 

        // Swapping the dimension sizes
        transposedDimensionSizes[dim1] = dimensionSizes_[dim2]; 
        transposedDimensionSizes[dim2] = dimensionSizes_[dim1];
        
        // Initializing the new tensor
        Tensor<T>* tensorTransposed = new Tensor<T>(transposedDimensionSizes);

        std::vector<uint64_t> original, switched;
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

    // SPECIAL OPERATOR OVERLOADS ---------------------------------------------------------------------------------------------
    // Does not need macros.
    template <class T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& tensor2){
        
        this->tensor_ = tensor2.tensor_;
        this->dimensionSizes_ = tensor2.dimensionSizes_;

        // TODO: decide if to actually copy this
        // TODO: can it be done without if statements?
        // Argument for yes: tensor that does not have these functions defined is not comparable
        if(tensor2.userEquals_){
            setEquals(tensor2.userEquals_);
        }

        if(tensor2.userOrder_){
            setOrder(tensor2.userOrder_);
        }

        // TODO: decide what to do with tensorOutput and itemOutput
        return *this;
    }

    // Do not simplify
    template <class T>
    bool Tensor<T>::operator==(const Tensor<T>& tensor2) const{

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

            return (*equals_)(a, b); //was: compareItems

        }) && (this->dimensionSizes_ == tensor2.dimensionSizes_); // Could be also: !(this->tensor_.size() - tensor2.tensor_.size())
    }

    template <class T>
    bool Tensor<T>::operator!=(const Tensor<T> &tensor2) const
    {
        return !(*this == tensor2);
    }

    // OPERATOR OVERLOADS -----------------------------------------------------------------------------------------------------
    // Operator overload implemetations are often repetetive. To reduce code duplicates, macros are created for the overloads
    // and their variants. A naming convention has been created to include in macro name information about its abstract
    // signature.
    //
    // Notation explanation:
    // The last letters in macro name (after the last _) mean type of operation (similar to function signature).
    // T - Tensor<T>
    // V - value of type T from Tensor<T>
    // o - operation
    // oe - operation in place
    // r - results in
    //
    // Examples: 
    // ToTrT means Tensor performing Operation with Tensor Resulting in new Tensor (Tensor operation Tensor = Tensor).
    // ToeV means Tensor performing Operation with Value in place (Tensor oepration= Value).

    // ARITHMETIC BINARY GENERIC MACRO ----------------------------------------------------------------------------------------
    // Artihmetic binary is an binary operation on two arithmetic types (or ones with overloaded operators acting like 
    // arithmetic).
    #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    /**/\
        template <class T>\
        inline Tensor<T>* Tensor<T>::operator OP_SYMBOL(const Tensor<T>& tensor2) const{\
    /**/\
            return applyAndReturn(*this, tensor2, [](const T& tensorItem, const T& tensor2Item){\
                return tensorItem OP_SYMBOL tensor2Item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    /**/\
        template<class T>\
        inline Tensor<T>* operator OP_SYMBOL(const Tensor<T>& tensor, const T& value){\
    /**/\
            return Tensor<T>::forEachAndReturn(tensor, [&value](const T& item){\
                return item OP_SYMBOL value;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
        template<class T>\
        inline Tensor<T>* operator OP_SYMBOL(const T& value, const Tensor<T>& tensor){\
    /**/\
            /* Do not delegate switched argument operator! While on numbers set the operation would be often commutative, */\
            /* it is not guaranteed to be so on every type and operation!*/\
            return Tensor<T>::forEachAndReturn(tensor, [&value](const T& item){\
                return value OP_SYMBOL item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    /**/\
        template <class T>\
        void Tensor<T>::operator OP_SYMBOL##=(const Tensor<T>& tensor2){\
    /**/\
            apply(tensor2, [](T& tensorItem, const T& tensor2Item){\
                tensorItem OP_SYMBOL##= tensor2Item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    /**/\
        template<class T>\
        void Tensor<T>::operator OP_SYMBOL##=(const T& value){\
    /**/\
            forEach([&value](T& item){\
                item OP_SYMBOL##= value;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    ARITHMETIC_BINARY(+)
    ARITHMETIC_BINARY(-)
    ARITHMETIC_BINARY(*)
    ARITHMETIC_BINARY(/)
    ARITHMETIC_BINARY(%)

    #define LOGICAL_BINARY(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    LOGICAL_BINARY(&&)
    LOGICAL_BINARY(||)

    #undef ARITHMETIC_BINARY_ToTrT // Macros no longer needed
    #undef ARITHMETIC_BINARY_ToVrT
    #undef ARITHMETIC_BINARY_VoTrT
    #undef ARITHMETIC_BINARY_ToeT
    #undef ARITHMETIC_BINARY_ToeV

    #undef LOGICAL_BINARY

    #undef ARITHMETIC_BINARY

/*
    // (+) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    inline Tensor<T>* Tensor<T>::operator+(const Tensor<T>& tensor2) const{

        return applyAndReturn(*this, tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem + tensor2Item;
        });
    }

    template<class T>
    inline Tensor<T>* operator+(const Tensor<T>& tensor, const T& value){
        //return applyAndReturn(tensor, value, [](const T& tensorItem, const T& singularValue){
        //    return tensorItem + singularValue;
        //});

        return forEachAndReturn(tensor, [&value](const T& item){
            return item + value;
        });
    }

    template<class T>
    inline Tensor<T>* operator+(const T& value, const Tensor<T>& tensor){

        // do not delegate switched argument operator
        return forEachAndReturn(tensor, [&value](const T& item){
            return value + item;
        });
    }

    template <class T>
    void Tensor<T>::operator+=(const Tensor<T>& tensor2){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem += tensor2Item;
        });
    }

    template<class T>
    void Tensor<T>::operator+=(const T& value){

        forEach([&value](T& item){
            item += value;
        });
    }

    // (-) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator-(const Tensor<T>& tensor2) const{

        return applyAndReturn(*this, tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem - tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* operator-(const Tensor<T>& tensor, const T& value){

        return forEachAndReturn(tensor, [&value](const T& item){
            return item - value;
        });
    }

    template <class T>
    Tensor<T>* operator-(const T& value, const Tensor<T>& tensor){
        
        return forEachAndReturn(tensor, [&value](const T& item){
            return value - item;
        });
    }

    template <class T>
    void Tensor<T>::operator-=(const Tensor<T>& tensor2){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem -= tensor2Item;
        });
    }

    template<class T>
    void Tensor<T>::operator-=(const T& value){

        forEach([&value](T& item){
            item -= value;
        });
    }

    // (*) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T> *Tensor<T>::operator*(const Tensor<T> &tensor2) const{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem * tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* operator*(const Tensor<T>& tensor, const T& value){

        return forEachAndReturn(tensor, [&value](const T& item){
            return item * value;
        });
    }

    template <class T>
    Tensor<T>* operator*(const T& value, const Tensor<T>& tensor){
        
        return forEachAndReturn(tensor, [&value](const T& item){
            return value * item;
        });
    }

    template <class T>
    void Tensor<T>::operator*=(const Tensor<T> &tensor2){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem *= tensor2Item;
        });
    }

    template<class T>
    void Tensor<T>::operator*=(const T& value){

        forEach([&value](T& item){
            item *= value;
        });
    }

    // (/) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T> *Tensor<T>::operator/(const Tensor<T> &tensor2) const{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem / tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* operator/(const Tensor<T>& tensor, const T& value){

        return forEachAndReturn(tensor, [&value](const T& item){
            return item / value;
        });
    }

    template <class T>
    Tensor<T>* operator/(const T& value, const Tensor<T>& tensor){
        
        return forEachAndReturn(tensor, [&value](const T& item){
            return value / item;
        });
    }

    template <class T>
    void Tensor<T>::operator/=(const Tensor<T> &tensor2){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem /= tensor2Item;
        });
    }

    template<class T>
    void Tensor<T>::operator/=(const T& value){

        forEach([&value](T& item){
            item /= value;
        });
    }

    // (%) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T> *Tensor<T>::operator%(const Tensor<T> &tensor2) const{
        
        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            return tensorItem % tensor2Item;
        });
    }

    template <class T>
    Tensor<T>* operator%(const Tensor<T>& tensor, const T& value){

        return forEachAndReturn(tensor, [&value](const T& item){
            return item % value;
        });
    }

    template <class T>
    Tensor<T>* operator%(const T& value, const Tensor<T>& tensor){
        
        return forEachAndReturn(tensor, [&value](const T& item){
            return value % item;
        });
    }

    template <class T>
    void Tensor<T>::operator%=(const Tensor<T> &tensor2){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            tensorItem %= tensor2Item;
        });
    }

    template<class T>
    void Tensor<T>::operator%=(const T& value){

        forEach([&value](T& item){
            item %= value;
        });
    }
*/

    // BITWISE BINARY GENERIC MACRO -------------------------------------------------------------------------------------------
    // Bitwise operations (including bitshift) will support floating types using bitcast inside the functions.
    #define BITWISE_BINARY_ToTrT(OP_SYMBOL)\
    /**/\
        template<typename T>\
        inline Tensor<T>* Tensor<T>::operator OP_SYMBOL(const Tensor<T>& tensor2) const{\
    /**/\
            return applyAndReturn(tensor2, [](const T tensorItem, const T tensor2Item){\
    /**/\
                if constexpr(std::is_floating_point<T>::value){\
                    auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);\
                    auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);\
                    return std::bit_cast<T>(itemBits OP_SYMBOL item2Bits);\
                }else{\
                    return tensorItem OP_SYMBOL tensor2Item;\
                }\
            });\
        }\
    /**/

    #define BITWISE_BINARY_ToVrT(OP_SYMBOL)\
    /**/\
        template <class T>\
        inline Tensor<T>* operator OP_SYMBOL(const Tensor<T>& tensor, const T& value){\
    /**/\
            integral_if_float<T> valueBits = bitcast_if_float(value);\
            /*integral_if_float<T> valueBits = std::bit_cast<typename integral_if_float<T>::type>(value);*/\
    /**/\
            return Tensor<T>::forEachAndReturn(tensor, [&valueBits](const T& item){\
    /**/\
                if constexpr(std::is_floating_point<T>::value){\
                    auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);\
                    return std::bit_cast<T>(itemBits OP_SYMBOL valueBits);\
                }else{\
                    return item OP_SYMBOL valueBits;\
                }\
            });\
        }\
    /**/

    #define BITWISE_BINARY_VoTrT(OP_SYMBOL)\
    /**/\
        template <class T>\
        inline Tensor<T>* operator OP_SYMBOL(const T& value, const Tensor<T>& tensor){\
    /**/\
            integral_if_float<T> valueBits = bitcast_if_float(value);\
    /**/\
            return Tensor<T>::forEachAndReturn(tensor, [&valueBits](const T& item){\
    /**/\
                if constexpr(std::is_floating_point<T>::value){\
                    auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);\
                    return std::bit_cast<T>(valueBits OP_SYMBOL itemBits);\
                }else{\
                    return valueBits OP_SYMBOL item;\
                }\
            });\
        }\
    /**/

    #define BITWISE_BINARY_ToeT(OP_SYMBOL)\
    /**/\
        template <class T>\
        void Tensor<T>::operator OP_SYMBOL##=(const Tensor<T>& tensor2){\
    /**/\
            apply(tensor2, [](T& tensorItem, const T& tensor2Item){\
    /**/\
                if constexpr(std::is_floating_point<T>::value){\
                    auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);\
                    auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);\
                    tensorItem = std::bit_cast<T>(itemBits OP_SYMBOL item2Bits);\
                }else{\
                    tensorItem OP_SYMBOL##= tensor2Item;\
                }\
            });\
        }\
    /**/

    #define BITWISE_BINARY_ToeV(OP_SYMBOL)\
    /**/\
        template <class T>\
        void Tensor<T>::operator OP_SYMBOL##=(const T &value){\
    /**/\
            integral_if_float<T> valueBits = bitcast_if_float(value);\
    /**/\
            forEach([&valueBits](T& item){\
    /**/\
                if constexpr(std::is_floating_point<T>::value){\
                    auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);\
                    item = std::bit_cast<T>(itemBits OP_SYMBOL valueBits);\
                }else{\
                    item OP_SYMBOL##= valueBits;\
                }\
            });\
        }\
    /**/

    // To expand all possible operator overloads for binary bitwise operations
    #define BITWISE_BINARY(OP_SYMBOL)\
        BITWISE_BINARY_ToTrT(OP_SYMBOL)\
        BITWISE_BINARY_ToVrT(OP_SYMBOL)\
        BITWISE_BINARY_VoTrT(OP_SYMBOL)\
        BITWISE_BINARY_ToeT(OP_SYMBOL)\
        BITWISE_BINARY_ToeV(OP_SYMBOL)

    BITWISE_BINARY(|)
    BITWISE_BINARY(&)
    BITWISE_BINARY(^)

    // Some bitwise overloads for binary operations are not making sense for bitshift
    #define BITSHIFTLIKE(OP_SYMBOL)\
        BITWISE_BINARY_ToTrT(OP_SYMBOL)\
        BITWISE_BINARY_ToVrT(OP_SYMBOL)\
        BITWISE_BINARY_ToeT(OP_SYMBOL)\
        BITWISE_BINARY_ToeV(OP_SYMBOL)

    BITSHIFTLIKE(<<)
    BITSHIFTLIKE(>>)

    #undef BITWISE_BINARY_ToTgT // Macros no longer needed 
    #undef BITWISE_BINARY_ToVgT
    #undef BITWISE_BINARY_VoTgT
    #undef BITWISE_BINARY_ToeT
    #undef BITWISE_BINARY_ToeV
    #undef BITSHIFTLIKE
    #undef BITWISE_BINARY 

/*
    // (|) --------------------------------------------------------------------------------------------------------------------
   
    template<typename T>
    Tensor<T>* Tensor<T>::operator|(const Tensor<T>& tensor2) const{

        return applyAndReturn(tensor2, [](const T tensorItem, const T tensor2Item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                return std::bit_cast<T>(itemBits | item2Bits);
            }else{
                return tensorItem | tensor2Item;
            }
        });
    }

    template <class T>
    Tensor<T>* operator|(const Tensor<T>& tensor, const T& value){

        integral_if_float<T> valueBits = bitcast_if_float(value);
        //integral_if_float<T> valueBits = std::bit_cast<typename integral_if_float<T>::type>(value);

        return forEachAndReturn(tensor, [&valueBits](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                return std::bit_cast<T>(itemBits | valueBits);
            }else{
                return item | valueBits;
            }
        });
    }

    template <class T>
    inline Tensor<T>* operator|(const T& value, const Tensor<T>& tensor){

        integral_if_float<T> valueBits = bitcast_if_float(value);

        return forEachAndReturn(tensor, [&valueBits](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                return std::bit_cast<T>(valueBits | itemBits);
            }else{
                return valueBits | item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator|=(const Tensor<T>& tensor2){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits | item2Bits);
            }else{
                tensorItem |= tensor2Item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator|=(const T &value)
    {
        integral_if_float<T> valueBits = bitcast_if_float(value);

        forEach([&valueBits](T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                item = std::bit_cast<T>(itemBits | valueBits);
            }else{
                item |= valueBits;
            }
        });
    }

    // (&) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator&(const Tensor<T>& tensor2) const{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                return std::bit_cast<T>(itemBits & item2Bits);
            }else{
                return tensorItem & tensor2Item;
            }
        });
    }

    template <class T>
    Tensor<T>* operator&(const Tensor<T>& tensor, const T& value){

        integral_if_float<T> valueBits = bitcast_if_float(value);
        //integral_if_float<T> valueBits = std::bit_cast<typename integral_if_float<T>::type>(value);

        return forEachAndReturn(tensor, [&valueBits](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                return std::bit_cast<T>(itemBits & valueBits);
            }else{
                return item & valueBits;
            }
        });
    }

    template <class T>
    inline Tensor<T>* operator&(const T& value, const Tensor<T>& tensor){

        integral_if_float<T> valueBits = bitcast_if_float(value);

        return forEachAndReturn(tensor, [&valueBits](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                return std::bit_cast<T>(valueBits & itemBits);
            }else{
                return valueBits & item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator&=(const Tensor<T>& tensor2){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            
            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits & item2Bits);
            }else{
                tensorItem &= tensor2Item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator&=(const T &value)
    {
        integral_if_float<T> valueBits = bitcast_if_float(value);

        forEach([&valueBits](T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                item = std::bit_cast<T>(itemBits & valueBits);
            }else{
                item &= valueBits;
            }
        });
    }
    
    // (^) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator^(const Tensor<T>& tensor2) const{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            
            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                return std::bit_cast<T>(itemBits ^ item2Bits);
            }else{
                return tensorItem ^ tensor2Item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator^=(const Tensor<T>& tensor2){
        
        apply(tensor2, [](T& tensorItem, const T& tensor2Item){
            
            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits ^ item2Bits);
            }else{
                tensorItem ^= tensor2Item;
            }
        });
    }
*/

/*
    // (<<) -------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator<<(const Tensor<T> &tensor2) const{

        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits << item2Bits);
            }else{
                tensorItem << tensor2Item;
            }
        });
    }

    template <class T>
    Tensor<T>* operator<<(const Tensor<T>& tensor, const T& value){

        integral_if_float<T> valueBits = bitcast_if_float(value);
    
        return forEachAndReturn(tensor, [&valueBits](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                return std::bit_cast<T>(itemBits << valueBits);
            }else{
                return item << valueBits;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator<<=(const Tensor<T> &tensor2){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits << item2Bits);
            }else{
                tensorItem <<= tensor2Item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator<<=(const T& value){

        integral_if_float<T> valueBits = bitcast_if_float(value);
    
        forEach([&valueBits](T& item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(item);
                item = std::bit_cast<T>(itemBits << valueBits);
            }else{
                item <<= valueBits;
            }
        });
    }

    // (>>) -------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator>>(const Tensor<T> &tensor2) const{
        
        return applyAndReturn(tensor2, [](const T& tensorItem, const T& tensor2Item){
            
            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits >> item2Bits);
            }else{
                tensorItem >> tensor2Item;
            }
        });
    }

    template <class T>
    void Tensor<T>::operator>>=(const Tensor<T> &tensor2){

        apply(tensor2, [](T& tensorItem, const T& tensor2Item){

            if constexpr(std::is_floating_point<T>::value){
                auto itemBits = std::bit_cast<typename to_integral<T>::type>(tensorItem);
                auto item2Bits = std::bit_cast<typename to_integral<T>::type>(tensor2Item);
                tensorItem = std::bit_cast<T>(itemBits >> item2Bits);
            }else{
                tensorItem >>= tensor2Item;
            }
        });
    }
*/

    // (~) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator~(){

        return forEachAndReturn([](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                return std::bit_cast<T>(~std::bit_cast<typename to_integral<T>::type>(item));
            } else{
                return ~item;
            }
        });
    }

    // (!) --------------------------------------------------------------------------------------------------------------------

    template <class T>
    Tensor<T>* Tensor<T>::operator!()
    {
        
        return forEachAndReturn([](const T& item){

            if constexpr(std::is_floating_point<T>::value){
                return std::bit_cast<T>(!std::bit_cast<typename to_integral<T>::type>(item));
            } else{
                return !item;
            }
        });
    }

    template <class T>
    void Tensor<T>::complementInPlace(){

        forEach([](T& item){
            item = ~item; // TODO: isnt there needed a float specialization?
        });
    }

    template <class T>
    Tensor<T>* Tensor<T>::operator+() const{
        return new Tensor<T>(*this);
    }

    template <class T>
    void Tensor<T>::plusInPlace() const{

    }

    template <class T>
    Tensor<T>* Tensor<T>::operator-() const{

        Tensor<T>* newTensor = new Tensor<T>(*this);
        newTensor->negateInPlace();
        return newTensor;
    }

    template <class T>
    inline void Tensor<T>::negateInPlace(){

        forEach([](T& item){
            item = -item;
        });
    }

    template <class T>
    template <apply_and_return_callable<T> C>
    inline Tensor<T>* Tensor<T>::applyAndReturn(const Tensor<T>& tensor2, C&& operation) const{

        //std::transform(tensor_.begin(), tensor_.end(), tensor2.tensor_.begin(), resultTensor->tensor_.begin(), operation);
        return Tensor<T>::applyAndReturn(*this, tensor2, std::forward<C>(operation));
    }

    template <class T>
    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B, apply_and_return_callable<T> C>
    Tensor<T>* gema::Tensor<T>::applyAndReturn(const A& operand1, const B& operand2, C&& operation) // static
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>){
        
        const Tensor<T>* tensorOperand = type_pick<Tensor<T>>(operand1, operand2);
        Tensor<T>* resultTensor = new Tensor<T>(tensorOperand);

        // TODO: find a way to deal with bool or maybe get rid of it
        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensorOperand->tensor_.size(); ++i){

            if constexpr(std::is_same_v<T, bool>){

                if constexpr (std::is_same_v<A, B>){ // this all just because stupid bool
                    bool op1Item = operand1.tensor_.at(i);
                    bool op2Item = operand2.tensor_.at(i);
                    resultTensor->tensor_.at(i) = operation(op1Item, op2Item);

                }else if constexpr (std::is_same_v<A, T>){
                    bool op2Item = operand2.tensor_.at(i);
                    resultTensor->tensor_.at(i) = operation(operand1, op2Item);

                }else if constexpr (std::is_same_v<B, T>){
                    bool op1Item = operand1.tensor_.at(i);
                    resultTensor->tensor_.at(i) = operation(op1Item, operand2);
                }
            }else{
                if constexpr (std::is_same_v<A, B>){
                    resultTensor->tensor_[i] = operation(operand1.tensor_[i], operand2.tensor_[i]);
                }else if constexpr (std::is_same_v<A, T>){
                    resultTensor->tensor_[i] = operation(operand1, operand2.tensor_[i]);
                }else if constexpr (std::is_same_v<B, T>){
                    resultTensor->tensor_[i] = operation(operand1.tensor_[i], operand2);
                }
            }
        }

        return resultTensor;
    }

    template <class T>
    template <apply_callable<T> C>
    inline void Tensor<T>::apply(const Tensor<T>& tensor2, C&& operation){

        /*for(uint64_t i = 0; i < tensor_.size(); i++){

            if constexpr(std::is_same<T, bool>::value){
                bool tensorItemValue = tensor_.at(i);
                bool tensor2ItemValue = tensor2.tensor_.at(i);
                operation(tensorItemValue, tensor2ItemValue);
            }else{
                operation(tensor_[i], tensor2.tensor_[i]);
            }
        }*/
        Tensor<T>::apply(*this, tensor2, std::forward<C>(operation));
    }

    template <class T>
    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B, apply_callable<T> C> 
    void Tensor<T>::apply(A& operand1, const B& operand2, C&& operation) // static
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>){

        const Tensor<T>* tensorOperand = type_pick<Tensor<T>>(operand1, operand2);

        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensorOperand->tensor_.size(); ++i){

            if constexpr(std::is_same_v<T, bool>){
                if constexpr (std::is_same_v<A, B>){
                    bool op1Item = operand1.tensor_.at(i);
                    bool op2Item = operand2.tensor_.at(i);
                    operation(op1Item, op2Item);
                    operand1.tensor_.at(i) = op1Item;

                }else if constexpr (std::is_same_v<A, T>){
                    bool op2Item = operand2.tensor_.at(i);
                    operation(operand1, op2Item);
                    operand2.tensor_.at(i) = op2Item;

                }else if constexpr (std::is_same_v<B, T>){
                    bool op1Item = operand1.tensor_.at(i);
                    operation(op1Item, operand2);
                    operand1.tensor_.at(i) = op1Item;
                }
            }else{
                if constexpr (std::is_same_v<A, B>){
                    operation(operand1.tensor_[i], operand2.tensor_[i]);
                }else if constexpr (std::is_same_v<A, T>){
                    operation(operand1, operand2.tensor_[i]);
                }else if constexpr (std::is_same_v<B, T>){
                    operation(operand1.tensor_[i], operand2);
                }
            }
        }
        
    }

    template <class T>
    template <foreach_and_return_callable<T> C>
    inline Tensor<T>* Tensor<T>::forEachAndReturn(C&& operation) const{

        return Tensor<T>::forEachAndReturn(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_and_return_callable<T> C>
    Tensor<T>* Tensor<T>::forEachAndReturn(const Tensor<T>& tensor, C&& operation) // static
    {
        Tensor<T>* resultTensor = new Tensor<T>(&tensor);

        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensor.tensor_.size(); ++i){
            resultTensor->tensor_[i] = operation(tensor.tensor_[i]);
        }

        return resultTensor;
    }

    template <class T>
    template <foreach_callable<T> C>
    inline void Tensor<T>::forEach(C&& operation){

        Tensor<T>::forEach(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_callable<T> C>
    void Tensor<T>::forEach(Tensor<T>& tensor, C&& operation){ // static

        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensor.tensor_.size(); ++i){

            if constexpr(std::is_same<T, bool>::value){
                bool value = tensor.tensor_.at(i);
                operation(value);
                tensor.tensor_.at(i) = value;
            }else{
                operation(tensor.tensor_[i]);
            }
        }
        
        // 2.
        //std::transform(tensor.tensor_.begin(), tensor.tensor_.end(), tensor.tensor_.begin(), apply);
    }

    template <class T>
    Tensor<T>::~Tensor(){
        //
    }



    // private methods:

    template <class T>
    std::vector<uint64_t> Tensor<T>::getCoords(int itemIndex) const{

        std::vector<uint64_t> coordinates;
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
    int Tensor<T>::getIndex(const std::vector<uint64_t>& coordinates) const{
        
        int itemIndex = 0;
        int dimensionProduct = 1;

       for(uint64_t i = dimensionSizes_.size() - 1; i < dimensionSizes_.size(); --i){

            itemIndex += coordinates[i] * dimensionProduct;
            dimensionProduct *= dimensionSizes_[i];
       }

        return itemIndex;
    }

    template <class T>
    std::vector<uint64_t> Tensor<T>::littleGetCoords(int itemIndex) const{

        std::vector<uint64_t> coordinates;
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
    int Tensor<T>::littleGetIndex(const std::vector<uint64_t>& coordinates) const{
        
        int itemIndex = 0;
        int dimensionProduct = 1;

       for(uint64_t i = 0; i < dimensionSizes_.size(); ++i){

            itemIndex += coordinates[i] * dimensionProduct;
            dimensionProduct *= dimensionSizes_[i];
       }

        return itemIndex;
    }

    template <class T>
    int Tensor<T>::calculateNumberOfItems(const std::vector<uint64_t>& dimensionSizes) const{

        return std::accumulate(dimensionSizes_.begin(), dimensionSizes_.end(), 1, std::multiplies<int>());
    }

    template <class T>
    inline bool Tensor<T>::compareItems(const T& a, const T& b) const requires(!std::is_floating_point<T>::value){
        
        return a == b;
    }

    template <class T>
    inline bool Tensor<T>::compareItems(const T a, const T b) const requires(std::is_floating_point<T>::value){
        
        T veightedEpsilon = std::numeric_limits<T>::epsilon();
        return std::fabs(a - b) < veightedEpsilon;
    }

    template <class T>
    void Tensor<T>::defaultFunctions(){

        //equals_ = &defaultEquals_;

        /*if constexpr (std::is_floating_point<T>::value){

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
        }*/
    }
}
