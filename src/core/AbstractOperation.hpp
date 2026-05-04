#ifndef ABSTRACT_OPERATION_HPP
#define ABSTRACT_OPERATION_HPP

namespace gema {

    
template<template <typename> class Derived, typename T>
class AbstractOperation {

    public:

    #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    /**/\
        friend auto operator OP_SYMBOL(const Derived<T>& tensor1, const Derived<T>& tensor2)\
        requires requires (T a, T b) {a OP_SYMBOL b;}{\
    /**/\
            return tensor1.applyAndReturn(tensor1, tensor2, [](const T& tensorItem, const T& tensor2Item){\
                    return tensorItem OP_SYMBOL tensor2Item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    /**/\
        inline friend auto operator OP_SYMBOL(const Derived<T>& tensor, const T& value)\
        requires requires (T a, T b) {a OP_SYMBOL b;}{\
    /**/\
            return Derived<T>::forEachAndReturn(tensor, [&value](const T& item){\
                return item OP_SYMBOL value;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    /**/\
        inline friend auto operator OP_SYMBOL(const T& value, const Derived<T>& tensor)\
        requires requires (T a, T b) {a OP_SYMBOL b;}{\
    /**/\
            /* Do not delegate switched argument operator! While on numbers set the operation would be often commutative, */\
            /* it is not guaranteed to be so on every type and operation!*/\
            return Derived<T>::forEachAndReturn(tensor, [&value](const T& item){\
                return value OP_SYMBOL item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    /**/\
        friend void operator OP_SYMBOL##=(Derived<T>& tensor1, const Derived<T>& tensor2)\
        requires requires (T a, T b) {a OP_SYMBOL##= b;}{\
    /**/\
            tensor1.apply(tensor2, [](T& tensorItem, const T& tensor2Item){\
                tensorItem OP_SYMBOL##= tensor2Item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    /**/\
        friend void operator OP_SYMBOL##=(Derived<T>& tensor, const T& value)\
        requires requires (T a, T b) {a OP_SYMBOL##= b;}{\
    /**/\
            tensor.forEach([&value](T& item){\
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
    ARITHMETIC_BINARY(|)
    ARITHMETIC_BINARY(&)
    ARITHMETIC_BINARY(^)
    ARITHMETIC_BINARY(%)

    // Some logical overloads for binary operations are not making sense for logical operators
    #define LOGICAL_BINARY(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    LOGICAL_BINARY(&&)
    LOGICAL_BINARY(||)

    // Some bitwise overloads for binary operations are not making sense for bitshift
    #define BITSHIFTLIKE(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    BITSHIFTLIKE(<<)
    BITSHIFTLIKE(>>)

    #undef ARITHMETIC_BINARY_ToTrT // Macros no longer needed
    #undef ARITHMETIC_BINARY_ToVrT
    #undef ARITHMETIC_BINARY_VoTrT
    #undef ARITHMETIC_BINARY_ToeT
    #undef ARITHMETIC_BINARY_ToeV

    #undef BITSHIFTLIKE
    #undef LOGICAL_BINARY

    #undef ARITHMETIC_BINARY



    // #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     template <typename Other> requires (!std::same_as<Other, Derived<T>>)\
    //     friend auto operator OP_SYMBOL(const Other& tensor1, const Other& tensor2) = delete;

    // #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     template <typename Other> requires (!std::same_as<Other, Derived<T>>)\
    //     inline friend auto operator OP_SYMBOL(const Other& tensor, const T& value) = delete;

    // #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    //     template <typename Other> requires (!std::same_as<Other, Derived<T>>)\
    //     inline friend auto operator OP_SYMBOL(const T& value, const Other& tensor) = delete;

    // #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     template <typename Other> requires (!std::same_as<Other, Derived<T>>)\
    //     friend void operator OP_SYMBOL##=(Other& tensor1, const Other& tensor2) = delete;

    // #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    //     template <typename Other> requires (!std::same_as<Other, Derived<T>>)\
    //     friend void operator OP_SYMBOL##=(Other& tensor, const T& value) = delete;

    // #define ARITHMETIC_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // ARITHMETIC_BINARY(+)
    // ARITHMETIC_BINARY(-)
    // ARITHMETIC_BINARY(*)
    // ARITHMETIC_BINARY(/)
    // ARITHMETIC_BINARY(|)
    // ARITHMETIC_BINARY(&)
    // ARITHMETIC_BINARY(^)
    // ARITHMETIC_BINARY(%)

    // #define LOGICAL_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    // LOGICAL_BINARY(&&)
    // LOGICAL_BINARY(||)

    // #define BITSHIFTLIKE(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // BITSHIFTLIKE(<<)
    // BITSHIFTLIKE(>>)

    // #undef ARITHMETIC_BINARY_ToTrT
    // #undef ARITHMETIC_BINARY_ToVrT
    // #undef ARITHMETIC_BINARY_VoTrT
    // #undef ARITHMETIC_BINARY_ToeT
    // #undef ARITHMETIC_BINARY_ToeV

    // #undef BITSHIFTLIKE
    // #undef LOGICAL_BINARY

    // #undef ARITHMETIC_BINARY






    // #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     inline auto operator OP_SYMBOL(const Derived& tensor2) const\
    //     requires requires (T a, T b) {a OP_SYMBOL b;};

    // #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     inline friend auto operator OP_SYMBOL(const Derived& tensor, const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL b;};

    // #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    //     inline friend auto operator OP_SYMBOL(const T& value, const Derived& tensor)\
    //     requires requires (T a, T b) {a OP_SYMBOL b;};

    // #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     void operator OP_SYMBOL##=(const Derived& tensor2)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;};

    // #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    //     void operator OP_SYMBOL##=(const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;};

    // #define ARITHMETIC_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // ARITHMETIC_BINARY(+)
    // ARITHMETIC_BINARY(-)
    // ARITHMETIC_BINARY(*)
    // ARITHMETIC_BINARY(/)
    // ARITHMETIC_BINARY(|)
    // ARITHMETIC_BINARY(&)
    // ARITHMETIC_BINARY(^)
    // ARITHMETIC_BINARY(%)

    // #define LOGICAL_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    // LOGICAL_BINARY(&&)
    // LOGICAL_BINARY(||)

    // #define BITSHIFTLIKE(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // BITSHIFTLIKE(<<)
    // BITSHIFTLIKE(>>)

    // #undef ARITHMETIC_BINARY_ToTrT
    // #undef ARITHMETIC_BINARY_ToVrT
    // #undef ARITHMETIC_BINARY_VoTrT
    // #undef ARITHMETIC_BINARY_ToeT
    // #undef ARITHMETIC_BINARY_ToeV

    // #undef BITSHIFTLIKE
    // #undef LOGICAL_BINARY

    // #undef ARITHMETIC_BINARY

    protected:

    const Derived<T>& self() const {
        return static_cast<const Derived<T>&>(*this);
    }

};


}


#endif