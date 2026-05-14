#ifndef ABSTRACT_OPERATION_HPP
#define ABSTRACT_OPERATION_HPP

namespace gema {

    
//template<template <typename, typename> class Derived, typename T, typename IMemoryBackend>
template<typename Derived>
class AbstractOperation {

    template<typename U> using T = typename U::value_type;
    template<typename U> using IMemoryBackend = typename U::memory_backend;

    public:

    #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    /**/\
        template<typename D = Derived>\
        friend auto operator OP_SYMBOL(const Derived& tensor1, const Derived& tensor2)\
        requires requires (T<D> a, T<D> b) {a OP_SYMBOL b;}{\
    /**/\
            return tensor1.applyAndReturn(tensor1, tensor2, [](const T<D>& tensorItem, const T<D>& tensor2Item){\
                    return tensorItem OP_SYMBOL tensor2Item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    /**/\
        template<typename D = Derived>\
        inline friend auto operator OP_SYMBOL(const Derived& tensor, const T<D>& value)\
        requires requires (T<D> a, T<D> b) {a OP_SYMBOL b;}{\
    /**/\
            return Derived::forEachAndReturn(tensor, [value](const T<D>& item){\
                return item OP_SYMBOL value;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    /**/\
        template<typename D = Derived>\
        inline friend auto operator OP_SYMBOL(const T<D>& value, const Derived& tensor)\
        requires requires (T<D> a, T<D> b) {a OP_SYMBOL b;}{\
    /**/\
            /* Do not delegate switched argument operator! While on numbers set the operation would be often commutative, */\
            /* it is not guaranteed to be so on every type and operation!*/\
            return Derived::forEachAndReturn(tensor, [value](const T<D>& item){\
                return value OP_SYMBOL item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    /**/\
        template<typename D = Derived>\
        friend void operator OP_SYMBOL##=(Derived& tensor1, const Derived& tensor2)\
        requires requires (T<D> a, T<D> b) {a OP_SYMBOL##= b;}{\
    /**/\
            tensor1.apply(tensor2, [](T<D>& tensorItem, const T<D>& tensor2Item){\
                tensorItem OP_SYMBOL##= tensor2Item;\
            });\
        }\
    /**/

    #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    /**/\
        template<typename D = Derived>\
        friend void operator OP_SYMBOL##=(Derived& tensor, const T<D>& value)\
        requires requires (T<D> a, T<D> b) {a OP_SYMBOL##= b;}{\
    /**/\
            tensor.forEach([value](T<D>& item){\
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



    #define UNARY_OPERATION(OP_SYMBOL)\
        template<typename D = Derived>\
        auto operator OP_SYMBOL() const\
        requires requires (T<D> a) {OP_SYMBOL a;}{\
    /**/\
            return self().forEachAndReturn([](const T<D>& item){\
                return OP_SYMBOL item;\
            });\
        }\
    /**/

    UNARY_OPERATION(~)
    UNARY_OPERATION(!)
    UNARY_OPERATION(+)
    UNARY_OPERATION(-)

    #undef UNARY_OPERATION



    #define UNARY_OPERATION_INPLACE(OP_SYMBOL, OP_NAME)\
        template<typename D = Derived>\
        void OP_NAME##InPlace(){\
            self().forEach([](T<D>& item){\
                item = OP_SYMBOL item;\
            });\
        }

    UNARY_OPERATION_INPLACE(~, complement)
    UNARY_OPERATION_INPLACE(+, plus)
    UNARY_OPERATION_INPLACE(-, opposite)

    #undef UNARY_OPERATION_INPLACE



    #define PREFIX_POSTFIX(OP_SYMBOL)\
        template<typename D = Derived>\
        Derived& operator OP_SYMBOL(){\
    /**/\
            self().forEach([](T<D>& item){\
                OP_SYMBOL item;\
            });\
    /**/\
            return *this;\
        }\
    /**/\
        template<typename D = Derived>\
        Derived operator OP_SYMBOL(int) const{\
            Derived temporary(*this);\
            operator++();\
            return temporary;\
        }\
    /**/

    PREFIX_POSTFIX(++)
    PREFIX_POSTFIX(--)

    #undef PREFIX_POSTFIX




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

    const Derived& self() const {
        return static_cast<const Derived&>(*this);
    }

};


}


#endif