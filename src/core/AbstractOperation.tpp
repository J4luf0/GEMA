#include "AbstractOperation.hpp"

namespace gema {

    // #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    // /**/\
    //     template<class Derived, typename T>\
    //     auto AbstractOperation<Derived, T>::operator OP_SYMBOL(const Derived& tensor2) const\
    //     requires requires (T a, T b) {a OP_SYMBOL b;}{\
    // /**/\
    //         return self().applyAndReturn(*this, tensor2, [](const T& tensorItem, const T& tensor2Item){\
    //                 return tensorItem OP_SYMBOL tensor2Item;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    // /**/\
    //     template<class Derived, typename T>\
    //     inline auto operator OP_SYMBOL(const Derived& tensor, const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL b;}{\
    // /**/\
    //         return Derived::forEachAndReturn(tensor, [&value](const T& item){\
    //             return item OP_SYMBOL value;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    // /**/\
    //     template<class Derived, typename T>\
    //     inline auto operator OP_SYMBOL(const T& value, const Derived& tensor)\
    //     requires requires (T a, T b) {a OP_SYMBOL b;}{\
    // /**/\
    //         /* Do not delegate switched argument operator! While on numbers set the operation would be often commutative, */\
    //         /* it is not guaranteed to be so on every type and operation!*/\
    //         return Derived::forEachAndReturn(tensor, [&value](const T& item){\
    //             return value OP_SYMBOL item;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    // /**/\
    //     template<class Derived, typename T>\
    //     void AbstractOperation<Derived, T>::operator OP_SYMBOL##=(const Derived& tensor2)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;}{\
    // /**/\
    //         apply(tensor2, [](T& tensorItem, const T& tensor2Item){\
    //             tensorItem OP_SYMBOL##= tensor2Item;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    // /**/\
    //     template<class Derived, typename T>\
    //     void AbstractOperation<Derived, T>::operator OP_SYMBOL##=(const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;}{\
    // /**/\
    //         self().forEach([&value](T& item){\
    //             item OP_SYMBOL##= value;\
    //         });\
    //     }\
    // /**/

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

    // // Some logical overloads for binary operations are not making sense for logical operators
    // #define LOGICAL_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    // LOGICAL_BINARY(&&)
    // LOGICAL_BINARY(||)

    // // Some bitwise overloads for binary operations are not making sense for bitshift
    // #define BITSHIFTLIKE(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // BITSHIFTLIKE(<<)
    // BITSHIFTLIKE(>>)

    // #undef ARITHMETIC_BINARY_ToTrT // Macros no longer needed
    // #undef ARITHMETIC_BINARY_ToVrT
    // #undef ARITHMETIC_BINARY_VoTrT
    // #undef ARITHMETIC_BINARY_ToeT
    // #undef ARITHMETIC_BINARY_ToeV

    // #undef BITSHIFTLIKE
    // #undef LOGICAL_BINARY

    // #undef ARITHMETIC_BINARY

}
