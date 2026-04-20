#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Tensor.hpp"

namespace gema{

template<class T, TensorType<T> ITensor = Tensor<T>>
class Matrix{

    private:

    ITensor<T> tensor_;

    public:

    Matrix(const int64_t x, const int64_t y);

    Matrix(const int64_t x, const int64_t y, const LinearContainer<T>& newMatrixData);

    Matrix(const Matrix<T>& otherMatrix);

    Matrix(Matrix<T>&& otherMatrix);

    Matrix();



    const std::vector<uint64_t>& getDimensionSizes() const;

    uint64_t getNumberOfDimensions() const;

    uint64_t getNumberOfItems() const;

    T& getItem(const std::vector<uint64_t>& coordinates);

    void setItem(const T& value, const std::vector<uint64_t>& coordinates);

    LinearContainer<T>& getData();

    Tensor<T>& setData(const LinearContainer<T>& tensorItems);

    std::string toString() const;

    template<typename U> friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

    void fillWith(const T& fill);

    Tensor<T> transposition(const int dim1 = 0, const int dim2 = 1) const;

    void resize(const uint64_t dim1, const uint64_t dim2);

    Tensor<T>& operator=(const Tensor<T>& tensor2);

    bool operator==(const Tensor<T>& tensor2) const;

    bool operator!=(const Tensor<T>& tensor2) const;


    #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        inline auto operator OP_SYMBOL(const Tensor<T>& tensor2) const\
        requires requires (T a, T b) {a OP_SYMBOL b;};

    #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        template<typename U> friend inline auto operator OP_SYMBOL(const Tensor<U>& tensor, const U& value)\
        requires requires (U a, U b) {a OP_SYMBOL b;};

    #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
        template<typename U> friend inline auto operator OP_SYMBOL(const U& value, const Tensor<U>& tensor)\
        requires requires (U a, U b) {a OP_SYMBOL b;};

    #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
        void operator OP_SYMBOL##=(const Tensor<T>& tensor2)\
        requires requires (T a, T b) {a OP_SYMBOL##= b;};

    #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
        void operator OP_SYMBOL##=(const T& value)\
        requires requires (T a, T b) {a OP_SYMBOL##= b;};

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

    #define LOGICAL_BINARY(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    LOGICAL_BINARY(&&)
    LOGICAL_BINARY(||)

    #define BITSHIFTLIKE(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
        ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    BITSHIFTLIKE(<<)
    BITSHIFTLIKE(>>)

    #undef ARITHMETIC_BINARY_ToTrT
    #undef ARITHMETIC_BINARY_ToVrT
    #undef ARITHMETIC_BINARY_VoTrT
    #undef ARITHMETIC_BINARY_ToeT
    #undef ARITHMETIC_BINARY_ToeV

    #undef BITSHIFTLIKE
    #undef LOGICAL_BINARY

    #undef ARITHMETIC_BINARY

    Matrix<T>& inverse() const;

    void inverseInPlace();

    void matrixMultiplication(const Matrix& otherMatrix);
    
    virtual ~Matrix();



    private:

    std::vector<uint64_t> getCoords(uint64_t itemIndex) const;

    uint64_t getIndex(const std::vector<uint64_t>& coordinates) const;

};

}

#include "Matrix.tpp"

#endif