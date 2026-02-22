#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "LinearContainer.hpp"

namespace gema{

// Forward declaration just for the concepts.
template<class T>
class Tensor;

// Avoids std::vector<bool> specialization, that would be otherwise problematic
template<class T>
struct tensor_storage_type{
    using type = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
};
    
// Concept that checks if type X is of type T or Tensor<T>. Useful for operator overloads.
template <typename X, class T>
concept is_tensor_or_t = std::is_same_v<X, T> || std::is_same_v<X, Tensor<T>>;

/// Checks for void(T&, const T&) signature.
template <typename C, class T>
concept apply_callable = std::is_invocable_r_v<void, C, T&, const T&>;

/// Checks for T(const T&, const T&) signature.
template <typename C, class T>
//concept apply_and_return_callable = std::is_invocable_r_v<T, C, const T&, const T&>;
concept apply_and_return_callable = std::is_invocable_v<C, const T&, const T&>;

/// Checks for void(T&) signature.
template <typename C, class T>
concept foreach_callable = std::is_invocable_r_v<void, C, T&>;

// Checks for T(const T&) signature.
template <typename C, class T>
//concept foreach_and_return_callable = std::is_invocable_r_v<T, C, const T&>;
concept foreach_and_return_callable = std::is_invocable_v<C, const T&>;

// Checks for bool(const T&, const T&) signature.
template <typename C, class T>
concept equals_callable = std::is_invocable_r_v<bool, C, const T&, const T&>;

// Checks for int(const T&, const T&) signature.
template <typename C, class T>
concept order_callable = std::is_invocable_r_v<int, C, const T&, const T&>;

// Signature of equals function, returning equalness of arguments represented by bool.
template <class T> using EqualsCallable = bool(const T&, const T&);

// Signature of order fuction, returning order of arguments represented as int.
template <class T> using OrderCallable = int(const T&, const T&);

// ============================================================================================================================
/**
 * @brief Class representing generic tensor.
 * 
 * @par
 * This class is a dynamic N-dimensional array also known as tensor. At construction user specifies by vector argument how many
 * dimensions the tensor will have - size of that vector argument, and size of each dimension - defined dimension by dimension
 * as whole positive number in the vector argument. User uses coordinates to work with items inside tensor. Coordinates is 
 * a vector of same size as number of dimensions and number from 0 to highest allowed index in corresponding dimension 
 * (dimension size - 1). For simple operations, most important user methods are Tensor<T>::getItem and Tensor<T>::setItem.
 * 
 * @par
 * Inner workings - algorithms
 * @par
 * Tensor is represented inside class as one dimensional vector with two most important methods: Tensor<T>::getCoords that
 * calculates made-up coordinates of the tensor as if it actually was N-dimensional array, and inverse method
 * Tensor<t>::getIndex that returns actual vector index when provided with coordinates. These methods are hidden implementation
 * and thus, the tensor can act on the outside like actual tensor with N dimensions.
 * 
 * @par
 * Inner workings - data
 * @par
 * To achieve this, the class stores two main attributes: 1. The tensor data vector itself, that is accessed by index that can
 * be obtained from coordinates using Tensor<t>::getIndex method. 2. Vector storing dimension sizes that defines tensors number
 * of dimensions by its length and size of each dimension by every item integer value in the vector.
 * 
 * @par
 * Mathematical conventions
 * @par
 * By default, the tensor uses something that could be called "big endian" representation. That means the change of first 
 * coordinate will shift the resulting index by largest amount and the last coordinate should only shift the index by one.
 * 
 * @par
 * Terminology
 * @par
 * Item of tensor is unit of data of generic type. Its index is index in tensors physical vector representation. Coordinates
 * are how tensor communicates with the outside and to access item in a tensor, they must be converted to index using 
 * Tensor<t>::getIndex.
 * 
 * @par
 * Safety
 * @par
 * For speed purposes, this class doesn´t implement any checking against bad input. A wrapper class will be implemented for
 * safe use. Working directly with this class might be dangerous if the user is not sure about validity of the data.
 * 
 * @tparam Type of data that is stored in the tensor.
 * 
 * @warning Even though the bool is supported, it is advised to use char or 8 bit int instead, unless user is looking to take 
 * advantage of std::vector bit bool storing for effectivity in memory (might be less optimized for methods that iterate like
 * forEach(), etc...) and thus slower.
 */
template<class T> 
class Tensor {

    private:

    /// The tensor data itself, represented by vector containing all the items.
    LinearContainer<T> tensor_;
    //std::vector<typename tensor_storage_type<T>::type> tensor_;
    /// Size od every tensor dimension.
    std::vector<uint64_t> dimensionSizes_;

    /// Vector one to one with dimensionSizes_ where value on [n] tells how big jump corresponds to one increment of n-th
    /// dimension on flattened data. Used for optimization, shall not be leaked outside.
    std::vector<uint64_t> dimensionJumps_;
    std::map<std::vector<uint64_t>, uint64_t> recentAccessCache_; // Maybe make it its own helper class


    /// Function compares items in tensor and represents equality by bool.
    static std::function<EqualsCallable<T>> defaultEquals_;
    std::function<EqualsCallable<T>>* equals_ = &defaultEquals_;
    std::function<EqualsCallable<T>> userEquals_;

    /// Function orders items in a way: (less, equal, more) -> (-1, 0, 1).
    static std::function<OrderCallable<T>> defaultOrder_;
    std::function<OrderCallable<T>>* order_ = &defaultOrder_;
    std::function<OrderCallable<T>> userOrder_;

    std::function<void(const T&)> tensorOutput_;
    std::function<void(const T&, const std::vector<uint64_t>&)> itemOutput_;

    public:

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets dimensionSizes, calculates number of items and then allocates them on tensor, then sets functional
     * attributes values yet the default lambda itself is decided at compile time. The result is empty tensor, with defined 
     * dimensions and allocated space.
     *
     * @param newTensorDimensionSizes Vector filled with sizes of dimensions.
    */
    Tensor(const std::vector<uint64_t>& newTensorDimensionSizes);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets dimensionSizes, then fills the tensor with given data. Following safety rules of this class, this function
     * does not check for correct size of the data and will forcibly make tensor with given dimension sizes, whether it means
     * to discard data or fill rest of tensor with default values.
     * 
     * @param newTensorDimensionSizes Vector filled with sizes of dimensions.
     * @param tensorItems one dimensional vector of items to be added by order.
     */
    Tensor(const std::vector<uint64_t>& newTensorDimensionSizes, const LinearContainer<T>& newTensorData);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Copy constructor, makes the object the same as the parameter object.
     * 
     * @param otherTensor Tensor to be copied.
     */
    Tensor(const Tensor<T>& otherTensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Move constructor, makes the object the same as the parameter object.
     * 
     * @param otherTensor Tensor to be moved.
     */
    Tensor(Tensor<T>&& otherTensor) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Similar use to copy constructor but it does not copy values. Instead it creates tensor of the same dimension size
     * and item count as the one in parameter. Useful when copy of values is not important but performance is.
     * 
     * @param otherTensor Tensor whose dimension sizes and item count is copied.
     */
    Tensor(const Tensor<T>* otherTensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Empty constructor so it can be declared without being initalized - trying to do something with
     * uninitialized tensor is sure undefined behavior, not recommended.
     * 
     * @warning Not recommended to use.
     */
    Tensor();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Gets all dimension sizes.
     * 
     * @return Vector containing one int per dimension with value of its size.
    */
    const std::vector<uint64_t>& getDimensionSizes() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Gets the number of dimensions of a tensor.
     * 
     * @return A number of dimensions.
    */
    uint64_t getNumberOfDimensions() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Gets the number of items in a tensor.
     * 
     * @return Number of items in a tensor.
     */
    uint64_t getNumberOfItems() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Gets item on provided coordinates, returns the item by value because large objects should have been already 
     * represented by pointer or reference.
     * 
     * @param coordinates vector of coordinates specifying the item to be returned.
     * 
     * @return Item on the provided coordinates.
    */
    T& getItem(const std::vector<uint64_t>& coordinates);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets one value into tensor onto the desired coordinates.
     *  
     * @param value a value of generic type that will be stored in the tensor.
     * @param coordinates a vector of coordinates to place the value to.
    */
    void setItem(const T& value, const std::vector<uint64_t>& coordinates);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Exposes tensor data as reference to std::vector. Modifying its elements will directly modify elements in tensor.
     * 
     * @return Reference to data of the tensor.
     * 
     * @note Exposes the inner implementation of tensor (the flattened data), use carefully.
     */
    LinearContainer<T>& getData();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets one dimensional array and puts its items into tensor by order, if the array is longer than number of items 
     * in a tensor, only those that fit will be added.
     * 
     * @param tensorItems one dimensional vector of items to be added by order.
     * 
     * @note Risky and kinda shows the inner implementation by dodging the coordinate to index calculation, but its much faster
     * and can be beneficial if user knows what it is doing and needs to put many values in a tensor at once.
    */
    Tensor<T>& setData(const LinearContainer<T>& tensorItems);

    // void setEquals(const std::function<EqualsCallable<T>>& equals);

    // void setOrder(const std::function<OrderCallable<T>>& order);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets the output of the tensor through this->showTensor() method.
     * 
     * @param tensorOutput function that defines the output of the tensor.
    */
    void setTensorOutput(const std::function<void(const T&)>& tensorOutput);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets the output of the items through this->showItem() method.
     * 
     * @param tensorOutput function that defines the output of the tensor.
    */
    void setItemOutput(const std::function<void(const T&, const std::vector<uint64_t>&)>& itemOutput);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Calculates, if all tensor dimensions have the same size.
     * 
     * @return Bool @b true if the tensor is equilateral and @b false if not.
    */
    bool isEquilateral() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Generates a string from tensor items in form of parsable curly bracket hierarchy. Inverse to "parse".
     * 
     * @return A parsable string representing the tensor.
     */
    std::string toString() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Puts user readable tensor output (from Tensor<T>::toString) into stream.
     * 
     * @param os output stream.
     * @param tansor tensor to output.
     * 
     * @return Output stream.
     */
    template<typename U> friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Parses std::string specifying the tensor dimension sizes and values. Inverse to "toString".
     * 
     * @param tensorString string in correct format to be parsed.
     * @param parseItem function to convert string to item.
     */
    void parse(const std::string& tensorString, const std::function<const T(const std::string&)>& parseItem);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Fills tensor with passed value.
     * 
     * @param fill the value to be filled into all items in tensor.
    */
    void fillWith(const T& fill);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Swaps two dimensions in a tensor.
     * 
     * @param dim1 first dimension to swap, default value is 0.
     * @param dim2 second dimension to swap, default value is 1.
     * 
     * @return A pointer to new allocated tensor, that got two dimensions transposed.
    */
    Tensor<T> transposition(const int dim1 = 0, const int dim2 = 1) const;


    
    // OPERATOR OVERLOADS -----------------------------------------------------------------------------------------------------

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Copies one tensor to another by value.
     * 
     * @param tensor2 tensor which values are copied into this tensor.
     * 
     * @return Reference to this tensor after the copying.
     */
    Tensor<T>& operator=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Compares two tensors, checks if all items are equal and if the dimension sizes are equal.
     * 
     * @param tensor2 a second tensor to be compared by value.
     * 
     * @return Boolean @b true if the tensors are the same and @b false in not.
     */
    bool operator==(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Compares two tensors, checks if all items are equal and if the dimension sizes are equal. Then returns bool
     * negation of the result.
     * 
     * @param tensor2 a second tensor to be compared by value.
     * 
     * @return Boolean @b false if the tensors are the same and @b true in not.
     */
    bool operator!=(const Tensor<T>& tensor2) const;



    // ARITHMETIC OPERATOR OVERLOADS ------------------------------------------------------------------------------------------

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds tensor with tensor item by item and returns result as new tensor. Does no size checking.
     * 
     * @param tensor2 addend tensor.
     * 
     * @return Pointer to new resulting tensor.
    */
    inline auto operator+(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a + b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds every tensor item to value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor addend tensor.
     * @param value addend value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend inline auto operator+(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a + b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds value to every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value addend value.
     * @param tensor addend tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend inline auto operator+(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a + b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds tensor to tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 addend tensor.
    */
    void operator+=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a += b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds value to tensor item by item in place. Does no size checking.
     * 
     * @param value addend value.
    */
    void operator+=(const T& value)
    requires requires (T a, T b) {a += b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts tensor with tensor item by item and returns result as new tensor. Does no size checking.
     * 
     * @param tensor2 substrahend tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    //template <class U> requires requires (U a, U b) { a - b; }
    auto operator-(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a - b;};
    //-> Tensor<decltype(std::declval<U>() - std::declval<U>())>;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts value from every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param tensor minuend tensor.
     * @param value substrahend value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator-(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a - b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts every tensor item from value and returns result as new tensor. Does no size checking.
     * 
     * @param value minuend value.
     * @param tensor substrahend tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator-(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a - b;};
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts tensor from a tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 substrahend tensor.
     */
    void operator-=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a -= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts value from tensor item by item in place. Does no size checking.
     * 
     * @param value substrahend value.
    */
    void operator-=(const T& value)
    requires requires (T a, T b) {a -= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies tensor with tensor item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 multiplier tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator*(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a * b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor multiplicand tensor.
     * @param value multiplier value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator*(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a * b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param value multiplicand value.
     * @param tensor multiplier tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator*(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a * b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies tensor from a tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 product tensor.
     */
    void operator*=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a *= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies tensor item by item by value in place. Does no size checking.
     * 
     * @param value multiplier value.
    */
    void operator*=(const T& value)
    requires requires (T a, T b) {a *= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides tensor with tensor item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator/(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a / b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor dividend tensor.
     * @param value divisor value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator/(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a / b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies value by every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value dividend value.
     * @param tensor divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator/(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a / b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides tensor from a tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 divisor tensor.
     */
    void operator/=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a /= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides tensor item by item by value in place. Does no size checking.
     * 
     * @param value divisor value.
    */
    void operator/=(const T& value)
    requires requires (T a, T b) {a /= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator%(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor dividend tensor.
     * @param value divisor value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator%(const Tensor<U>& tensor, const U& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo value by every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value dividend value.
     * @param tensor divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator%(const U& value, const Tensor<U>& tensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo item by item in place. Does no size checking.
     * 
     * @param tensor2 divisor tensor.
     */
    void operator%=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo every tensor item by value in place. Does no size checking.
     * 
     * @param value divisor value.
    */
    void operator%=(const T& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical "and" item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 right-hand size operand.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator&&(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a && b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical "and" with every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor left-hand side operand.
     * @param value right-hand size operand.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator&&(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a && b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical "and" with value by every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value left-hand side operand.
     * @param tensor right-hand size operand.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator&&(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a && b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical "or" item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 right-hand size operand.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator||(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a || b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical "or" with every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor left-hand side operand.
     * @param value right-hand size operand.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator||(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a || b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical "or" with value by every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value left-hand side operand.
     * @param tensor right-hand size operand.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator||(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a || b;};



    // BITWISE OPERATOR OVERLOADS ---------------------------------------------------------------------------------------------

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "or" on each item in a tensor and returns result as new tensor.
     * 
     * @param tensor2 second tensor to perform operation against.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator|(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a | b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "or" between every tensor item and value and returns result as new tensor. 
     * Does no size checking.
     * 
     * @param tensor operand tensor.
     * @param value operand value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator|(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a | b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "or" between value and every tensor item and returns result as new tensor. 
     * Does no size checking.
     * 
     * @param value operand value.
     * @param tensor operand tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend inline auto operator|(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a | b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "or" on each item in a tensor in place.
     * 
     * @param tensor2 second tensor to perform operation against.
     */
    void operator|=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a |= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "or" between each item in a tensor and value in place. Does no size checking.
     * 
     * @param value operand value.
    */
    void operator|=(const T& value)
    requires requires (T a, T b) {a |= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" on each item in a tensor and returns result as new tensor.
     * 
     * @param tensor2 second tensor to perform operation against.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator&(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a & b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" between every tensor item and value and returns result as new tensor. 
     * Does no size checking.
     * 
     * @param tensor operand tensor.
     * @param value operand value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator&(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a & b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" between value and every tensor item and returns result as new tensor. 
     * Does no size checking.
     * 
     * @param value operand value.
     * @param tensor operand tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator&(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a & b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" on each item in a tensor in place.
     * 
     * @param tensor2 second tensor to perform operation against.
     */
    void operator&=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a &= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" between each item in a tensor and value in place. Does no size checking.
     * 
     * @param value operand value.
    */
    void operator&=(const T& value)
    requires requires (T a, T b) {a &= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" on each item in a tensor and returns result as new tensor.
     * 
     * @param tensor2 second tensor to perform operation against.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator^(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a ^ b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" between every tensor item and value and returns result as new tensor. 
     * Does no size checking.
     * 
     * @param tensor operand tensor.
     * @param value operand value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator^(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a ^ b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" between value and every tensor item and returns result as new tensor. 
     * Does no size checking.
     * 
     * @param value operand value.
     * @param tensor operand tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator^(const U& value, const Tensor<U>& tensor)
    requires requires (U a, U b) {a ^ b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" on each item in a tensor in place.
     * 
     * @param tensor2 second tensor to perform operation against.
     */
    void operator^=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a ^= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" between each item in a tensor and value in place. Does no size checking.
     * 
     * @param value operand value.
    */
    void operator^=(const T& value)
    requires requires (T a, T b) {a ^= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs left bit shift as this tensors item being shifted by amount in second tensors item and returns result 
     * as new tensor. Does not size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator<<(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a << b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs left bit shift between every tensor item by given value and returns result as new tensor. Does no size 
     * checking.
     * 
     * @param tensor operand tensor.
     * @param value operand value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator<<(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a << b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs left bit shift as this tensors item being shifted by amount in second tensors item in place. Does not 
     * size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     */
    void operator<<=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a <<= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs left bit shift on each item in a tensor by given value in place. Does no size checking.
     * 
     * @param value operand value.
    */
    void operator<<=(const T& value)
    requires requires (T a, T b) {a <<= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs right bit shift as this tensors item being shifted by amount in second tensors item and returns result 
     * as new tensor. Does not size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     * 
     * @return Pointer to new resulting tensor.
     */
    auto operator>>(const Tensor<T>& tensor2) const
    requires requires (T a, T b) {a >> b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs right bit shift between every tensor item by given value and returns result as new tensor. Does no size 
     * checking.
     * 
     * @param tensor operand tensor.
     * @param value operand value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend auto operator>>(const Tensor<U>& tensor, const U& value)
    requires requires (U a, U b) {a >> b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs right bit shift as this tensors item being shifted by amount in second tensors item in place. Does not 
     * size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     */
    void operator>>=(const Tensor<T>& tensor2)
    requires requires (T a, T b) {a >>= b;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs right bit shift on each item in a tensor by given value in place. Does no size checking.
     * 
     * @param value operand value.
    */
    void operator>>=(const T& value)
    requires requires (T a, T b) {a >>= b;};



    // UNARY OPERATOR OVERLOADS -----------------------------------------------------------------------------------------------
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise negation on each item and returns result as new tensor. Does not change original tensor.
     * 
     * @return Resulting tensor.
     */
    auto operator~() const
    requires requires (T a) {~a;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs logical negation on each item and returns result as new tensor. Does not change original tensor.
     * 
     * @return Resulting tensor.
     */
    auto operator!() const
    requires requires (T a) {!a;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary plus on each item and returns result as new tensor. Does not change original tensor.
     * 
     * @return Resulting tensor.
     * 
     * @warning Even if items have unary + implemented as identity (doing nothing), this overload will still traverse to apply
     * the operation.
     */
    auto operator+() const
    requires requires (T a) {+a;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary minus on each item and returns result as new tensor. Does not change original tensor.
     * 
     * @return Resulting tensor.
     */
    auto operator-() const
    requires requires (T a) {-a;};

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise negation (~) on each item in a tensor.
     */
    void complementInPlace();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary plus (+) on a tensor.
     */
    void plusInPlace();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary minus (-) on all items of the tensor.
     */
    void oppositeInPlace();

    Tensor<T>& operator++();

    Tensor<T> operator++(int) const;

    Tensor<T>& operator--();

    Tensor<T> operator--(int) const;

    

    // OPERATOR OVERLOADS END -------------------------------------------------------------------------------------------------



    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Allows to apply custom operation between each item of two tensors, items from this tensor as first operand 
     * and items from the second tensor passed as parameter as second operand.
     * 
     * @param tensor2 a second tensor to use the operation against as second operand.
     * @param operation a binary function that defines operation between two items.
     * 
     * @return A pointer to new resulting tensor.
     */
    template <apply_and_return_callable<T> C>
    inline auto applyAndReturn(const Tensor<T>& tensor2, C&& operation) const;
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Allows to apply custom operation two arguments where either one of them is tensor and one of them is value
     * of type T, or both arguments are tensor. In case of value, the operation is performed on every tensor item with value in
     * given order, in case of both arguments being tensor, the operation is performed item on item. In this case, both tensors
     * should have same sizes. Results of operation are stored in new tensor.
     * 
     * @param operand1 first operand either tensor or value of type T.
     * @param operand2 second operand either tensor or value of type T.
     * @param operation binary operation returning T and having correct signature defined in concept.
     * 
     * @return A pointer to new resulting tensor.
     */
    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B, apply_and_return_callable<T> C> 
    static auto applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Allows to apply custom operation between each item of two tensors and then store the result into caller tensor.
     * 
     * @param tensor2 a second tensor to use the operation against as second operand.
     * @param operation a binary function that defines operation between two items.
     */
    template <apply_callable<T> C>
    inline void apply(const Tensor<T>& tensor2, C&& operation); /*requires(std::is_invocable_r_v<void, C, T&, const T&>)*/

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Allows to apply custom operation two arguments where either one of them is tensor and one of them is value
     * of type T, or both arguments are tensor. In case of value, the operation is performed on every tensor item with value in
     * given order, in case of both arguments being tensor, the operation is performed item on item. In this case, both tensors
     * should have same sizes. Operation is done in place, where the storing operand is the first one, with exception of case,
     * where the first operand is value of type T.
     * 
     * @param operand1 first operand either tensor or value of type T.
     * @param operand2 second operand either tensor or value of type T.
     * @param operation binary operation returning T and having correct signature defined in concept.
     */
    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B, apply_callable<T> C> 
    static void apply(A& operand1, const B& operand2, C&& operation)
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Creates new instance with same dimensions as this tensor and applies passed callable on all items, then writes 
     * items into new instance, then returns the new instance. Looping order is unspecified.
     * 
     * @param operation unary operation returning T and having correct signature defined in concept.
     * 
     * @return A pointer to new resulting tensor.
     */
    template <foreach_and_return_callable<T> C>
    inline auto forEachAndReturn(C&& operation) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Creates new instance with same dimensions as given tensor and applies passed callable on all items, then writes 
     * items into new instance, then returns the new instance. Looping order is unspecified.
     * 
     * @param tensor tensor to perform the operation on.
     * @param operation unary operation returning T and having correct signature defined in concept.
     * 
     * @return A pointer to new resulting tensor.
     */
    template <foreach_and_return_callable<T> C>
    static auto forEachAndReturn(const Tensor<T>& tensor, C&& operation);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Applies operation on all items with passed callable. Looping order is unspecified.
     * 
     * @param operation unary operation, having correct signature defined in concept.
     */
    template <foreach_callable<T> C> 
    inline void forEach(C&& operation);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Applies operation on all items in given tensor with passed callable. Looping order is unspecified.
     * 
     * @param tensor to perform the operation on.
     * @param operation unary operation, having correct signature defined in concept.
     */
    template <foreach_callable<T> C>
    static void forEach(Tensor<T>& tensor, C&& operation);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Takes given coordinates as reference and changes it to next coordinates in ascending order. If given coordinates
     * are of the last item, it will loop over to coordinates of first item. Useful for in order traversal and TERRIBLE for 
     * accesing far (random) items. Shall not be used for jumps bigger than ~(10 * number of dimensions) at a time.
     * 
     * @param coordinates coordinates to be changed into coordinates of next item.
     * @param dimensionSizes used as upper exclusive limits for how coordinate values can be high.
     * 
     * @return Is @b true if given coordinate looped over to start, otherwise @b false.
     */
    static bool incrementCoords(std::vector<uint64_t>& coordinates, const std::vector<uint64_t>& dimensionSizes);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Virtual destructor.
    */
    virtual ~Tensor();



    private:

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Calculates coordinates from items index in a tensor, this is inverse method of "getIndex" method.
     * 
     * @param itemIndex it is the index of item that is stored in the tensor.
     * 
     * @return Coordinates of the item in the tensor.
    */
    std::vector<uint64_t> getCoords(uint64_t itemIndex) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Get items index in a tensor, this is inverse method of "getCoords".
     * 
     * @param coordinates an array of coordinates of one item in the tensor.
     * 
     * @return Index of one item in the tensor.
    */
    uint64_t getIndex(const std::vector<uint64_t>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Little endian implementation, thus not used by default. Calculates coordinates from items index in tensor, this
     * is inverse method of "littleGetIndex()" method.
     * 
     * @param itemIndex it is the index of item that is stored in the tensor.
     * 
     * @return Coordinates of the item in the tensor.
     * 
     * @note Not in use.
     */
    std::vector<uint64_t> littleGetCoords(int itemIndex) const;
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Little endian implementation, thus not used by default. Get items index number in tensor, this is inverse method
     * of "littleGetCoords()".
     * 
     * @param coordinates an array of coordinates of one item in tensor.
     * 
     * @return Index of one item in tensor when represented in one dimension.
     * 
     * @note Not in use.
     */
    int littleGetIndex(const std::vector<uint64_t>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Calculates the number of items in a tensor based on dimension sizes and resizes tensor to that number.
     * 
     * @return Number of items in tensor.
    */
    uint64_t updateNumberOfItems();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Calculates dimensionJumps_.
     * 
     * @return Number of items in tensor.
     */
    uint64_t updateDimensionJump();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Updates number of items and dimensionJumps_ at once. Is slightly faster that using methods updateNumberOfItems
     * and updateDimensionJump individually.
     */
    void update();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Compares two items using "==" and has two specializations for double and float using epsilon-abs comparison.
     * 
     * @param a first operand to compare.
     * @param b second operand to compare.
     * 
     * @return Boolean @b true if both operands are same and @b false if not.
     */
    inline bool compareItems(const T& a, const T& b) const requires(!std::is_floating_point<T>::value);
    inline bool compareItems(const T a, const T b) const requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Supposed to only run from constructor and set std::function attributes of this class. It just assigns values
     * already chosen during compile time.
     * 
     * @warning Do not use outside of constructor!
     */
    void defaultFunctions();
}; // end Tensor

} // end gema

#include "Tensor.tpp"

#endif