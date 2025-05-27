#ifndef TENSOR_HPP
#define TENSOR_HPP

namespace gema{

// Forward declaration just for the concepts.
template<class T>
class Tensor;
    
// Concept that checks if type X is of type T or Tensor<T>. Useful for operator overloads.
template <typename X, typename T>
concept is_tensor_or_t = std::is_same_v<X, T> || std::is_same_v<X, Tensor<T>>;

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
 * @warning Even though the bool is supported, it is advised to use char instead, unless user is looking to take advantage
 * of std::vector bit bool storing for effectivity in memory (might be less optimized for methods that iterate like
 * forEach(), etc...) and thus slower.
 */
template<class T> 
class Tensor {

    private:

    std::vector<T> tensor_;                     /// The tensor data itself, represented by vector containing all the items.
    std::vector<uint64_t> dimensionSizes_;      /// Size od every tensor dimension.

    /// Function compares items in tensor and represents equality by bool.
    static std::function<bool(const T&, const T&)> defaultEquals_;
    std::function<bool(const T&, const T&)>* equals_ = &defaultEquals_;
    std::function<bool(const T&, const T&)> userEquals_;

    /// Function orders items in a way: (less, equal, more) -> (-1, 0, 1).
    static std::function<int(const T&, const T&)> defaultOrder_;
    std::function<int(const T&, const T&)>* order_ = &defaultOrder_;
    std::function<int(const T&, const T&)> userOrder_;

    
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
     * @brief Copy constructor, makes the object the same as the parameter object. For reference data types, it only copies
     * the references.
     * 
     * @param otherTensor Tensor to be copied.
     */
    Tensor(const Tensor<T>& otherTensor);

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
    T getItem(const std::vector<uint64_t>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets one value into tensor onto the desired coordinates.
     *  
     * @param value a value of generic type that will be stored in the tensor.
     * @param coordinates a vector of coordinates to place the value to.
    */
    void setItem(const T& value, const std::vector<uint64_t>& coordinates);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets one dimensional array and puts its items into tensor by order, if the array is longer than number of items 
     * in a tensor, only those that fit will be added.
     * 
     * @param tensorItems one dimensional vector of items to be added by order.
     * 
     * @note Risky and kinda shows the inner implementation by dodging the coordinate to index calculation, but its much faster
     * and can be beneficial if user knows what it is doing and needs to put many values in a tensor at once.
    */
    void setItems(const std::vector<T>& tensorItems);

    /** -----------------------------------------------------------------------------------------------------------------------
     * 
     */
    void setEquals(const std::function<bool(const T&, const T&)>& equals);

    void setOrder(const std::function<int(const T&, const T&)>& order);

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
     * @brief Parses std::string specifying the tensor dimension sizes and values. Inverse to "toString".
     * 
     * @param tensorString string in correct format to be parsed.
     * @param parseItem function to convert string to item.
     */
    void parse(const std::string& tensorString, const std::function<const T(const std::string&)>& parseItem);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Copies a tensor, meaning the items in it get copied, and if there is a pointer type stored
     * in a tensor, then the values pointed to by those pointers will be copied too.
     * 
     * @return Pointer to deep copy of this tensor.
     * 
     * @warning Do not use, will be deprecated and probably does not work correctly.
    */
    constexpr Tensor<T>* copy() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Fills tensor with passed value. Has specialization for bool because of how std::vector is implemented.
     * 
     * @param fill the value to be filled into all items in tensor.
    */
    void fillWith(const T& fill) requires(!std::is_same<T, bool>::value);
    void fillWith(const T& fill) requires(std::is_same<T, bool>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Copies one tensor to another by value.
     * 
     * @param tensor2 tensor which values are copied into this tensor.
     * 
     * @return Reference to this tensor after the copying.
     */
    Tensor<T>& operator=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Swaps two dimensions in a tensor.
     * 
     * @param dim1 first dimension to swap, default value is 0.
     * @param dim2 second dimension to swap, default value is 1.
     * 
     * @return A pointer to new allocated tensor, that got two dimensions transposed.
    */
    Tensor<T>* transposition(const int dim1 = 0, const int dim2 = 1) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Compares two tensors, checks if all items are equal and if the dimension sizes are equal.
     * 
     * @param tensor2 a second tensor to be compared by value.
     * 
     * @return Boolean @b true if the tensors are the same and @b false in not.
     */
    bool operator==(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds tensor with tensor item by item and returns result as new tensor. Does no size checking.
     * 
     * @param tensor2 addend tensor.
     * 
     * @return Pointer to new resulting tensor.
    */
    Tensor<T>* operator+(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds every tensor item to value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor addend tensor.
     * @param value addend value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<U>* operator+(const Tensor<U>& tensor, const U& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds value to every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value addend value.
     * @param tensor addend tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<U>* operator+(const U& value, const Tensor<U>& tensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds tensor to tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 addend tensor.
    */
    void operator+=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Adds value to tensor item by item in place. Does no size checking.
     * 
     * @param value addend value.
    */
    void operator+=(const T& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts tensor with tensor item by item and returns result as new tensor. Does no size checking.
     * 
     * @param tensor2 substrahend tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator-(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts value from every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param tensor minuend tensor.
     * @param value substrahend value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<U>* operator-(const Tensor<U>& tensor, const U& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts every tensor item from value and returns result as new tensor. Does no size checking.
     * 
     * @param value minuend value.
     * @param tensor substrahend tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<U>* operator-(const U& value, const Tensor<U>& tensor);
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts tensor from a tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 substrahend tensor.
     */
    void operator-=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Substracts value from tensor item by item in place. Does no size checking.
     * 
     * @param value substrahend value.
    */
    void operator-=(const T& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies tensor with tensor item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 multiplier tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator*(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor multiplicand tensor.
     * @param value multiplier value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<T>* operator*(const Tensor<U>& tensor, const U& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param value multiplicand value.
     * @param tensor multiplier tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<T>* operator*(const U& value, const Tensor<U>& tensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies tensor from a tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 product tensor.
     */
    void operator*=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies tensor item by item by value in place. Does no size checking.
     * 
     * @param value multiplier value.
    */
    void operator*=(const T& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides tensor with tensor item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator/(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor dividend tensor.
     * @param value divisor value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<T>* operator/(const Tensor<U>& tensor, const U& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Multiplies value by every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value dividend value.
     * @param tensor divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<T>* operator/(const U& value, const Tensor<U>& tensor);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides tensor from a tensor item by item in place. Does no size checking.
     * 
     * @param tensor2 divisor tensor.
     */
    void operator/=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Divides tensor item by item by value in place. Does no size checking.
     * 
     * @param value divisor value.
    */
    void operator/=(const T& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo item by item and returns result as new tensor. Does not size checking.
     * 
     * @param tensor2 divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator%(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo every tensor item by value and returns result as new tensor. Does no size checking.
     * 
     * @param tensor dividend tensor.
     * @param value divisor value.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<T>* operator%(const Tensor<U>& tensor, const U& value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs modulo value by every tensor item and returns result as new tensor. Does no size checking.
     * 
     * @param value dividend value.
     * @param tensor divisor tensor.
     * 
     * @return Pointer to new resulting tensor.
     */
    template<typename U> friend Tensor<T>* operator%(const U& value, const Tensor<U>& tensor);

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
     * @brief Performs bitwise "or" on each item in a tensor and returns result as new tensor. Is specialized for floating 
     * values so it can do bitwise operation on them too.
     * 
     * @param tensor2 second tensor to perform operation against.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator|(const Tensor<T>& tensor2) const;

    //template<typename F = T, typename std::enable_if<!std::is_floating_point<F>::value, double>::type = 0.>
    //Tensor<T>* operator|(const Tensor<T>& tensor2) const;
    //template<typename F = T, typename std::enable_if<std::is_floating_point<F>::value, double>::type = 0.>
    //Tensor<T>* operator|(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "or" on each item in a tensor in place. Is specialized for floating values
     * so it can do bitwise operation on them too.
     * 
     * @param tensor2 second tensor to perform operation against.
     */
    void operator|=(const Tensor<T>& tensor2) requires(!std::is_floating_point<T>::value);
    void operator|=(const Tensor<T>& tensor2) requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" on each item in a tensor and returns result as new tensor. Is specialized for floating 
     * values so it can do bitwise operation on them too.
     * 
     * @param tensor2 second tensor to perform operation against.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator&(const Tensor<T>& tensor2) const requires(!std::is_floating_point<T>::value);
    Tensor<T>* operator&(const Tensor<T>& tensor2) const requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "and" on each item in a tensor in place. Is specialized for floating values so it can do bitwise
     * operation on them too.
     * 
     * @param tensor2 second tensor to perform operation against.
     */
    void operator&=(const Tensor<T>& tensor2) requires(!std::is_floating_point<T>::value);
    void operator&=(const Tensor<T>& tensor2) requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" on each item in a tensor and returns result as new tensor. Is specialized for floating 
     * values so it can do bitwise operation on them too.
     * 
     * @param tensor2 second tensor to perform operation against.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator^(const Tensor<T>& tensor2) const requires(!std::is_floating_point<T>::value);
    Tensor<T>* operator^(const Tensor<T>& tensor2) const requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise "xor" on each item in a tensor in place. Is specialized for floating values so it can do bitwise
     * operation on them too.
     * 
     * @param tensor2 second tensor to perform operation against.
     */
    void operator^=(const Tensor<T>& tensor2) requires(!std::is_floating_point<T>::value);
    void operator^=(const Tensor<T>& tensor2) requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs left bit shift as this tensors item being shifted by amount in second tensors item and returns result 
     * as new tensor. Does not size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator<<(const Tensor<T>& tensor2) const requires(!std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs left bit shift as this tensors item being shifted by amount in second tensors item in place. Does not 
     * size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     */
    void operator<<=(const Tensor<T>& tensor2) requires(!std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs right bit shift as this tensors item being shifted by amount in second tensors item and returns result 
     * as new tensor. Does not size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     * 
     * @return Pointer to new resulting tensor.
     */
    Tensor<T>* operator>>(const Tensor<T>& tensor2) const requires(!std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs right bit shift as this tensors item being shifted by amount in second tensors item in place. Does not 
     * size checking.
     * 
     * @param tensor2 tensor specifying bit shift amount.
     */
    void operator>>=(const Tensor<T>& tensor2) requires(!std::is_floating_point<T>::value);
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise negation on each item in a tensor. Is specialized for floating values so it can do bitwise
     * operation on them too. Has specialization on bool type that uses ! (not) instead of ~ on the inside, user always uses ~.
     */
    Tensor<T>* operator~() requires(!std::is_floating_point<T>::value && !std::is_same<T, bool>::value);
    Tensor<T>* operator~() requires(std::is_same<T, bool>::value);
    Tensor<T>* operator~() requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs bitwise negation on each item in a tensor.
     */
    void complementInPlace();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary plus on a tensor and returns it as new object. Does nothing to items so it is basically a copy.
     * 
     * @return Pointer to new object.
     */
    Tensor<T>* operator+() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary plus on a tensor. Does actually nothing. Is here only for completeness.
     */
    void plusInPlace() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary minus on all items of copy of this object. Returns the pointer to the new object.
     * 
     * @return New object with items from this object negated.
     */
    Tensor<T>* operator-() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Performs unary minus on all items of the tensor.
     */
    inline void negateInPlace();

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Allows to apply custom operation between each item of two tensors, items from this tensor as first operand 
     * and items from the second tensor passed as parameter as second operand.
     * 
     * @param tensor2 a second tensor to use the operation against as second operand.
     * @param operation a binary function that defines operation between two items.
     * 
     * @return A pointer to resulting tensor.
     */
    inline Tensor<T>* applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation) const;
    // requires(!std::is_floating_point<T>::value);
    /*inline Tensor<T>* applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation)
    const requires(std::is_floating_point<T>::value);*/

    /*template <class A, class B> static Tensor<T>* applyAndReturn(const A& operand1, const B& operand2l,  
    const std::function<void(T&, const T&)>& operation)
    requires((std::is_same<A, T>::value || std::is_same<A, Tensor<T>>::value) && 
             (std::is_same<B, T>::value || std::is_same<B, Tensor<T>>::value) && 
             !std::is_same<A, B>::value);*/

    
    /** -----------------------------------------------------------------------------------------------------------------------
     * 
     * 
     */
    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B> 
    static Tensor<T>* applyAndReturn(const A& operand1, const B& operand2, const std::function<T(const T&, const T&)> &operation)
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>);//is_tensor_or_t<A, T> && is_tensor_or_t<B, T> && !std::is_same_v<A, B>

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Allows to apply custom operation between each item of two tensors and then store the result
     * into the caller tensor.
     * 
     * @param tensor2 a second tensor to use the operation against as second operand.
     * @param operation a binary function that defines operation between two items.
     */
    inline void apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    requires(!std::is_same<T, bool>::value);
    inline void apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    requires(std::is_same<T, bool>::value);

    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B> 
    static void apply(A& operand1, const B& operand2, const std::function<void(T&, const T&)>& operation)
 requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Creates new instance with same dimensions as given tensor and applies passed function on all items, then writes 
     * items into new instance, then returns the new instance.
     * 
     * @param tensor tensor to perform the operation on.
     * @param operation function defining operation on item.
     * 
     * @return Pointer to new instance upon its items was the function used.
     */
    static Tensor<T>* forEachAndReturn(const Tensor<T>& tensor, const std::function<T(const T&)>& operation);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Creates new instance with same dimensions as this tensor and applies passed function on all items, then writes 
     * items into new instance, then returns the new instance.
     * 
     * @param apply function that will be applied on all items.
     * 
     * @return Pointer to new instance upon its items was the function used.
     */
    inline Tensor<T>* forEachAndReturn(const std::function<T(const T&)>& operation) const;


    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Applies function on all items with passed function.
     * 
     * @param apply function that will be applied on all items.
     */
    void forEach(const std::function<void(T&)>& operation) requires(!std::is_same<T, bool>::value);
    void forEach(const std::function<void(T&)>& operation) requires(std::is_same<T, bool>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Applies function on all items in given tensor with passed function.
     * 
     * @param tensor to perform the operation on.
     * @param operation function that will be applied on all items.
     */
    static void forEach(const Tensor<T>& tensor, const std::function<void(T&)>& operation);

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
    std::vector<uint64_t> getCoords(int itemIndex) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Get items index in a tensor, this is inverse method of "getCoords".
     * 
     * @param coordinates an array of coordinates of one item in the tensor.
     * 
     * @return Index of one item in the tensor.
    */
    int getIndex(const std::vector<uint64_t>& coordinates) const;

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
     * @brief Calculates the number of possible items in a tensor based on given dimension sizes.
     * @par
     * Is used to calculate the number of items before the tensor itself is allocated, and should be useless afterwards.
     * 
     * @param dimensionSizes vector containing size of each dimension.
     * 
     * @return Total number of items that can fit into a tensor.
     * 
     * @warning Do not use outside of constructor! Same result can be achieved by getting size of the tensor itself!
    */
    int calculateNumberOfItems(const std::vector<uint64_t>& dimensionSizes) const;

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