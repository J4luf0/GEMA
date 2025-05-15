#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>

//#define uint64t uint64_t

namespace GeMa{

// ============================================================================================================================
/**
 * @brief Class representing generic tensor.
 *
 * @par
 * Inner workings - algorithms
 * @par
 * Tensor is represented inside class as one dimensional vector with two most important methods: Tesor<T>::getCoords that
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
template<class T> class Tensor{

    private:

    std::vector<T> tensor_;                  // The tensor data itself, represented by vector containing all the elements.
    std::vector<int> dimensionSizes_;        // Size od every tensor dimension.

    std::function<void(const T&)> tensorOutput_;
    std::function<void(const T&)> itemOutput_;

    std::function<bool(const T&, const T&)> equals_;    // Function compares items in tensor and represents equality by bool.
    std::function<int(const T&, const T&)> order_;      // Function orders items in a way: (less, equal, more) -> (-1, 0, 1).

    public:

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Sets dimensionSizes, calculates number of items and then allocates them on tensor, then sets functional
     * attributes values yet the default lambda itself is decided at compile time. The result is empty tensor, with defined 
     * dimensions and allocated space.
     *
     * @param newTensorDimensionSizes a vector filled with sizes of dimensions.
    */
    Tensor(const std::vector<int>& newTensorDimensionSizes) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Copy constructor, makes the object the same as the parameter object.
     * 
     * @param otherTensor a tensor to be copied.
     */
    Tensor(const Tensor<T>& otherTensor) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Empty constructor so it can be declared without being initalized - trying to do something with
     * uninitialized tensor is sure undefined behavior, not recommended.
     */
    Tensor() noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public getter to get all dimension sizes.
     * 
     * @return Vector containing one int per dimension with value of its size.
    */
    const std::vector<int>& getDimensionSizes() const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public getter to get the number of dimensions of a tensor.
     * 
     * @return A number of dimensions.
    */
    uint64_t getNumberOfDimensions() const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public getter to get the number of items in a tensor.
     * 
     * @return Number of items in a tensor.
     */
    uint64_t getNumberOfItems() const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public getter to get item on provided coordinates, returns the item by value because large objects should
     * have been already represented by pointer or reference.
     * 
     * @param coordinates vector of coordinates specifying the item to be returned.
     * 
     * @return Item on the provided coordinates.
    */
    T getItem(const std::vector<int>& coordinates) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public setter to assign one value into tensor onto the desired coordinates.
     *  
     * @param value a value of generic type that will be stored in the tensor.
     * @param coordinates a vector of coordinates that the value will be assigned to.
    */
    void setItem(const T& value, const std::vector<int>& coordinates) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public setter that takes in one dimensional array and puts its items into tensor by order, if the array is 
     * longer than number of items in a tensor, only those that fit will be added.
     * 
     * @param tensorItems one dimensional vector of items to be added by order.
    */
    void setItems(const std::vector<T>& tensorItems) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public setter that allows the user to set the output of the tensor through this->showTensor() method.
     * 
     * @param tensorOutput function that defines the output of the tensor.
    */
    void setTensorOutput(const std::function<void(const T&)> tensorOutput) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public setter that allows the user to set the output of the items through this->showItem() method.
     * 
     * @param tensorOutput function that defines the output of the tensor.
    */
    void setItemOutput(const std::function<void(const T&)> itemOutput) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method to calculate, if the tensor dimensions have the same sizes.
     * 
     * @return Bool true if the tensor is equilateral and false if not.
    */
    bool isEquilateral() const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method that generates a string from tensor items in form of parsable curly bracket hierarchy.
     * 
     * @return A parsable string representing the tensor.
     */
    std::string toString() const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method that parses std::string specifying the tensor dimension sizes and values.
     * 
     * @param tensor string in correct format to be parsed.
     */
    void parse(const std::string& tensor) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method to deep copy a tensor, meaning the items in it get copied, and if there is a pointer type stored
     * in a tensor, then the values pointed to by those pointers will be copied too.
     * 
     * @return Pointer to deep copy of this tensor.
    */
    constexpr Tensor<T>* copy() const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method that fills tensor with passed value. Has specialization for bool because of how std::vector 
     * is implemented.
     * 
     * @param fill the value to be filled into all items in tensor.
    */
    void fillWith(const T& fill) noexcept requires(!std::is_same<T, bool>::value);
    void fillWith(const T& fill) noexcept requires(std::is_same<T, bool>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to deep copy one tensor to another.
     * 
     * @param tensor2 tensor which values are deep copied into this tensor.
     * 
     * @return Reference to this tensor after the copying.
     */
    Tensor<T>& operator=(const Tensor<T>& tensor2) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method to swap two dimensions in a tensor.
     * 
     * @param dim1 first dimension to swap, default value is 0.
     * @param dim2 second dimension to swap, default value is 1.
     * 
     * @return A pointer to new allocated tensor, that has got transposed two dimensions.
    */
    Tensor<T>* transposition(const int dim1 = 0, const int dim2 = 1) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to compare two tensors, iterating through all items the return value is bool true if items
     * are all the same and false if there is atleast one different or the dimensions didnt match.
     * 
     * @param tensor2 a second tensor to be compared.
     * 
     * @return Boolean @b true if the tensors are the same and @b false in not.
     */
    bool operator==(const Tensor<T>& tensor2) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to add two tensors of the same size, both by reference.
     * 
     * @param tensor2 a second tensor to be added as reference (the same as the first).
     * 
     * @return A pointer to new allocated tensor that is the sum of the both.
    */
    Tensor<T>* operator+(const Tensor<T>& tensor2) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to substract parameter tensor from this tensor.
     * 
     * @param tensor2 a second tensor to be substracted.
     * 
     * @return A pointer to new allocated tensor that the result of substraction.
     */
    Tensor<T>* operator-(const Tensor<T>& tensor2) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to add tensor to tensor.
     * 
     * @param tensor2 a tensor to be added to this tensor.
    */
    void operator+=(const Tensor<T>& tensor2) noexcept;
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to substract a tensor from a tensor.
     * 
     * @param tensor2 a tensor to be substracted from this tensor.
     */
    void operator-=(const Tensor<T>& tensor2) noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise or on each item in a tensor.
     * 
     * @param tensor2 a second tensor to perform operation against.
     * 
     * @return A pointer to new allocated resulting tensor.
     */
    Tensor<T>* operator|(const Tensor<T>& tensor2) const noexcept requires(!std::is_floating_point<T>::value);
    Tensor<T>* operator|(const Tensor<T>& tensor2) const noexcept requires(std::is_floating_point<T>::value);

    //template<typename F = T, typename std::enable_if<!std::is_floating_point<F>::value, double>::type = 0.>
    //Tensor<T>* operator|(const Tensor<T>& tensor2) const;
    //template<typename F = T, typename std::enable_if<std::is_floating_point<F>::value, double>::type = 0.>
    //Tensor<T>* operator|(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise or on each item in a tensor and save result in this tensor.
     * 
     * @param tensor2 a second tensor to perform operation against.
     */
    void operator|=(const Tensor<T>& tensor2) noexcept requires(!std::is_floating_point<T>::value);
    void operator|=(const Tensor<T>& tensor2) noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise and on each item in a tensor.
     * 
     * @param tensor2 a second tensor to perform operation against.
     * 
     * @return A pointer to new allocated resulting tensor.
     */
    Tensor<T>* operator&(const Tensor<T>& tensor2) const noexcept requires(!std::is_floating_point<T>::value);
    Tensor<T>* operator&(const Tensor<T>& tensor2) const noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise and on each item in a tensor and save result in this tensor.
     * 
     * @param tensor2 a second tensor to perform operation against.
     */
    void operator&=(const Tensor<T>& tensor2) noexcept requires(!std::is_floating_point<T>::value);
    void operator&=(const Tensor<T>& tensor2) noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise xor on each item in a tensor.
     * 
     * @param tensor2 a second tensor to perform operation against.
     * 
     * @return A pointer to new allocated resulting tensor.
     */
    Tensor<T>* operator^(const Tensor<T>& tensor2) const noexcept requires(!std::is_floating_point<T>::value);
    Tensor<T>* operator^(const Tensor<T>& tensor2) const noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise xor on each item in a tensor and save result in this tensor.
     * 
     * @param tensor2 a second tensor to perform operation against.
     */
    void operator^=(const Tensor<T>& tensor2) noexcept requires(!std::is_floating_point<T>::value);
    void operator^=(const Tensor<T>& tensor2) noexcept requires(std::is_floating_point<T>::value);
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public overload to perform bitwise negation on each item in a tensor.
     * 
     * @note Has specialization on bool type that uses ! (not) instead of ~.
     */
    void operator~() noexcept requires(!std::is_floating_point<T>::value && !std::is_same<T, bool>::value);
    void operator~() noexcept requires(std::is_same<T, bool>::value);
    void operator~() noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method that allows to apply custom operation between each item of two tensors, items from this
     * tensor as first operand and items from the second tensor passed as parameter as second operand.
     * 
     * @param tensor2 a second tensor to use the operation against as second operand.
     * @param operation a binary function that defines operation between two items.
     * 
     * @return A pointer to resulting tensor.
     */
    inline Tensor<T>* applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation)
    const noexcept requires(!std::is_floating_point<T>::value);
    inline Tensor<T>* applyAndReturn(const Tensor<T>& tensor2, const std::function<T(const T&, const T&)>& operation)
    const noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method that allows to apply custom operation between each item of two tensors and then store the result
     * into the fisrt tensor.
     * 
     * @param tensor2 a second tensor to use the operation against as second operand.
     * @param operation a binary function that defines operation between two items.
     */
    inline void apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    noexcept requires(!std::is_same<T, bool>::value);
    inline void apply(const Tensor<T>& tensor2, const std::function<void(T&, const T&)>& operation)
    noexcept requires(std::is_same<T, bool>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Public method to apply function on all elements thru passed function.
     * 
     * @param apply function that will be applied on all items.
     */
    void forEach(const std::function<void(T&)>& apply) noexcept requires(!std::is_same<T, bool>::value);
    void forEach(const std::function<void(T&)>& apply) noexcept requires(std::is_same<T, bool>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Virtual destructor.
    */
    virtual ~Tensor() noexcept;



    private:

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Calculates coordinates from items index in a tensor, this is inverse method of "getIndex()" method.
     * 
     * @param itemIndex it is the index of item that is stored in the tensor.
     * 
     * @return Coordinates of the item in the tensor.
    */
    std::vector<int> getCoords(int itemIndex) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Get items index in a tensor, this is inverse method of "getCoords()".
     * 
     * @param coordinates an array of coordinates of one item in the tensor.
     * 
     * @return Index of one item in the tensor.
    */
    int getIndex(const std::vector<int>& coordinates) const noexcept;

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
    std::vector<int> littleGetCoords(int itemIndex) const noexcept;
    
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
    int littleGetIndex(const std::vector<int>& coordinates) const noexcept;

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
    int calculateNumberOfItems(const std::vector<int>& dimensionSizes) const noexcept;

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Compares two items using "==" and has two specializations for double and float using epsilon-abs comparison.
     * 
     * @param a first operand to compare.
     * @param b second operand to compare.
     * 
     * @return Boolean @b true if both operands are same and @b false if not.
     */
    inline bool compareItems(const T& a, const T& b) const noexcept requires(!std::is_floating_point<T>::value);
    inline bool compareItems(const T a, const T b) const noexcept requires(std::is_floating_point<T>::value);

    /** -----------------------------------------------------------------------------------------------------------------------
     * @brief Supposed to only run from constructor and set std::function attributes of this class. It just assigns values
     * already chosen during compile time.
     * 
     * @warning Do not use outside of constructor!
     */
    void defaultFunctions() noexcept;
}; // end Tensor

} // end GeMa

#include "Tensor.tpp"

#endif