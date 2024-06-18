
#ifndef ITENSOR_HPP
#define ITENSOR_HPP

#include <functional>

/** ===========================================================================================================================
 * Class for work with generic tensors
 *
 * Tensor is represented in class as one dimensional array with two key methods getCoords() that calculates the made-up
 * coordinates of the tensor and inverse method getIndex() that return actual vector index when provided with coordinates
 * 
 * Tensor is made out of items, each having unique coordinates
 * 
 * This is inner class that doesnt implement any checking against bad input, that will be implemented in wrapper class
 */
template<class T> class Tensor{

    private:

    std::vector<T> tensor;                  // The tensor itself, represented by one-dimensional vector containing all the elements
    std::vector<int> dimensionSizes;        // Size od each tensor dimension

    std::function<void(const T&)> tensorOutput;
    std::function<void(const T&)> itemOutput;

    public:

    /** -----------------------------------------------------------------------------------------------------------------------
     * Tensor() constructor - Sets dimensionSizes, calculates number of items and then allocates them on tensor, 
     * the result is empty tensor, with defined dimensions and allocated space
     *
     * @param newTensorDimensionSizes - a vector filled with sizes of dimensions
    */
    Tensor(const std::vector<int>& newTensorDimensionSizes);

    /** -----------------------------------------------------------------------------------------------------------------------
     * getDimensionSizes() - Public method to get all dimension sizes
     * 
     * @return - vector containing one int per dimension with value of its size
    */
    const std::vector<int>& getDimensionSizes() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getNumberOfDimensions() - Public method to get the number of dimensions of a tensor
     * 
     * @return - a number of dimensions
    */
    uint64_t getNumberOfDimensions() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getItem() - Public method to get item on provided coordinates, returns the item by value because large objects should
     * have been already represented by pointer or reference
     * 
     * @param coordinates - vector of coordinates specifying the item to be returned
     * 
     * @return - item on the provided coordinates
    */
    T getItem(const std::vector<int>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getPointer() - Public method that forcibly returns address of item. Use with caution, because it casts to (void*) and
     * if the item is already of pointer type, then it might need to dereference twice
     * 
     * @param coordinates - vector of coordinates specifying the item to be returned
     * 
     * @return - address of item on the provided coordinates
    */
    void* getPointer(const std::vector<int>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * assign() - Public method to assign one value into tensor onto the desired coordinates
     *  
     * @param value - a value of generic type that will be stored in the tensor
     * @param coordinates - a vector of coordinates that the value will be assigned to
    */
    void setItem(const T& value, const std::vector<int>& coordinates);

    /** -----------------------------------------------------------------------------------------------------------------------
     * setItems() - Public method that takes in one dimensional array and puts its items into tensor by order, if the array is 
     * longer than number of items in a tensor, only those that fit will be added
     * 
     * @param tensorItems - one dimensional array of items to be added by order
    */
    void setItems(const std::vector<T>& tensorItems);

    /** -----------------------------------------------------------------------------------------------------------------------
     * setTensorOutput() - Public method that allows the user to set the output of the tensor through this->showTensor() method
     * 
     * @param tensorOutput - function that defines the output of the tensor
    */
    void setTensorOutput(const std::function<void(const T&)> tensorOutput);

    /** -----------------------------------------------------------------------------------------------------------------------
     * setTensorOutput() - Public method that allows the user to set the output of the items through this->showItem() method
     * 
     * @param tensorOutput - function that defines the output of the tensor
    */
    void setItemOutput(const std::function<void(const T&)> itemOutput);

    /** -----------------------------------------------------------------------------------------------------------------------
     * isTensorEquilateral() - Public method to calculate, if the tensor dimensions have the same sizes
     * 
     * @return - boolean true if the tensor is equilateral and false if not
    */
    bool isTensorEquilateral() const;

    void forEach(const std::function<void(T&)>& apply);

    /** -----------------------------------------------------------------------------------------------------------------------
     * fillWith() - Public method that fills tensor with passed value
     * 
     * @param fill - the value to be filled into all items in tensor
    */
    void fillWith(const T& fill);

    /** -----------------------------------------------------------------------------------------------------------------------
     * transposition() - Public method to swap two dimensions in a tensor
     * 
     * @param dim1 - first dimension to swap, default value is 0
     * @param dim2 - second dimension to swap, default value is 1
     * 
     * @return - a pointer to new allocated tensor, that has got transposed two dimensions
    */
    Tensor<T>* transposition(const int dim1 = 0, const int dim2 = 1) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * operator== - Public overload to compare two tensors, iterating through all items the return value is bool true if items
     * are all the same and false if there is atleast one different or the dimensions didnt match
     * 
     * @param tensor2 - a second tensor to be compared
     * 
     * @return - boolean true if the tensors are the same and false in not
     */
    bool operator==(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * operator+() - Public overload to add two tensors of the same size, both by reference
     * 
     * @param tensor2 - a second tensor to be added as reference (the same as the first)
     * 
     * @return - a pointer to new allocated tensor that is the sum of the both
    */
    Tensor<T>* operator+(const Tensor<T>& tensor2) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * operator+=() - Public overload to add tensor to tensor
     * 
     * @param tensor2 - a tensor to be added to this tensor
    */
    void operator+=(const Tensor<T>& tensor2);

    /** -----------------------------------------------------------------------------------------------------------------------
     * operator=() - Public overload to deep copy the tensor.
     * 
     * @return - pointer to deep copy of this tensor
    */
    constexpr Tensor<T>& operator=(const Tensor<T>& assigner) const;
    constexpr Tensor<T>* operator=(const Tensor<T>* assigner) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * showTensor() - Public method to output the whole tensor into std::cout
     * 
     * Note: Is currently working only for 1d, 2d and maybe 3d tensors
    */
    void showTensor() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * showItem() - Public method to write into cout the item index based on the coordinates input
     * 
     * @param coordinates - address of coordinates in a tensor
    */
    void showItem(const std::vector<int>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * ~Tensor() - Virtual destructor
    */
    virtual ~Tensor();



    private:

    /** -----------------------------------------------------------------------------------------------------------------------
     * getCoords() - Private method to get coordinates from itemNumber in tensor, this is inverse method of "getIndex()" method
     * 
     * @param itemNumber - it is the index of item that is stored in the tensor
     * 
     * @return - coordinates of the item in the tensor
     * 
     * Note: Tensor is actually stored as one-dimensional array so any item can be represented either in artificial coordinate 
     * system or by just index in the actual array
    */
    std::vector<int> getCoords(int itemNumber) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getIndex() - Private method to get items index number in tensor, this is inverse method of "getCoords()"
     * 
     * @param coordinates - an array of coordinates of one item in tensor
     * 
     * @return - index of one item in tensor when represented in one dimension
     * 
     * Note: Tensor is actually stored as one-dimensional array so any item can be represented either in artificial coordinate
     * system or by just index in the actual array
    */
    int getIndex(const std::vector<int>& coordinates) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * getNumberOfItems() - Private method to calculate the number of possible items in a tensor based on given dimension sizes
     * This method calculates the number of items before the tensor itself is allocated, and should be useless afterwards
     * 
     * @param dimensionSizes - vector containing size of each dimension
     * 
     * @return - total number of items that can fit into a tensor
    */
    int getNumberOfItems(const std::vector<int>& dimensionSizes) const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * copy() - Private method to deep copy a tensor, meaning the items in it get copied, but if the items in it are of pointer
     * type, then it is not guaranteed to copy values of those pointers
     * 
     * @return - pointer to deep copy of this tensor
    */
    constexpr Tensor<T>* copy() const;

    /** -----------------------------------------------------------------------------------------------------------------------
     * constructorMessage() - Private method to output message to console about the object creation
     * 
     * @param dimensionSizes - the dimension sizes to be output
    */
    void constructorMessage(const std::vector<int>& dimensionSizes) const;
};

#endif