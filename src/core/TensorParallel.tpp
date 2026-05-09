#include <vector>

#include <sycl/sycl.hpp>

#include "TensorParallel.hpp"
#include "MemoryBackendUSM.hpp"

namespace gema{
   
    
    template <class T>
    TensorParallel<T>::TensorParallel(const LinearContainer<uint64_t>& newTensorDimensionSizes)
    : tensor_(newTensorDimensionSizes, MemoryBackendUSM<T, usmKind_>(queue_)){
        // this->tensor_ = Tensor<T>(MemoryBackendUSM<T, usmKind_>(queue_));
        // this->dimensionSizes_ = LinearContainer<uint64_t>(newTensorDimensionSizes, MemoryBackendUSM<uint64_t, usmKind_>(queue_));
        // this->update();
    }

    template <class T>
    TensorParallel<T>::TensorParallel(const TensorParallel<T>& otherTensor)
    : tensor_(otherTensor){
        queue_ = otherTensor->queue_;
    }

    template <class T>
    TensorParallel<T>::TensorParallel(TensorParallel<T>&& otherTensor) noexcept 
    : tensor_(std::move(otherTensor)){
        queue_ = std::move(otherTensor->queue_);
    }

    template <class T>
    TensorParallel<T>::TensorParallel(const TensorParallel<T>* otherTensor)
    : tensor_(otherTensor){
        queue_ = otherTensor->queue_;
    }

    template <class T>
    TensorParallel<T>::TensorParallel(){
        
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::operator=(const TensorParallel<T>& otherTensor){
        tensor_ = otherTensor;
        queue_ = otherTensor.queue_;
        return *this;
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::operator=(TensorParallel<T>&& otherTensor) noexcept {
        tensor_ = std::move(otherTensor);
        queue_ = std::move(otherTensor.queue_);
        return *this;
    }

    template <class T>
    const LinearContainer<uint64_t>& TensorParallel<T>::getDimensionSizes() const {
        tensor_.getDimensionSizes();
    }

    template <class T>
    uint64_t TensorParallel<T>::getNumberOfDimensions() const {
        return tensor_.getNumberOfDimensions();
    }

    template <class T>
    uint64_t TensorParallel<T>::getNumberOfItems() const {
        return tensor_.getNumberOfItems();
    }

    template <class T>
    T &TensorParallel<T>::getItem(const LinearContainer<uint64_t>& coordinates){
         return tensor_.getItem(coordinates);
    }

    // template <class T>
    // TensorParallel<T> TensorParallel<T>::transpositionAndReturn(const int dim1, const int dim2) const {

    //     if(dim1 == dim2) return TensorParallel<T>(this->dimensionSizes_);

    //     // Copying the dimensionSizes
    //     // Change assigment to just construction of correct size
    //     std::vector<uint64_t> transposedDimensionSizes = this->dimensionSizes_; 

    //     // Swapping the dimension sizes
    //     transposedDimensionSizes[dim1] = this->dimensionSizes_[dim2]; 
    //     transposedDimensionSizes[dim2] = this->dimensionSizes_[dim1];
        
    //     // Initializing the new tensor
    //     TensorParallel<T> tensorTransposed = TensorParallel<T>(transposedDimensionSizes);

    //     // std::vector<uint64_t> original, switched;
    //     // original.resize(dimensionSizes_.size());
    //     // switched.resize(dimensionSizes_.size());

    //     const uint64_t dimensionCount = this->dimensionSizes_.size();

    //     const T* rawOldDimensionSizes = sycl::malloc(dimensionCount, *queue_, usmKind_);
    //     queue_->memcpy(rawOldDimensionSizes, this->dimensionSizes_.data(), sizeof(uint64_t) * dimensionCount);

    //     const T* rawData = this->tensor_.getData();
    //     T* rawTransposedData = tensorTransposed.getData();

    //     T* originalCoords = sycl::malloc(dimensionCount, *queue_, usmKind_);
    //     T* switchedCoords = sycl::malloc(dimensionCount, *queue_, usmKind_);

    //     queue_->parallel_for(this->tensor_.size(), [=](sycl::id<1> idx){

    //         size_t i = idx[0];

    //         // The swap of two desired coordinates
    //         for(uint64_t j = 0; j < dimensionCount; j++){
    //             switchedCoords[j] = originalCoords[j];
    //         }
            
    //         switchedCoords[dim1] = originalCoords[dim2];
    //         switchedCoords[dim2] = originalCoords[dim1];

    //         rawTransposedData[tensorTransposed.getIndex(switchedCoords)] = std::move(rawData[i]);

    //         Tensor<T>::incrementCoords(switchedCoords, rawOldDimensionSizes);

    //     }).wait();

    //     return tensorTransposed;
    // }

    // template <class T>
    // void TensorParallel<T>::transposition(const uint64_t dim1, const uint64_t dim2){

    //     if(dim1 == dim2) return;

    //     // Copying the dimensionSizes
    //     // Change assigment to just construction of correct size
    //     const LinearContainer<uint64_t> oldDimensionSizes = this->dimensionSizes_;
    //     std::span<const uint64_t> oldDimensionSizesView = oldDimensionSizes;

    //     // Swapping the dimension sizes
    //     const uint64_t temporaryDimensionSize1 = this->dimensionSizes_[dim1];
    //     this->dimensionSizes_[dim1] = this->dimensionSizes_[dim2];
    //     this->dimensionSizes_[dim2] = temporaryDimensionSize1;

    //     const uint64_t itemCount = this->updateDimensionJump();
    //     const uint64_t dimensionCount = this->dimensionSizes_.size();
        
    //     // Initializing the new data
    //     LinearContainer<T> newData(itemCount, MemoryBackendUSM<T, usmKind_>(queue_));

    //     T* oldDataRaw = this->tensor_.data();
    //     T* newDataRaw = newData.data();

    //     uint64_t* coordsBuffer = sycl::malloc<uint64_t>(itemCount * dimensionCount, *queue_, usmKind_);

    //     queue_->parallel_for(itemCount, [=](sycl::id<1> idx){

    //         size_t i = idx[0];

    //         uint64_t* coords = coordsBuffer + i * dimensionCount;
    //         Tensor<T>::getCoords(i, oldDimensionSizesView, coords);

    //         uint64_t temporaryCoord1 = coords[dim1];
    //         coords[dim1] = coords[dim2];
    //         coords[dim2] = temporaryCoord1;

    //         std::span<const uint64_t> coordsView{coords, dimensionCount};

    //         newDataRaw[Tensor<T>::getIndex(coordsView)] = std::move(oldDataRaw[i]);

    //     }).wait();

    //     this->tensor_ = std::move(newData);
    // }

    template <class T>
    void TensorParallel<T>::resize(const LinearContainer<uint64_t>& newDimensionSizes){

    }

    template <class T>
    void TensorParallel<T>::resize(const uint64_t newDimensionSize, const uint64_t dimensionIndex){

    }

    template <class T>
    void TensorParallel<T>::addDimension(const uint64_t newDimensionSize, const uint64_t putBefore){

    }

    template <class T>
    void TensorParallel<T>::removeDimension(const uint64_t removedDimensionIndex){

    }

    template <class T>
    template <apply_and_return_callable_parallel<T> C>
    auto TensorParallel<T>::applyAndReturn(const TensorParallel<T>& tensor2, C&& operation) const {
        return TensorParallel<T>::applyAndReturn(*this, tensor2, std::forward<C>(operation));
    }

    template <class T>
    template <typename A, typename B, apply_and_return_callable_parallel<T> C>
    /*static*/ auto TensorParallel<T>::applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(tensor_or_t_or_bothtensor_parallel<A, B, T>){

        using opReturnType = decltype(operation(std::declval<T>(), std::declval<T>()));
        const TensorParallel<T>* tensorOperand = type_pick<TensorParallel<T>>(operand1, operand2);
        sycl::queue* queue = tensorOperand->queue_;

        TensorParallel<opReturnType> resultTensor = TensorParallel<opReturnType>(tensorOperand->getDimensionSizes());
        opReturnType* resultRawData = resultTensor.getData();

        const T* operand1Raw;
        const T* operand2Raw;

        if constexpr (std::is_same_v<A, B>){
            operand1Raw = operand1.tensor_.data();
            operand2Raw = operand2.tensor_.data();
        }else if constexpr (std::is_same_v<A, T>){
            operand2Raw = operand2.tensor_.data();
        }else if constexpr (std::is_same_v<B, T>){
            operand1Raw = operand1.tensor_.data();
        }

        queue->parallel_for(tensorOperand->tensor_.size(), [=](sycl::id<1> idx){

            size_t i = idx[0];
            if constexpr (std::is_same_v<A, B>){
                resultRawData[i] = operation(operand1Raw[i], operand2Raw[i]);
            }else if constexpr (std::is_same_v<A, T>){
                resultRawData[i] = operation(operand1, operand2Raw[i]);
            }else if constexpr (std::is_same_v<B, T>){
                resultRawData[i] = operation(operand1Raw[i], operand2);
            }
        }).wait();

        return resultTensor;
    }

    template <class T>
    template <apply_callable_parallel<T> C>
    void TensorParallel<T>::apply(const TensorParallel<T>& tensor2, C&& operation){
        TensorParallel<T>::apply(*this, tensor2, std::forward<C>(operation));
    }

    template <class T>
    template <apply_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::apply(TensorParallel<T>& operand1, const TensorParallel<T>& operand2, C&& operation){

        const sycl::queue* queue = operand1->queue_;
        T* operand1Raw = operand1.getData();
        const T* operand2Raw = operand2.getData();

        queue->parallel_for(operand1->tensor_.size(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operand1Raw[i], operand2Raw[i]);
        }).wait();
    }

    template <class T>
    template <apply_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::apply(TensorParallel<T>& operand1, const T& operand2, C&& operation){

        const sycl::queue* queue = operand1->queue_;
        T* operand1Raw = operand1.getData();

        queue->parallel_for(operand1->tensor_.size(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operand1Raw[i], operand2);
        }).wait();
    }

    template <class T>
    template <apply_reverse_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::apply(const T& operand1, TensorParallel<T>& operand2, C&& operation){

        const sycl::queue* queue = operand2->queue_;
        T* operand2Raw = operand2.getData();

        queue->parallel_for(operand2->tensor_.size(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operand1, operand2Raw[i]);
        }).wait();
    }

    template <class T>
    template <foreach_and_return_callable_parallel<T> C>
    auto TensorParallel<T>::forEachAndReturn(C&& operation) const{
        return TensorParallel<T>::forEachAndReturn(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_and_return_callable_parallel<T> C>
    /*static*/ auto TensorParallel<T>::forEachAndReturn(const TensorParallel<T>& tensor, C&& operation){

        const T* operandRaw = tensor.tensor_.data();
        sycl::queue* queue = tensor.queue_;

        using opReturnType = decltype(operation(std::declval<T>()));
        Tensor<opReturnType> resultTensor = Tensor<opReturnType>(tensor.getDimensionSizes());
        T* resultRawData = resultTensor.getData();

        queue->parallel_for(tensor.tensor_.size(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            resultRawData[i] = operation(operandRaw[i]);
        }).wait();

        return resultTensor;
    }

    template <class T>
    template <foreach_callable_parallel<T> C>
    void TensorParallel<T>::forEach(C&& operation){
        TensorParallel<T>::forEach(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::forEach(TensorParallel<T>& tensor, C&& operation){

        const sycl::queue* queue = tensor.queue_;
        T* operandRaw = tensor.getData();

        queue->parallel_for(tensor.tensor_.size(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operandRaw[i]);
        }).wait();
    }
}