#include <vector>

#include <sycl/sycl.hpp>

#include "TensorParallel.hpp"
#include "MemoryBackendUSM.hpp"

namespace gema{
   
    
    template <class T>
    TensorParallel<T>::TensorParallel(const std::vector<uint64_t>& newTensorDimensionSizes) 
    : Tensor<T>(newTensorDimensionSizes) {
        this->tensor_ = LinearContainer<T>(MemoryBackendUSM<T, sycl::usm::alloc::device>(queue_));
    }

    template <class T>
    TensorParallel<T>::TensorParallel(const TensorParallel<T>& otherTensor){
        *this = otherTensor;
    }

    template <class T>
    TensorParallel<T>::TensorParallel(TensorParallel<T>&& otherTensor) noexcept {
        *this = std::move(otherTensor);
    }

    template <class T>
    TensorParallel<T>::TensorParallel(const TensorParallel<T>* otherTensor) 
    : Tensor<T>(otherTensor){
        queue_ = otherTensor->queue_;
    }

    template <class T>
    TensorParallel<T>::TensorParallel(){
        
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::operator=(const TensorParallel<T>& otherTensor){
        Tensor<T>::operator=(otherTensor);
        queue_ = otherTensor.queue_;
        return *this;
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::operator=(TensorParallel<T>&& otherTensor) noexcept {
        Tensor<T>::operator=(std::move(otherTensor));
        queue_ = std::move(otherTensor.queue_);
        return *this;
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
        const sycl::queue* queue = tensorOperand->queue_;

        TensorParallel<opReturnType> resultTensor = TensorParallel<opReturnType>(tensorOperand->getDimensionSizes());
        opReturnType* resultRawData = resultTensor.getData();

        T* operand1Raw;
        T* operand2Raw;

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

    // template <class T>
    // template <typename A, typename B, apply_callable_parallel<T> C>
    // void TensorParallel<T>::apply(A& operand1, const B& operand2, C&& operation)
    // requires(tensor_or_t_or_bothtensor_parallel<A, B, T>){

    //     Tensor<T>* tensorOperand = type_pick<Tensor<T>>(operand1, operand2);
    //     const sycl::queue* queue = tensorOperand->queue_;

    //     T* operand1Raw;
    //     T* operand2Raw;

    //     if constexpr (std::is_same_v<A, B>){
    //         operand1Raw = operand1.getData();
    //         operand2Raw = operand2.getData();
    //     }else if constexpr (std::is_same_v<A, T>){
    //         operand2Raw = operand2.getData();
    //     }else if constexpr (std::is_same_v<B, T>){
    //         operand1Raw = operand1.getData();
    //     }

    //     queue->parallel_for(tensorOperand->tensor_.size(), [=](sycl::id<1> idx){

    //         size_t i = idx[0];

    //         if constexpr (std::is_same_v<A, B>){
    //             T lhs = static_cast<T>(operand1Raw[i]);
    //             const T rhs = static_cast<T>(operand2Raw[i]);
    //             operation(lhs, rhs);
    //             operand1Raw[i] = static_cast<T>(lhs);
    //         }else if constexpr (std::is_same_v<A, T>){
    //             const T rhs = static_cast<T>(operand2Raw[i]);
    //             operation(operand1, rhs);
    //             operand2Raw[i] = static_cast<T>(rhs);
    //         }else if constexpr (std::is_same_v<B, T>){
    //             T lhs = static_cast<T>(operand1Raw[i]);
    //             operation(lhs, operand2);
    //             operand1Raw[i] = static_cast<T>(lhs);
    //         }
    //     }).wait();

    // }

    template <class T>
    template <foreach_and_return_callable_parallel<T> C>
    auto TensorParallel<T>::forEachAndReturn(C&& operation) const{
        return TensorParallel<T>::forEachAndReturn(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_and_return_callable_parallel<T> C>
    /*static*/ auto TensorParallel<T>::forEachAndReturn(const TensorParallel<T>& tensor, C&& operation){

        T* operandRaw = tensor.tensor_.data();
        const sycl::queue* queue = tensor.queue_;

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