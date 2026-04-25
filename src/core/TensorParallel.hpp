#ifndef TENSOR_PARALLEL_HPP
#define TENSOR_PARALLEL_HPP

#include <sycl/sycl.hpp>

#include "Tensor.hpp"

namespace gema{

template<class T>
class TensorParallel : public Tensor<T>{

    inline static sycl::queue queue_{sycl::property::queue::in_order{}};

    public:

    TensorParallel(const std::vector<uint64_t>& newTensorDimensionSizes);

    TensorParallel(const std::vector<uint64_t>& newTensorDimensionSizes, const LinearContainer<T>& newTensorData);

    TensorParallel(const TensorParallel<T>& otherTensor);

    TensorParallel(TensorParallel<T>&& otherTensor) noexcept;

    TensorParallel(const TensorParallel<T>* otherTensor);

    TensorParallel();

    template <apply_and_return_callable<T> C>
    inline auto applyAndReturn(const Tensor<T>& tensor2, C&& operation) const;
    
    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B, apply_and_return_callable<T> C> 
    static auto applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>);

    template <apply_callable<T> C>
    inline void apply(const Tensor<T>& tensor2, C&& operation);

    template <is_tensor_or_t<T> A, is_tensor_or_t<T> B, apply_callable<T> C> 
    static void apply(A& operand1, const B& operand2, C&& operation)
    requires(std::is_same_v<A, Tensor<T>> || std::is_same_v<B, Tensor<T>>);

    template <foreach_and_return_callable<T> C>
    inline auto forEachAndReturn(C&& operation) const;

    template <foreach_and_return_callable<T> C>
    static auto forEachAndReturn(const Tensor<T>& tensor, C&& operation);

    template <foreach_callable<T> C> 
    inline void forEach(C&& operation);

    template <foreach_callable<T> C>
    static void forEach(Tensor<T>& tensor, C&& operation);

};

}

#include "TensorParallel.tpp"

#endif