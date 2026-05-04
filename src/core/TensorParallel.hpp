#ifndef TENSOR_PARALLEL_HPP
#define TENSOR_PARALLEL_HPP

#include <sycl/sycl.hpp>

#include "TensorConcept.hpp"
#include "Tensor.hpp"

namespace gema{

template<class T>
class TensorParallel;

// Concept that checks if type X is of type T or TensorParallel<T>. Useful for operator overloads.
template <typename X, class T>
concept is_tensorparallel_or_t = std::is_same_v<X, T> || std::is_same_v<X, TensorParallel<T>>;

template <typename A, typename B, class T>
concept tensor_or_t_or_bothtensor_parallel = 
    (std::is_same_v<std::remove_cvref_t<A>, TensorParallel<T>> && std::is_same_v<std::remove_cvref_t<B>, TensorParallel<T>>) ||
    (std::is_same_v<std::remove_cvref_t<A>, TensorParallel<T>> && std::is_same_v<std::remove_cvref_t<B>, T>) ||
    (std::is_same_v<std::remove_cvref_t<A>, T> && std::is_same_v<std::remove_cvref_t<B>, TensorParallel<T>>);


template<class T>
class TensorParallel : public Tensor<T>, public AbstractOperation<TensorParallel, T>{

    inline static sycl::queue queueGlobal_{sycl::property::queue::in_order{}};

    sycl::queue* queue_ = &queueGlobal_;

    constexpr static sycl::usm::alloc usmKind_ = sycl::usm::alloc::device;

    public:

    TensorParallel(const LinearContainer<uint64_t>& newTensorDimensionSizes);

    TensorParallel(const LinearContainer<uint64_t>&, const LinearContainer<T>&) = delete;

    TensorParallel(const TensorParallel<T>& otherTensor);

    TensorParallel(TensorParallel<T>&& otherTensor) noexcept;

    TensorParallel(const TensorParallel<T>* otherTensor);

    TensorParallel();

    TensorParallel<T>& operator=(const TensorParallel<T>& otherTensor);
    
    TensorParallel<T>& operator=(TensorParallel<T>&& otherTensor) noexcept;

    TensorParallel<T> transpositionAndReturn(const uint64_t dim1 = 0, const uint64_t dim2 = 1) const;

    void transposition(const uint64_t dim1 = 0, const uint64_t dim2 = 1);

    void resize(const LinearContainer<uint64_t>& newDimensionSizes);

    void resize(const uint64_t newDimensionSize, const uint64_t dimensionIndex);
    
    void addDimension(const uint64_t newDimensionSize, const uint64_t putBefore);

    void removeDimension(const uint64_t removedDimensionIndex);

    template <apply_and_return_callable_parallel<T> C>
    auto applyAndReturn(const TensorParallel<T>& tensor2, C&& operation) const;
    
    template <typename A, typename B, apply_and_return_callable_parallel<T> C> 
    static auto applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(tensor_or_t_or_bothtensor_parallel<A, B, T>);

    template <apply_callable_parallel<T> C>
    void apply(const TensorParallel<T>& tensor2, C&& operation);

    // template <typename A, typename B, apply_callable_parallel<T> C> 
    // static void apply(A& operand1, const B& operand2, C&& operation)
    // requires(tensor_or_t_or_bothtensor_parallel<A, B, T>);

    template <apply_callable_parallel<T> C>
    static void apply(TensorParallel<T>& operand1, const TensorParallel<T>& operand2, C&& operation);

    template <apply_callable_parallel<T> C>
    static void apply(TensorParallel<T>& operand1, const T& operand2, C&& operation);

    template <apply_reverse_callable_parallel<T> C>
    static void apply(const T& operand1, TensorParallel<T>& operand2, C&& operation);


    template <foreach_and_return_callable_parallel<T> C>
    auto forEachAndReturn(C&& operation) const;

    template <foreach_and_return_callable_parallel<T> C>
    static auto forEachAndReturn(const TensorParallel<T>& tensor, C&& operation);

    template <foreach_callable_parallel<T> C> 
    void forEach(C&& operation);

    template <foreach_callable_parallel<T> C>
    static void forEach(TensorParallel<T>& tensor, C&& operation);

};

}

#include "TensorParallel.tpp"

#endif