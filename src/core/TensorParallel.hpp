#ifndef TENSOR_PARALLEL_HPP
#define TENSOR_PARALLEL_HPP

#include <sycl/sycl.hpp>

#include "MemoryBackendConcept.hpp"
#include "MemoryBackendUSM.hpp"
#include "TensorConcept.hpp"
#include "Tensor.hpp"

namespace gema{

template<class T>//, MemoryBackendConcept<T> IMemoryBackend = MemoryBackendUSM<T, sycl::usm::alloc::device>
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
class TensorParallel : /*public Tensor<T>,*/public AbstractOperation<TensorParallel<T>>{

    

    inline static sycl::queue queueGlobal_{sycl::property::queue::in_order{}};

    constexpr static sycl::usm::alloc usmDataKind_ = sycl::usm::alloc::device;
    constexpr static sycl::usm::alloc usmMetadataKind_ = sycl::usm::alloc::shared;

    sycl::queue* queue_ = &queueGlobal_;

    Tensor<T, MemoryBackendUSM<T, usmDataKind_>, MemoryBackendUSM<uint64_t, usmMetadataKind_>> tensor_;

    public:

    template<typename U>
    using type = TensorParallel<U>;

    using value_type = T;
    using memory_backend = MemoryBackendUSM<T, sycl::usm::alloc::device>;


    TensorParallel(const LinearContainer<uint64_t>& newTensorDimensionSizes);

    TensorParallel(const LinearContainer<uint64_t>&, const LinearContainer<T>&) = delete;

    TensorParallel(const TensorParallel<T>& otherTensor);

    TensorParallel(TensorParallel<T>&& otherTensor) noexcept;

    TensorParallel(const TensorParallel<T>* otherTensor);

    TensorParallel();

    TensorParallel<T>& operator=(const TensorParallel<T>& otherTensor);
    
    TensorParallel<T>& operator=(TensorParallel<T>&& otherTensor) noexcept;



    const LinearContainer<uint64_t>& getDimensionSizes() const;

    uint64_t getNumberOfDimensions() const;

    uint64_t getNumberOfItems() const;



    T& getItem(const LinearContainer<uint64_t>& coordinates);

    void setItem(const T& value, const LinearContainer<uint64_t>& coordinates);

    T* getData();
    const T* getData() const;

    TensorParallel<T>& setData(const LinearContainer<T>& tensorItems);


    bool isValidCoordinates(const LinearContainer<uint64_t>& coords) const;

    static bool isValidCoordinates(const LinearContainer<uint64_t>& coords, const LinearContainer<uint64_t>& dimensionSizes);

    bool isEquilateral() const;

    bool operator==(const TensorParallel<T>& otherTensor) const;

    bool operator!=(const TensorParallel<T>& otherTensor) const;




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