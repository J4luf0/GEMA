#ifndef TENSOR_CONCEPTS_HPP
#define TENSOR_CONCEPTS_HPP

#include <cstdint>
#include <type_traits>
#include <vector>

namespace gema {


// Forward declaration for the concepts.
template<class T>
class Tensor;



// Concept that checks if given type is tensor or derivate of it.
template <typename Type, class T>
concept TensorType = std::is_same_v<Type, Tensor<T>> || std::derived_from<Type, Tensor<T>>;


// template <typename C, typename A, typename B, class T>
// concept apply_callable = (std::is_same_v<A, T> && std::is_invocable_r_v<void, C, const T&, T&>) ||
//                          (std::is_same_v<B, T> && std::is_invocable_r_v<void, C, T&, const T&>) || 
//                          ((!std::is_same_v<A, T> && !std::is_same_v<B, T>) && std::is_invocable_r_v<void, C, T&, const T&>);

// Tensor computation callables

/// Checks for void(T&, const T&) invocable signature.
template <typename C, class T>
concept apply_callable = std::is_invocable_r_v<void, C, T&, const T&>;

/// Checks for void(const T&, T&) invocable signature.
template <typename C, class T>
concept apply_reverse_callable = std::is_invocable_r_v<void, C, const T&, T&>;

/// Checks for T(const T&, const T&) invocable signature.
template <typename C, class T>
//concept apply_and_return_callable = std::is_invocable_r_v<T, C, const T&, const T&>;
concept apply_and_return_callable = std::is_invocable_v<C, const T&, const T&>;

/// Checks for void(T&) invocable signature.
template <typename C, class T>
concept foreach_callable = std::is_invocable_r_v<void, C, T&>;

/// Checks for void(T&, const std::vector<uint64_t>&) invocable signature.
template <typename C, class T>
concept foreach_coord_callable = std::is_invocable_r_v<void, C, T&, const std::vector<uint64_t>&>;

// Checks for T(const T&) invocable signature.
template <typename C, class T>
//concept foreach_and_return_callable = std::is_invocable_r_v<T, C, const T&>;
concept foreach_and_return_callable = std::is_invocable_v<C, const T&>;


/// Checks for void(T&, const T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept apply_callable_parallel = apply_callable<C, T> && std::is_trivially_copyable_v<C>;

/// Checks for void(const T&, T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept apply_reverse_callable_parallel = apply_reverse_callable<C, T> && std::is_trivially_copyable_v<C>;

/// Checks for T(const T&, const T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept apply_and_return_callable_parallel = apply_and_return_callable<C, T> && std::is_trivially_copyable_v<C>;

/// Checks for void(T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept foreach_callable_parallel = foreach_callable<C, T> && std::is_trivially_copyable_v<C>;

/// Checks for void(T&, const std::vector<uint64_t>&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept foreach_coord_callable_parallel = foreach_coord_callable<C, T> && std::is_trivially_copyable_v<C>;

// Checks for T(const T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept foreach_and_return_callable_parallel = foreach_and_return_callable<C, T> && std::is_trivially_copyable_v<C>;


template <template <typename> class TensorT, class T>
concept tensor_computation_interface = requires (TensorT<T> tensor, T t){
    {true};
};




}



#endif