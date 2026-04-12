#ifndef TENSOR_PARALLEL_HPP
#define TENSOR_PARALLEL_HPP

#include "Tensor.hpp"

namespace gema{

template<class T>
class TensorParallel : Tensor<T>{

    public:

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