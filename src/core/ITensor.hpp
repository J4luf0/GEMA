#ifndef TENSOR_PARALLEL_HPP
#define TENSOR_PARALLEL_HPP

#include <concepts>
#include <cstdint>
#include <vector>
#include "LinearContainer.hpp"

namespace gema{

template <typename I, typename T>
concept ITensor = requires(I tensor) {
    requires std::constructible_from<I, const std::vector<uint64_t>&>;
    requires std::copy_constructible<T>;
    requires std::move_constructible<T>;
    requires std::default_initializable<I>;
    { tensor.getItem(std::declval<const std::vector<uint64_t>&>()) } -> std::convertible_to<T&>;
    { tensor.setItem(std::declval<const T&>()) };
    { tensor.getData() } -> std::same_as<LinearContainer<T>&>;
    { tensor.setData(std::declval<const LinearContainer<T>&>()) };
    { tensor.resize(std::declval<const std::vector<uint64_t>&>()) };
    { tensor.addDimension(std::declval<const uint64_t&>(), std::declval<const uint64_t&>()) };
    { tensor.removeDimension(std::declval<const uint64_t&>()) };
    { I::applyAndReturn(std::declval<const I&>(), std::declval<const T&>(), std::declval<T(const T&, const T&)>()) };
    { I::applyAndReturn(std::declval<const T&>(), std::declval<const I&>(), std::declval<T(const T&, const T&)>()) };
    { I::applyAndReturn(std::declval<const I&>(), std::declval<const I&>(), std::declval<T(const T&, const T&)>()) };
    { I::apply(std::declval<I&>(), std::declval<const T&>(), std::declval<void(T&, const T&)>()) } -> std::same_as<void>;
    { I::apply(std::declval<T&>(), std::declval<const I&>(), std::declval<void(T&, const T&)>()) } -> std::same_as<void>;
    { I::apply(std::declval<I&>(), std::declval<const I&>(), std::declval<void(T&, const T&)>()) } -> std::same_as<void>;
    { I::forEachAndReturn(std::declval<const I&>(), std::declval<T(const T&)>()) };
    { I::forEach(std::declval<I&>(), std::declval<void(T&)>()) } -> std::same_as<void>;
};

}

#endif