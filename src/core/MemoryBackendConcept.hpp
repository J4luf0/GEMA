#ifndef MEMORY_BACKEND_CONCEPT_HPP
#define MEMORY_BACKEND_CONCEPT_HPP

#include <concepts>
#include <cstddef>

namespace gema {

template<class B, class T>
concept MemoryBackendConcept = requires(B b, T* p, const T* cp, T value, std::size_t n, std::size_t ch) {

    // CONSTRUCTION

    //{ B() }; // Constructors actually shouldn´t be required
    requires std::copy_constructible<B>;
    requires std::move_constructible<B>;

    // ALLOCATION

    { b.allocate(n) } -> std::same_as<T*>;
    { b.deallocate(p, n) } -> std::same_as<void>;

    // CONSTRUCT / DESTROY

    { b.construct_at(p, value) } -> std::same_as<void>;
    { b.destroy_at(p) } -> std::same_as<void>;
    { b.destroy(p, p) } -> std::same_as<void>;

    // UNINITIALIZED OPS

    { b.uninitialized_copy(cp, cp, p) } -> std::same_as<T*>;
    { b.uninitialized_move(p, p, p) } -> std::same_as<T*>;
    { b.uninitialized_default_construct(p, p) } -> std::same_as<void>;
    { b.uninitialized_fill_n(p, n, value) } -> std::same_as<T*>;

    // RAW MEMORY

    { b.copy(p, cp, n) } -> std::same_as<void>;
    { b.memory_set(p, ch, n) } -> std::same_as<T*>;
    { b.compare(cp, cp, n) } -> std::same_as<int>;
};

}

#endif