#ifndef MEMORY_BACKEND_CONCEPT_HPP
#define MEMORY_BACKEND_CONCEPT_HPP

#include <concepts>
#include <cstddef>

namespace gema {

template<class B, class T>
concept MemoryBackendConcept = requires(B b, const B cb, T* p, const T* cp, T value, std::size_t n, std::size_t ch) {

    // CONSTRUCTION

    //{ B() }; // Constructors actually shouldn´t be required
    requires std::copy_constructible<B>;
    requires std::move_constructible<B>;

    // TYPE

    typename B::template type<T>;
    requires std::same_as<typename B::value_type, T>;

    // ALLOCATION

    { cb.allocate(n) } -> std::same_as<T*>;
    { cb.deallocate(p, n) } -> std::same_as<void>;

    // CONSTRUCT / DESTROY

    { cb.construct_at(p, value) } -> std::same_as<void>;
    { cb.destroy_at(p) } -> std::same_as<void>;
    { cb.destroy(p, p) } -> std::same_as<void>;

    // UNINITIALIZED OPS

    { cb.uninitialized_copy(cp, cp, p) } -> std::same_as<T*>;
    { cb.uninitialized_move(p, p, p) } -> std::same_as<T*>;
    { cb.uninitialized_default_construct(p, p) } -> std::same_as<void>;
    { cb.uninitialized_fill_n(p, n, value) } -> std::same_as<T*>;

    // RAW MEMORY

    { cb.copy(p, cp, n) } -> std::same_as<void>;
    { cb.memory_set(p, ch, n) } -> std::same_as<T*>;
    { cb.compare(cp, cp, n) } -> std::same_as<int>;

    { cb.copy_to_host(p, cp, n) } -> std::same_as<void>;
    { cb.copy_from_host(p, cp, n) } -> std::same_as<void>;
};

}

#endif