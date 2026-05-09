#ifndef MEMORY_BACKEND_HPP
#define MEMORY_BACKEND_HPP

#include <cstddef>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace gema {

template<class T, size_t Alignment = 64>
class MemoryBackend {

    public:

    template<typename U>
    using type = MemoryBackend<U>;
    using value_type = T;

    MemoryBackend();

    MemoryBackend(const MemoryBackend<T, Alignment>& memoryBackend);
    MemoryBackend(MemoryBackend<T, Alignment>&& memoryBackend) noexcept;
    MemoryBackend<T, Alignment>& operator=(const MemoryBackend<T, Alignment>& memoryBackend);
    MemoryBackend<T, Alignment>& operator=(MemoryBackend<T, Alignment>&& memoryBackend) noexcept;

    T* allocate(size_t n) const;
    void deallocate(T* pos, size_t n) const;

    void construct_at(T* pos, const T& value) const;
    void destroy_at(T* pos) const;
    void destroy(T* first, T* last) const;

    T* uninitialized_copy(const T* first, const T* last, T* dest) const;
    //iterator uninitialized_copy(const_iterator first, const_iterator last, iterator dest);

    T* uninitialized_move(T* first, T* last, T* dest) const;
    void uninitialized_default_construct(T* first, T* last) const;
    T* uninitialized_fill_n(T* dest, size_t count, const T& value) const;

    void copy(T* dest, const T* src, size_t count) const;
    T* memory_set(T* dest, size_t ch, size_t count ) const;
    int compare(const T* a, const T* b, size_t count) const;

    void copy_to_host(T* dest, const T* src, size_t count) const;
};

}

#include "MemoryBackend.tpp"

#endif