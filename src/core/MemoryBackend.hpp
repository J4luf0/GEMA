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

    MemoryBackend();

    MemoryBackend(const MemoryBackend<T, Alignment>& memoryBackend);
    MemoryBackend(MemoryBackend<T, Alignment>&& memoryBackend) noexcept;
    MemoryBackend<T, Alignment>& operator=(const MemoryBackend<T, Alignment>& memoryBackend);
    MemoryBackend<T, Alignment>& operator=(MemoryBackend<T, Alignment>&& memoryBackend) noexcept;

    T* allocate(size_t n);
    void deallocate(T* pos, size_t n);

    void construct_at(T* pos, const T& value);
    void destroy_at(T* pos);
    void destroy(T* first, T* last);

    T* uninitialized_copy(const T* first, const T* last, T* dest);
    T* uninitialized_move(T* first, T* last, T* dest);
    void uninitialized_default_construct(T* first, T* last);
    T* uninitialized_fill_n(T* dest, size_t count, const T& value);

    void copy(T* dest, const T* src, size_t count);
    T* memory_set(T* dest, size_t ch, size_t count );
    int compare(const T* a, const T* b, size_t count);
};

}

#include "MemoryBackend.tpp"

#endif