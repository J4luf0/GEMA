#ifndef MEMORY_BACKEND_USM_HPP
#define MEMORY_BACKEND_USM_HPP

#include <cstddef>

#include <sycl/sycl.hpp>

#include "MemoryBackend.hpp"

namespace gema {

template<class T, sycl::usm::alloc Kind, size_t Alignment = 64>
class MemoryBackendUSM : public MemoryBackend<T, Alignment> {

    private:

    const sycl::queue* queue_ = nullptr;

    public:

    MemoryBackendUSM(const sycl::queue* queue_);
    MemoryBackendUSM(const MemoryBackendUSM<T, Kind, Alignment>& memoryBackend);
    MemoryBackendUSM(MemoryBackendUSM<T, Kind, Alignment>&& memoryBackend) noexcept;
    MemoryBackendUSM() = delete;

    MemoryBackendUSM<T, Kind, Alignment>& operator=(const MemoryBackendUSM<T, Kind, Alignment>& memoryBackend);
    MemoryBackendUSM<T, Kind, Alignment>& operator=(MemoryBackendUSM<T, Kind, Alignment>&& memoryBackend) noexcept;

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

#include "MemoryBackendUSM.tpp"

#endif