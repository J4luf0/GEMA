#include <cstring>

#include "MemoryBackend.hpp"

//#define T_ALLOC_ALIGN class T, sycl::usm::alloc MemoryType, size_t Alignment

namespace gema {

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>::MemoryBackend(){

    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>::MemoryBackend(const MemoryBackend<T, Alignment>& memoryBackend){

    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>::MemoryBackend(MemoryBackend<T, Alignment>&& memoryBackend) noexcept{

    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>& MemoryBackend<T, Alignment>::operator=(const MemoryBackend<T, Alignment>& memoryBackend){
        return *this;
    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>& MemoryBackend<T, Alignment>::operator=(MemoryBackend<T, Alignment>&& memoryBackend) noexcept{
        return *this;
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::allocate(size_t n){

        if(n == 0) return nullptr;

        size_t bytes = n * sizeof(T);

        void* ptr = ::operator new(bytes, std::align_val_t{Alignment});
        return static_cast<T*>(ptr);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::deallocate(T* pos, size_t n){

        ::operator delete(pos, std::align_val_t{Alignment});
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::construct_at(T* pos, const T& value){
        std::construct_at(pos, value);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::destroy_at(T* pos){
        std::destroy_at(pos);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::destroy(T* first, T* last){
        std::destroy(first, last);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::uninitialized_copy(const T* first, const T* last, T* dest){
        return std::uninitialized_copy(first, last, dest);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::uninitialized_move(T* first, T* last, T* dest){
        return std::uninitialized_move(first, last, dest);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::uninitialized_default_construct(T* first, T* last){
        std::uninitialized_default_construct(first, last);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::uninitialized_fill_n(T* dest, size_t count, const T& value){
        return std::uninitialized_fill_n(dest, count, value);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::copy(T* dest, const T* src, size_t count){
        return std::memcpy(dest, src, count);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::memory_set(T* dest, size_t ch, size_t count){
        return std::memset(dest, ch, count);
    }

    template <class T, size_t Alignment>
    int MemoryBackend<T, Alignment>::compare(const T* a, const T* b, size_t count){
        return std::memcmp(a, b, count);
    }
}

#undef T_ALLOC_ALIGN