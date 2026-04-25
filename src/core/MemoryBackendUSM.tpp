#include <type_traits>
#include <sycl/sycl.hpp>

#include "MemoryBackendUSM.hpp"

//#define T_ALLOC_ALIGN class T, sycl::usm::alloc MemoryType, std::size_t Alignment

namespace gema {

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(const sycl::queue* queue_){
        this->queue_ = queue_;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(const MemoryBackendUSM<T, Kind, Alignment>& otherBackend){
        *this = otherBackend;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(MemoryBackendUSM<T, Kind, Alignment>&& otherBackend) noexcept{
        std::swap(otherBackend.queue_, this->queue_);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>& MemoryBackendUSM<T, Kind, Alignment>::operator=(
    const MemoryBackendUSM<T, Kind, Alignment>& otherBackend){
        queue_ = otherBackend.queue_;
        return *this;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>& MemoryBackendUSM<T, Kind, Alignment>::operator=(
    MemoryBackendUSM<T, Kind, Alignment>&& otherBackend) noexcept{
        queue_ = std::move(otherBackend.queue_);
        return *this;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T *MemoryBackendUSM<T, Kind, Alignment>::allocate(size_t n){

        if(n == 0) return nullptr;

        std::size_t bytes = n * sizeof(T);

        return sycl::aligned_alloc(Alignment, bytes, *queue_, Kind);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::deallocate(T* pos, size_t n){

        sycl::free(pos, *queue_);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::construct_at(T* pos, const T& value){

        if constexpr (Kind == sycl::usm::alloc::device) {
            queue_->submit([&](sycl::handler& h){
                h.single_task([=](){
                    new (pos) T(value);
                });
            }).wait();
        } else {
            std::construct_at(pos, value);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::destroy_at(T* pos){

        if constexpr (Kind == sycl::usm::alloc::device) {
            queue_->submit([&](sycl::handler& h){
                h.single_task([=](){
                    pos->~T();
                });
            }).wait();
        } else {
            std::destroy_at(pos);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::destroy(T* first, T* last){

        if constexpr (Kind == sycl::usm::alloc::device) {
            size_t n = last - first;
            queue_->parallel_for(n, [=](auto i){
                (first + i)->~T();
            }).wait();
        } else {
            std::destroy(first, last);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::uninitialized_copy(const T* first, const T* last, T* dest){

        size_t n = last - first;

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memcpy(dest, first, n * sizeof(T)).wait();

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(dest, first, n * sizeof(T));
            } else {
                std::uninitialized_copy(first, last, dest);
            }
        }

        return dest + n;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::uninitialized_move(T* first, T* last, T* dest){

        size_t n = last - first;

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memcpy(dest, first, n * sizeof(T)).wait();

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(dest, first, n * sizeof(T));
            } else {
                std::uninitialized_move(first, last, dest);
            }
        }

        return dest + n;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::uninitialized_default_construct(T* first, T* last){

        if constexpr (Kind == sycl::usm::alloc::device) {

            size_t n = last - first;
            queue_->parallel_for(n, [=](auto i){
                new (first + i) T();
            }).wait();

        } else {

            std::uninitialized_default_construct(first, last);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T *MemoryBackendUSM<T, Kind, Alignment>::uninitialized_fill_n(T* dest, size_t count, const T& value){

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->parallel_for(count, [=](auto i){
                new (dest + i) T(value);
            }).wait();

        } else {

            std::uninitialized_fill_n(dest, count, value);
        }

        return dest + count;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::copy(T* dest, const T* src, size_t count){

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memcpy(dest, src, count * sizeof(T)).wait();

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(dest, src, count * sizeof(T));
            } else {
                for(size_t i = 0; i < count; ++i){
                    dest[i] = src[i];
                }
            }
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::memory_set(T* dest, size_t ch, size_t count){

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memset(dest, ch, count * sizeof(T)).wait();

        } else {

            std::memset(dest, ch, count * sizeof(T));
        }

        return dest;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    int MemoryBackendUSM<T, Kind, Alignment>::compare(const T* a, const T* b, size_t count){

        if constexpr (Kind == sycl::usm::alloc::device) {

            // fallback → copy to host (expensive!)
            std::vector<T> tmpA(count), tmpB(count);

            queue_->memcpy(tmpA.data(), a, count * sizeof(T)).wait();
            queue_->memcpy(tmpB.data(), b, count * sizeof(T)).wait();

            return std::memcmp(tmpA.data(), tmpB.data(), count * sizeof(T));

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                return std::memcmp(a, b, count * sizeof(T));
            } else {
                for(size_t i = 0; i < count; ++i){
                    if(a[i] < b[i]) return -1;
                    if(a[i] > b[i]) return 1;
                }
                return 0;
            }
        }
    }
}

#undef T_ALLOC_ALIGN