#ifndef ALIGNED_ALLOCATOR_HPP
#define ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>

template<class T, std::size_t Alignment = 64>
struct AlignedAllocator {

    using value_type = T;

    AlignedAllocator() noexcept = default;

    template<class U>
    constexpr AlignedAllocator(const AlignedAllocator<U,Alignment>&) noexcept {}

    [[nodiscard]]
    T* allocate(std::size_t n){

        if(n == 0) return nullptr;

        std::size_t bytes = n * sizeof(T);

        void* ptr = ::operator new(bytes, std::align_val_t{Alignment});

        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept{
        ::operator delete(p, std::align_val_t{Alignment});
    }

    template<class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template<class T1, size_t A1, class T2, size_t A2>
constexpr bool operator==(const AlignedAllocator<T1,A1>&, const AlignedAllocator<T2,A2>&) noexcept {
    return A1 == A2;
}

template<class T1, size_t A1, class T2, size_t A2>
constexpr bool operator!=(const AlignedAllocator<T1,A1>& a, const AlignedAllocator<T2,A2>& b) noexcept {
    return !(a == b);
}

#endif