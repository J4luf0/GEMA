#ifndef LINEAR_CONTAINER_HPP
#define LINEAR_CONTAINER_HPP

#include <memory>

#include "MemoryBackend.hpp"
#include "MemoryBackendConcept.hpp"

namespace gema{

template<class T,
         MemoryBackendConcept<T> IMemoryBackend = MemoryBackend<T>//,
         //class A = AlignedAllocator<T, 64>/*std::allocator<T>*/
>
class LinearContainer{

public:

    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:

    // T* data_ = nullptr;
    // size_t size_ = 0;
    // size_t capacity_ = 0;

    T* begin_ = nullptr;
    T* end_   = nullptr;
    T* capEnd_= nullptr;

    IMemoryBackend memoryBackend_;

    //[[no_unique_address]] A alloc_;

public:

    LinearContainer() requires std::default_initializable<IMemoryBackend>;
    LinearContainer(const IMemoryBackend& memoryBackend);
    explicit LinearContainer(size_t n) requires std::default_initializable<IMemoryBackend>;
    LinearContainer(size_t n, const IMemoryBackend& memoryBackend);
    LinearContainer(std::initializer_list<T> init) requires std::default_initializable<IMemoryBackend>;
    //LinearContainer(const LinearContainer<T, IMemoryBackend>& other, const IMemoryBackend& memoryBackend);
    //LinearContainer(const std::vector<T>& init) requires std::default_initializable<IMemoryBackend>;
    //LinearContainer(std::span<const T> s) requires std::default_initializable<IMemoryBackend>;
    //LinearContainer(std::span<const T> s, const IMemoryBackend& memoryBackend);

    
    template<MemoryBackendConcept<T> IOtherMemoryBackend>
    LinearContainer(const LinearContainer<T, IOtherMemoryBackend>& other, const IMemoryBackend& memoryBackend);

    LinearContainer(const LinearContainer<T, IMemoryBackend>& other);
    LinearContainer(LinearContainer<T, IMemoryBackend>&& other) noexcept;

    LinearContainer<T, IMemoryBackend>& operator=(const LinearContainer<T, IMemoryBackend>& other);
    LinearContainer<T, IMemoryBackend>& operator=(LinearContainer<T, IMemoryBackend>&& other) noexcept;

    operator std::span<T>();
    operator std::span<const T>() const;

    ~LinearContainer();

    template <MemoryBackendConcept<T> DestBackend>
    LinearContainer<T, DestBackend> copyToBackend(const DestBackend& destBackend) const;

    IMemoryBackend getMemoryBackend() const;

    //LinearContainer<T, MemoryBackend<T>> copyToHost() requires (!std::is_same_v<IMemoryBackend, MemoryBackend<T>>);
    //LinearContainer<T, MemoryBackend<T>> copyFromHost() requires (!std::is_same_v<IMemoryBackend, MemoryBackend<T>>);


    void reserve(size_t n);
    void resize(size_t n);
    void clear();

    void push_back(const T& value);
    void pop_back();
    void swap(LinearContainer<T, IMemoryBackend>& other) noexcept;
    iterator insert(iterator pos, const T& value);
    iterator erase(iterator pos);
    

    void fill(const T& value);

    void assign(size_t count, const T& value);
    template<class I> void assign(I first, I last);
    void assign(std::initializer_list<T> ilist);

    bool operator==(const LinearContainer<T, IMemoryBackend>& other) const;
    auto operator<=>(const LinearContainer<T, IMemoryBackend>& other) const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    T* data();
    const T* data() const;

    T& front();
    const T& front() const;

    T& back();
    const T& back() const;

    size_t size() const;
    size_t capacity() const;

    // Iterators (zero-cost)
    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    reverse_iterator rbegin();
    reverse_iterator rend();

    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;

private:

    void push_back_slow(const T &value);
    void fastFill(T* dst, size_t count, const T &value);
};

} // end gema

#include "LinearContainer.tpp"

#endif