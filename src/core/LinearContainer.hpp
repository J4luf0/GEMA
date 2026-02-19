#ifndef LINEAR_CONTAINER_HPP
#define LINEAR_CONTAINER_HPP

#include "AlignedAllocator.hpp"
#include <memory>

namespace gema{

template<class T, class A = AlignedAllocator<T, 64>/*std::allocator<T>*/>
class LinearContainer{

public:

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

    [[no_unique_address]] A alloc_;

public:

    LinearContainer();
    explicit LinearContainer(size_t n);
    LinearContainer(std::initializer_list<T> init);

    LinearContainer(const LinearContainer& other);
    LinearContainer(LinearContainer&& other) noexcept;

    LinearContainer& operator=(const LinearContainer& other);
    LinearContainer& operator=(LinearContainer&& other) noexcept;

    ~LinearContainer();

    void reserve(size_t n);
    void resize(size_t n);
    void clear();

    void push_back(const T &value);
    void pop_back();
    void swap(LinearContainer &other) noexcept;

    void assign(size_t count, const T& value);
    template<class I> void assign(I first, I last);
    void assign(std::initializer_list<T> ilist);

    bool operator==(const LinearContainer &other) const;
    auto operator<=>(const LinearContainer &other) const;

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
    void fastFill(T *dst, size_t count, const T &value);
};

} // end gema

#include "LinearContainer.tpp"

#endif