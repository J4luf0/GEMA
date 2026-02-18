#ifndef LINEAR_CONTAINER_HPP
#define LINEAR_CONTAINER_HPP

#include <memory>

namespace gema{

template<class T, class A = std::allocator<T>>
class LinearContainer{

public:
    //using value_type = T;
    //using allocator_type = A;

    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:

    T* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
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
    iterator begin() { return data_; }
    iterator end()   { return data_ + size_; }

    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + size_; }

    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend()   { return reverse_iterator(begin()); }

    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(begin()); }
};

} // end gema

#include "LinearContainer.tpp"

#endif