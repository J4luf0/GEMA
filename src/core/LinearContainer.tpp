#include <cstdint>
#include <cstring>
#include "LinearContainer.hpp"

namespace gema{

    template<class T, class A>
    LinearContainer<T, A>::LinearContainer() {

    }

    template<class T, class A>
    LinearContainer<T, A>::LinearContainer(size_t n) {
        resize(n);
    }

    template<class T, class A>
    LinearContainer<T, A>::LinearContainer(std::initializer_list<T> init) {

        reserve(init.size());

        if constexpr(std::is_trivially_copyable_v<T>) {
            std::memcpy(data_, init.begin(), init.size() * sizeof(T));
            size_ = init.size();
        } else {
            for(const T& v : init){
                push_back(v);
            }
        }
    }

    template<class T, class A>
    LinearContainer<T, A>::LinearContainer(const LinearContainer& other) {

        reserve(other.size_);

        if constexpr(std::is_trivially_copyable_v<T>) {
            std::memcpy(data_, other.data_, other.size_ * sizeof(T));
            size_ = other.size_;
        } else {
            for(size_t i = 0; i < other.size_; ++i){
                std::allocator_traits<A>::construct(alloc_, data_ + i, other.data_[i]);
            }
            size_ = other.size_;
        }
    }

    template<class T, class A>
    LinearContainer<T, A>::LinearContainer(LinearContainer&& other) noexcept {
        swap(other);
    }

    template<class T, class A>
    LinearContainer<T, A>& LinearContainer<T, A>::operator=(const LinearContainer& other) {

        if(this == &other) return *this;
        clear();
        reserve(other.size_);

        if constexpr(std::is_trivially_copyable_v<T>) {
            std::memcpy(data_, other.data_, other.size_ * sizeof(T));
            size_ = other.size_;
        } else {
            for(size_t i = 0; i < other.size_; ++i){
                std::allocator_traits<A>::construct(alloc_, data_ + i, other.data_[i]);
            }
            size_ = other.size_;
        }

        return *this;
    }

    template<class T, class A>
    LinearContainer<T, A>& LinearContainer<T, A>::operator=(LinearContainer&& other) noexcept {
        swap(other);
        return *this;
    }

    template<class T, class A>
    LinearContainer<T, A>::~LinearContainer() {
        clear();
    }

    template<class T, class A>
    void LinearContainer<T, A>::reserve(size_t n) {

        if(n<=capacity_) return;

        T* newData = std::allocator_traits<A>::allocate(alloc_, n);

        if(data_) {
            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(newData, data_, size_*sizeof(T));
            } else {
                for(size_t i=0;i<size_;++i) {
                    std::allocator_traits<A>::construct(alloc_, newData+i, std::move(data_[i]));
                    std::allocator_traits<A>::destroy(alloc_, data_+i);
                }
            }
            std::allocator_traits<A>::deallocate(alloc_, data_, capacity_);
        }

        data_ = newData;
        capacity_ = n;
    }

    template<class T, class A>
    void LinearContainer<T, A>::resize(size_t n) {

        if(n>capacity_) reserve(n);
        size_ = n; // NO initialization
    }

    template<class T, class A>
    void LinearContainer<T, A>::clear() {

        if(data_) {
            if constexpr(!std::is_trivially_destructible_v<T>) {
                for(size_t i=0; i < size_; ++i){
                    std::allocator_traits<A>::destroy(alloc_, data_ + i);
                }
            }
            std::allocator_traits<A>::deallocate(alloc_, data_, capacity_);
        }

        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    template<class T, class A>
    void LinearContainer<T, A>::push_back(const T& value) {

        // branch-minimized growth
        size_t newCap = (capacity_) ? (capacity_ * 2) : 8;
        if(size_==capacity_){
            reserve(newCap);
        }

        std::allocator_traits<A>::construct(alloc_, data_ + size_, value);
        ++size_;
    }

    template<class T, class A>
    void LinearContainer<T, A>::pop_back() {

        --size_;
        if constexpr(!std::is_trivially_destructible_v<T>){
            std::allocator_traits<A>::destroy(alloc_, data_+size_);
        }
    }

    template<class T, class A>
    void LinearContainer<T, A>::swap(LinearContainer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    template<class T, class A>
    bool LinearContainer<T, A>::operator==(const LinearContainer& other) const {

        if(size_!=other.size_) return false;

        if constexpr(std::is_trivially_copyable_v<T>) {
            return std::memcmp(data_, other.data_, size_*sizeof(T))==0;
        } else {
            for(size_t i = 0; i < size_; ++i){
                if(!(data_[i]==other.data_[i])) return false;
            }
            return true;
        }
    }

    template<class T, class A>
    auto LinearContainer<T, A>::operator<=>(const LinearContainer& other) const {

        const size_t n = std::min(size_, other.size_);
        for(size_t i = 0; i < n; ++i) {
            if(auto cmp = data_[i] <=> other.data_[i]; cmp != 0){
                return cmp;
            }
        }

        return size_ <=> other.size_;
    }

    template<class T, class A>
    T& LinearContainer<T, A>::operator[](size_t i) { 
        return data_[i];
    }

    template<class T, class A>
    const T& LinearContainer<T, A>::operator[](size_t i) const { 
        return data_[i];
    }

    template <class T, class A>
    T* LinearContainer<T, A>::data(){
        return data_;
    }

    template <class T, class A>
    const T* LinearContainer<T, A>::data() const {
        return data_;
    }

    template <class T, class A>
    T& LinearContainer<T, A>::front(){
        return data_[0];
    }

    template <class T, class A>
    const T& LinearContainer<T, A>::front() const{
        return data_[0];
    }

    template <class T, class A>
    T& LinearContainer<T, A>::back(){
        return data_[size_ - 1];
    }

    template <class T, class A>
    const T& LinearContainer<T, A>::back() const{
        return data_[size_ - 1];
    }

    template <class T, class A>
    size_t LinearContainer<T, A>::size() const{
        return size_;
    }

    template <class T, class A>
    size_t LinearContainer<T, A>::capacity() const{
        return capacity_;
    }

    template <class T, class A>
    LinearContainer<T, A>::iterator LinearContainer<T, A>::begin(){
        return data_;
    }

    template <class T, class A>
    LinearContainer<T, A>::iterator LinearContainer<T, A>::end(){
        return data_ + size_;
    }

    template <class T, class A>
    LinearContainer<T, A>::const_iterator LinearContainer<T, A>::begin() const{
        return data_;
    }

    template <class T, class A>
    LinearContainer<T, A>::const_iterator LinearContainer<T, A>::end() const{
        return data_ + size_;
    }

    template <class T, class A>
    LinearContainer<T, A>::reverse_iterator LinearContainer<T, A>::rbegin(){
        return reverse_iterator(end());
    }

    template <class T, class A>
    LinearContainer<T, A>::reverse_iterator LinearContainer<T, A>::rend(){
        return reverse_iterator(begin());
    }

    template <class T, class A>
    LinearContainer<T, A>::const_reverse_iterator LinearContainer<T, A>::rbegin() const{
        return const_reverse_iterator(end());
    }

    template <class T, class A>
    LinearContainer<T, A>::const_reverse_iterator LinearContainer<T, A>::rend() const{
        return const_reverse_iterator(begin());
    }
}