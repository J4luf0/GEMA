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
            std::memcpy(begin_, init.begin(), init.size() * sizeof(T));
            end_ = begin_ + init.size();
        } else {
            for(const T& v : init){
                push_back(v);
            }
        }
    }

    template<class T, class A>
    LinearContainer<T, A>::LinearContainer(const LinearContainer& other) {

        size_t n = other.size();
        reserve(n);

        if constexpr(std::is_trivially_copyable_v<T>) {
            std::memcpy(begin_, other.begin_, n * sizeof(T));
            end_ = begin_ + n;
        } else {
            for(size_t i = 0; i < n; ++i){
                std::allocator_traits<A>::construct(alloc_, begin_ + i, other.begin_[i]);
            }
            end_ = begin_ + n;
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
        size_t n = other.size();
        reserve(n);

        if constexpr(std::is_trivially_copyable_v<T>) {
            std::memcpy(begin_, other.begin_, n * sizeof(T));
            end_ = begin_ + n;
        } else {
            for(size_t i = 0; i < n; ++i){
                std::allocator_traits<A>::construct(alloc_, begin_ + i, other.begin_[i]);
            }
            end_ = begin_ + n;
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

        T* oldBegin = begin_;
        size_t oldSize = size();

        T* newData = std::allocator_traits<A>::allocate(alloc_, n);

        if(oldBegin){

            if constexpr(std::is_trivially_copyable_v<T>){
                std::memcpy(newData, oldBegin, oldSize * sizeof(T));
            }else {
                for(size_t i = 0; i < oldSize; ++i) {
                    std::allocator_traits<A>::construct(alloc_, newData+i, std::move(oldBegin[i]));
                    std::allocator_traits<A>::destroy(alloc_, oldBegin+i);
                }
            }

            std::allocator_traits<A>::deallocate(alloc_, oldBegin, capacity());
        }

        begin_ = newData;
        end_   = newData + oldSize;
        capEnd_= newData + n;

    }

    template<class T, class A>
    void LinearContainer<T, A>::resize(size_t n) {

        size_t oldSize = size();

        if(n > capacity()){
            reserve(n);
        }

        end_ = begin_ + n;

        if constexpr(!std::is_trivially_default_constructible_v<T>) {
            for(size_t i = oldSize; i < n; ++i){
                std::allocator_traits<A>::construct(alloc_, begin_ + i);
            }
        }
    }

    template<class T, class A>
    void LinearContainer<T, A>::clear() {

        if(begin_) {
            if constexpr(!std::is_trivially_destructible_v<T>) {
                for(T* p = begin_; p != end_; ++p){
                    std::allocator_traits<A>::destroy(alloc_, p);
                }
            }

            std::allocator_traits<A>::deallocate(alloc_, begin_, capacity());
        }

        begin_ = end_ = capEnd_ = nullptr;
    }

    template<class T, class A>
    void LinearContainer<T, A>::push_back(const T& value) {

        T* pos = end_;
        end_ = pos + 1;

        if(end_ <= capEnd_) [[likely]]{

            if constexpr(std::is_trivially_copyable_v<T>){
                *pos = value;
            }else{
                std::allocator_traits<A>::construct(alloc_, pos, value);
            }

            return;
        }

        // slow path (rare)
        end_ = pos; // rollback
        push_back_slow(value);
    }

    template<class T, class A>
    void LinearContainer<T, A>::pop_back() {

        --end_;
        if constexpr(!std::is_trivially_destructible_v<T>){
            std::allocator_traits<A>::destroy(alloc_, end_);
        }
    }

    template<class T, class A>
    void LinearContainer<T, A>::swap(LinearContainer& other) noexcept {
        std::swap(begin_, other.begin_);
        std::swap(end_, other.end_);
        std::swap(capEnd_, other.capEnd_);
    }

    template<class T, class A>
    void LinearContainer<T, A>::assign(size_t count, const T& value){

        if(count > capacity()){
            reserve(count);
        }

        if constexpr(!std::is_trivially_destructible_v<T>) {
            if(count != size()){
                for(T* p = begin_; p != end_; ++p){
                    std::allocator_traits<A>::destroy(alloc_, p);
                }
            }
        }

        if constexpr(std::is_trivially_copyable_v<T>) {
            if(value == T{}){
                std::memset(begin_, 0, count * sizeof(T));
            }else{
                for(size_t i = 0; i < count; ++i) begin_[i] = value;
            }
        } else {
            for(size_t i = 0; i < count; ++i){
                std::allocator_traits<A>::construct(alloc_, begin_ + i, value);
            }
        }

        end_ = begin_ + count;
    }

    template<class T, class A>
    template<class I>
    void LinearContainer<T, A>::assign(I first, I last){

        size_t count = static_cast<size_t>(std::distance(first, last));

        if(count > capacity()){
            reserve(count);
        }

        if constexpr(!std::is_trivially_destructible_v<T>) {
            if(count != size()){
                for(T* p = begin_; p != end_; ++p){
                    std::allocator_traits<A>::destroy(alloc_, p);
                }
            }
        }

        if constexpr(std::is_pointer_v<I> && std::is_trivially_copyable_v<T>){
            std::memcpy(begin_, first, count * sizeof(T));
        }else {
            size_t i = 0;
            for(auto it = first; it != last; ++it, ++i){
                std::allocator_traits<A>::construct(alloc_, begin_ + i, *it);
            }
        }

        end_ = begin_ + count;
    }

    template<class T, class A>
    void LinearContainer<T, A>::assign(std::initializer_list<T> ilist){
        assign(ilist.begin(), ilist.end());
    }

    template<class T, class A>
    bool LinearContainer<T, A>::operator==(const LinearContainer& other) const {

        size_t n = size();
        if(n != other.size()) return false;

        if constexpr(std::is_trivially_copyable_v<T>){
            return std::memcmp(begin_, other.begin_, n * sizeof(T)) == 0;
        }

        for(size_t i = 0; i < n; ++i){
            if(!(begin_[i] == other.begin_[i])) return false;
        }

        return true;
    }

    template<class T, class A>
    auto LinearContainer<T, A>::operator<=>(const LinearContainer& other) const {

        size_t n = std::min(size(), other.size());

        for(size_t i = 0; i < n; ++i){
            if(auto cmp = begin_[i] <=> other.begin_[i]; cmp != 0){
                return cmp;
            }
        }

        return size() <=> other.size();
    }

    template<class T, class A>
    T& LinearContainer<T, A>::operator[](size_t i) { 
        return *(begin_ + i);
    }

    template<class T, class A>
    const T& LinearContainer<T, A>::operator[](size_t i) const { 
        return *(begin_ + i);
    }

    template <class T, class A>
    T* LinearContainer<T, A>::data(){
        return begin_;
    }

    template <class T, class A>
    const T* LinearContainer<T, A>::data() const {
        return begin_;
    }

    template <class T, class A>
    T& LinearContainer<T, A>::front(){
        return *begin_;
    }

    template <class T, class A>
    const T& LinearContainer<T, A>::front() const{
        return *begin_;
    }

    template <class T, class A>
    T& LinearContainer<T, A>::back(){
        return *(end_-1);
    }

    template <class T, class A>
    const T& LinearContainer<T, A>::back() const{
        return *(end_-1);
    }

    template <class T, class A>
    size_t LinearContainer<T, A>::size() const{
        return static_cast<size_t>(end_ - begin_);
    }

    template <class T, class A>
    size_t LinearContainer<T, A>::capacity() const{
        return static_cast<size_t>(capEnd_ - begin_);
    }

    template <class T, class A>
    LinearContainer<T, A>::iterator LinearContainer<T, A>::begin(){
        return begin_;
    }

    template <class T, class A>
    LinearContainer<T, A>::iterator LinearContainer<T, A>::end(){
        return end_;
    }

    template <class T, class A>
    LinearContainer<T, A>::const_iterator LinearContainer<T, A>::begin() const{
        return begin_;
    }

    template <class T, class A>
    LinearContainer<T, A>::const_iterator LinearContainer<T, A>::end() const{
        return end_;
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


    template<class T, class A>
    void LinearContainer<T,A>::push_back_slow(const T& value){

        size_t oldSize = size();
        size_t newCap  = capacity() ? (capacity() + (capacity() / 2) + 8) : 8;

        reserve(newCap);

        T* pos = begin_ + oldSize;

        if constexpr(std::is_trivially_copyable_v<T>){
            *pos = value;
        }else{
            std::allocator_traits<A>::construct(alloc_, pos, value);
        }
            
        end_ = pos + 1;
    }

    template<class T, class A>
    inline void LinearContainer<T,A>::fastFill(T* dst, size_t count, const T& value){

        if constexpr(std::is_trivially_copyable_v<T>){

            if(value == T{}){
                std::memset(dst, 0, count * sizeof(T));
                return;
            }

            // SIMD-friendly unrolled fill
            size_t i = 0;

            constexpr size_t UNROLL = 8;

            for(; i + UNROLL <= count; i += UNROLL){

                dst[i+0] = value;
                dst[i+1] = value;
                dst[i+2] = value;
                dst[i+3] = value;
                dst[i+4] = value;
                dst[i+5] = value;
                dst[i+6] = value;
                dst[i+7] = value;
            }

            for(; i < count; ++i){
                dst[i] = value;
            }
            
        }else{
            for(size_t i=0;i<count;++i){
                std::allocator_traits<A>::construct(alloc_, dst+i, value);
            }
        }
    }
}