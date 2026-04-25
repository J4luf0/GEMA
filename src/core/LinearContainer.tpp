#include <cstdint>
#include <cstring>

#include "LinearContainer.hpp"
#include "MemoryBackendConcept.hpp"

namespace gema{

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer() 
    requires std::default_initializable<IMemoryBackend> {

    }

    template <class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer(const IMemoryBackend& memoryBackend)
    : memoryBackend_(memoryBackend){

    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer(size_t n) 
    requires std::default_initializable<IMemoryBackend>{
        resize(n);
    }

    template <class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer(size_t n, const IMemoryBackend& memoryBackend)
    : memoryBackend_(memoryBackend){
        resize(n);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer(std::initializer_list<T> init) 
    requires std::default_initializable<IMemoryBackend> {

        size_t initSize = init.size();
        reserve(initSize);

        std::uninitialized_copy(init.begin(), init.end(), begin_);

        end_ = begin_ + initSize;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer(const LinearContainer& other) 
    : memoryBackend_(other.memoryBackend_){

        size_t otherSize = other.size();
        reserve(otherSize);

        std::uninitialized_copy(other.begin_, other.begin_ + otherSize, begin_);

        end_ = begin_ + otherSize;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::LinearContainer(LinearContainer&& other) noexcept {
        //swap(other);

        begin_  = other.begin_;
        end_    = other.end_;
        capEnd_ = other.capEnd_;
        memoryBackend_ = std::move(other.memoryBackend_);

        other.begin_ = other.end_ = other.capEnd_ = nullptr;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>& LinearContainer<T, IMemoryBackend, A>::operator=(const LinearContainer& other) {

        if(this == &other) return *this;
        
        memoryBackend_ = other.memoryBackend_;

        size_t otherSize = other.size();

        if(otherSize > capacity()){
            clear();
            reserve(otherSize);
        } else {
            if constexpr(!std::is_trivially_destructible_v<T>){
                std::destroy(begin_, end_);
            }
        }

        std::uninitialized_copy(other.begin_, other.begin_ + otherSize, begin_);

        end_ = begin_ + otherSize;

        return *this;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>& LinearContainer<T, IMemoryBackend, A>::operator=(LinearContainer&& other) noexcept {
        swap(other);
        memoryBackend_ = std::move(other.memoryBackend_);
        return *this;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::~LinearContainer() {
        clear();
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::reserve(size_t n) {

        if (n <= capacity())[[unlikely]] return;

        T* oldBegin = begin_;
        size_t oldSize = size();

        T* newData = std::allocator_traits<A>::allocate(alloc_, n);

        if(oldBegin){

            std::uninitialized_move(oldBegin, oldBegin + oldSize, newData);

            if constexpr(!std::is_trivially_destructible_v<T>) {
                std::destroy(oldBegin, oldBegin + oldSize);
            }

            std::allocator_traits<A>::deallocate(alloc_, oldBegin, capacity());
        }

        begin_ = newData;
        end_   = newData + oldSize;
        capEnd_= newData + n;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::resize(size_t n) {

        size_t oldSize = size();

        if(n > capacity()){
            reserve(n);
        }

        if(n > oldSize){

            std::uninitialized_default_construct(begin_ + oldSize, begin_ + n);

        } else if(n < oldSize){

            if constexpr(!std::is_trivially_destructible_v<T>) {
                std::destroy(begin_ + n, begin_ + oldSize);
            }
        }

        end_ = begin_ + n;

    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::clear() {

        if(begin_) {
            if constexpr(!std::is_trivially_destructible_v<T>) {
                std::destroy(begin_, end_);
            }

            std::allocator_traits<A>::deallocate(alloc_, begin_, capacity());
        }

        begin_ = end_ = capEnd_ = nullptr;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::push_back(const T& value) {

        T* pos = end_;

        if(pos < capEnd_) [[likely]]{

            std::construct_at(pos, value);
            ++end_;
            return;
        }

        // // slow path (rare)
        // end_ = pos; // rollback
        push_back_slow(value);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::pop_back() {

        --end_;
        if constexpr(!std::is_trivially_destructible_v<T>){
            std::destroy_at(end_);
            //std::allocator_traits<A>::destroy(alloc_, end_);
        }
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::swap(LinearContainer& other) noexcept {
        std::swap(begin_, other.begin_);
        std::swap(end_, other.end_);
        std::swap(capEnd_, other.capEnd_);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::fill(const T& value){
        fastFill(begin_, size(), value);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::assign(size_t count, const T& value){

        if(count > capacity()){
            reserve(count);
        }

        size_t oldSize = size();
        size_t common = std::min(oldSize, count);

        // assign into existing objects
        fastFill(begin_, common, value);

        if(count > oldSize){
            std::uninitialized_fill_n(begin_ + oldSize, count - oldSize, value);
        }else if(count < oldSize){
            if constexpr(!std::is_trivially_destructible_v<T>){
                std::destroy(begin_ + count, end_);
            }
        }

        end_ = begin_ + count;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    template<class I>
    void LinearContainer<T, IMemoryBackend, A>::assign(I first, I last){

        size_t count = static_cast<size_t>(std::distance(first, last));

        if(count > capacity()){
            reserve(count);
        }

        if constexpr(!std::is_trivially_destructible_v<T>) {
            std::destroy(begin_, end_);
        }

        std::uninitialized_copy(first, last, begin_);

        end_ = begin_ + count;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::assign(std::initializer_list<T> ilist){
        assign(ilist.begin(), ilist.end());
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    bool LinearContainer<T, IMemoryBackend, A>::operator==(const LinearContainer& other) const {

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

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    auto LinearContainer<T, IMemoryBackend, A>::operator<=>(const LinearContainer& other) const {

        size_t n = std::min(size(), other.size());

        for(size_t i = 0; i < n; ++i){
            if(auto cmp = begin_[i] <=> other.begin_[i]; cmp != 0){
                return cmp;
            }
        }

        return size() <=> other.size();
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    T& LinearContainer<T, IMemoryBackend, A>::operator[](size_t i) { 
        return *(begin_ + i);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    const T& LinearContainer<T, IMemoryBackend, A>::operator[](size_t i) const { 
        return *(begin_ + i);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    T* LinearContainer<T, IMemoryBackend, A>::data(){
        return begin_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    const T* LinearContainer<T, IMemoryBackend, A>::data() const {
        return begin_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    T& LinearContainer<T, IMemoryBackend, A>::front(){
        return *begin_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    const T& LinearContainer<T, IMemoryBackend, A>::front() const{
        return *begin_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    T& LinearContainer<T, IMemoryBackend, A>::back(){
        return *(end_-1);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    const T& LinearContainer<T, IMemoryBackend, A>::back() const{
        return *(end_-1);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    size_t LinearContainer<T, IMemoryBackend, A>::size() const{
        return static_cast<size_t>(end_ - begin_);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    size_t LinearContainer<T, IMemoryBackend, A>::capacity() const{
        return static_cast<size_t>(capEnd_ - begin_);
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::iterator LinearContainer<T, IMemoryBackend, A>::begin(){
        return begin_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::iterator LinearContainer<T, IMemoryBackend, A>::end(){
        return end_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::const_iterator LinearContainer<T, IMemoryBackend, A>::begin() const{
        return begin_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::const_iterator LinearContainer<T, IMemoryBackend, A>::end() const{
        return end_;
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::reverse_iterator LinearContainer<T, IMemoryBackend, A>::rbegin(){
        return reverse_iterator(end());
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::reverse_iterator LinearContainer<T, IMemoryBackend, A>::rend(){
        return reverse_iterator(begin());
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::const_reverse_iterator LinearContainer<T, IMemoryBackend, A>::rbegin() const{
        return const_reverse_iterator(end());
    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    LinearContainer<T, IMemoryBackend, A>::const_reverse_iterator LinearContainer<T, IMemoryBackend, A>::rend() const{
        return const_reverse_iterator(begin());
    }


    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::push_back_slow(const T& value){

        size_t oldSize = size();
        size_t newCap  = capacity() ? (capacity() + (capacity() / 2) + 8) : 8;

        reserve(newCap);

        std::construct_at(begin_ + oldSize, value);

        end_ = begin_ + oldSize + 1;

    }

    template<class T, MemoryBackendConcept<T> IMemoryBackend, class A>
    void LinearContainer<T, IMemoryBackend, A>::fastFill(T* dst, size_t count, const T& value){

        if constexpr(std::is_trivially_copyable_v<T>){

            if(value == T{}){
                std::memset(dst, 0, count * sizeof(T));
                return;
            }

            for(size_t i = 0; i < count; ++i){
                dst[i] = value;
            }

        }else{

            // todo: je toto správně pro netriviálně kopírovatelné?
            for(size_t i = 0; i < count; ++i){
                dst[i] = value;
            }
        }
    }
}