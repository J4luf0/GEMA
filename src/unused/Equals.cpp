#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <stdexcept>

#include "IEquals.hpp"
    
    template class Equals<char>;
    template class Equals<short>;
    template class Equals<int>;
    template class Equals<long long int>;
    template class Equals<float>;
    template class Equals<double>;

    template <typename O> class Equals<O&>;
    template <typename O> class Equals<O*>;

    // public methods

    template <class T>
    Equals<T>::Equals(const std::function<bool(T, T)>& compareFunction){
        comparatorDouble = compareFunction;
    }

    template <class T>
    bool Equals<T>::operator()(T a, T b) const{
        return comparatorDouble(a, b);
    }

    template <class T>
    bool Equals<T>::operator()(T* a, T* b) const{
        return comparatorDouble(*a, *b);
    }