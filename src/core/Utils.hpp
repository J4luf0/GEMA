#ifndef UTILS_HPP
#define UTILS_HPP

#include <concepts>
#include <cstdint>
#include <span>
#include <cmath>

namespace gema{

template<typename T, std::size_t Extent = std::dynamic_extent>
class span_view : public std::span<const T, Extent> {
    using Base = std::span<const T, Extent>;

public:
    using Base::Base;

    constexpr span_view(std::initializer_list<T> il)
        requires (Extent == std::dynamic_extent)
        : Base(il.begin(), il.size())
    {}

    constexpr operator std::span<const T, Extent>() const noexcept {
        return static_cast<Base>(*this);
    }
};


template<typename T>
struct DefaultEquals {

    bool operator()(const T& a, const T& b) const {
        if constexpr(std::is_floating_point_v<T>) {
            T epsilon = std::numeric_limits<T>::epsilon();
            return std::fabs(a - b) <= epsilon * std::max(std::fabs(a), std::fabs(b));
        } else {
            return a == b;
        }
    }
};


template<typename T>
struct DefaultOrder {

    int operator()(const T& a, const T& b) const {

        if constexpr(std::is_floating_point_v<T>) {
            T epsilon = std::numeric_limits<T>::epsilon();
            if (std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)))){
                return 0;
            }else if(a > b){
                return 1;
            }else{
                return -1;
            }

            return (a > b) ? 1 : -1;

        } else if constexpr(std::is_integral_v<T>) {
            return (a != b) * ((a > b) - (a < b));
        } else {
            return 0;
        }
    }
};

}

#endif

