// Not needed but interesting to have

// #ifndef LINEAR_CONTAINER_CONCEPT_HPP
// #define LINEAR_CONTAINER_CONCEPT_HPP

// #include <concepts>
// #include <iterator>
// #include <initializer_list>

// namespace gema {

// template<class C, class T>
// concept LinearContainerConcept = requires(C c, const C cc, T value, std::initializer_list<T> ilist, std::size_t n) {

//     // TYPES

//     typename C::iterator;
//     typename C::const_iterator;
//     typename C::reverse_iterator;
//     typename C::const_reverse_iterator;

//     requires std::same_as<typename C::iterator, T*>;
//     requires std::same_as<typename C::const_iterator, const T*>;

//     // CONSTRUCTORS

//     { C() };
//     { C(n) };
//     { C(ilist) };

//     { C(cc) };              // copy ctor
//     { C(std::move(c)) };    // move ctor

//     // ASSIGNMENT

//     { c = cc } -> std::same_as<C&>;
//     { c = std::move(c) } -> std::same_as<C&>;

//     // CAPACITY

//     { cc.size() } -> std::same_as<std::size_t>;
//     { cc.capacity() } -> std::same_as<std::size_t>;

//     // DATA ACCESS

//     { c.data() } -> std::same_as<T*>;
//     { cc.data() } -> std::same_as<const T*>;

//     { c.front() } -> std::same_as<T&>;
//     { cc.front() } -> std::same_as<const T&>;

//     { c.back() } -> std::same_as<T&>;
//     { cc.back() } -> std::same_as<const T&>;

//     { c[0] } -> std::same_as<T&>;
//     { cc[0] } -> std::same_as<const T&>;

//     // MODIFIERS

//     { c.reserve(n) } -> std::same_as<void>;
//     { c.resize(n) } -> std::same_as<void>;
//     { c.clear() } -> std::same_as<void>;

//     { c.push_back(value) } -> std::same_as<void>;
//     { c.pop_back() } -> std::same_as<void>;
//     { c.swap(c) } noexcept -> std::same_as<void>;

//     { c.fill(value) } -> std::same_as<void>;

//     { c.assign(n, value) } -> std::same_as<void>;
//     { c.assign(ilist) } -> std::same_as<void>;

//     // assign(iterator)
//     requires requires {
//         { c.assign(c.begin(), c.end()) } -> std::same_as<void>;
//     };

//     // COMPARISON

//     { cc == cc } -> std::convertible_to<bool>;
//     { cc <=> cc };

//     // ITERATORS

//     { c.begin() } -> std::same_as<T*>;
//     { c.end() } -> std::same_as<T*>;

//     { cc.begin() } -> std::same_as<const T*>;
//     { cc.end() } -> std::same_as<const T*>;

//     { c.rbegin() } -> std::same_as<typename C::reverse_iterator>;
//     { c.rend() } -> std::same_as<typename C::reverse_iterator>;

//     { cc.rbegin() } -> std::same_as<typename C::const_reverse_iterator>;
//     { cc.rend() } -> std::same_as<typename C::const_reverse_iterator>;
// };

// }

// #endif