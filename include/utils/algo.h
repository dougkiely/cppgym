#pragma once

#include <concepts>
#include <iostream>
#include <tuple>
#include <vector>

namespace gym::utils {

template<typename T>
concept indexed_container = requires(T t, size_t i) {
    typename T::size_type;
    typename T::value_type;
    { t.size() } -> std::same_as<typename T::size_type>;
    { t[i] } -> std::convertible_to<typename T::value_type>;
};

template<typename R, indexed_container T, indexed_container U>
constexpr auto zip(const T& a, const U& b) -> R {
    
    R ret;

    auto a_size = a.size(), b_size = b.size();

    auto max = a_size > b_size ? a_size : b_size;

    //ret.reserve(max); // reserve() member function only exists in std::vector

    for (size_t i=0; i < max; ++i) {

        typename T::value_type a_val{}, b_val{};
        
        if (i < a_size)
            a_val = a[i];

        if (i < b_size)
            b_val = b[i];

        ret.emplace_back(a_val, b_val);
    }

    return ret;
}

//template<typename T, typename U>
//constexpr auto zip(const T& a, const T& b,
//                    std::invocable<typename U::value_type, T, T> auto invoke_func) -> U {
////template<typename T, typename U, typename F>
//constexpr U zip(const T& a, const T& b, F func) requires requires {
//                    std::is_invocable_r_v<typename U::value_type, func, T, std::string>;
//                    std::same_as<typename T::value_type, std::string>; } {
template<typename R, indexed_container T, indexed_container U, typename F>
constexpr auto zip(const T& a, const U& b, F func) -> R
                requires
                    std::is_invocable_r_v<typename R::value_type, F,
                        typename T::value_type const&, typename T::value_type const&>
{
    R ret;

    auto a_size = a.size(), b_size = b.size();

    auto max = a_size > b_size ? a_size : b_size;

    //ret.reserve(max); // reserve() member function only exists in std::vector

    for (size_t i=0; i < max; ++i) {

        typename T::value_type a_val{}, b_val{};
        
        if (i < a_size)
            a_val = a[i];

        if (i < b_size)
            b_val = b[i];

        ret.push_back(std::move(func(a_val, b_val)));
    }

    return ret;
}

/*
//template<std::input_iterator T>
template<typename T>
auto zip(const T a, const T b, std::invocable<T, T> auto func)
    -> std::vector<std::tuple<typename T::value_type, typename T::value_type>> {
    
    std::vector<std::tuple<typename T::value_type, typename T::value_type>> vec_tuple;

    auto a_size = a.size(), b_size = b.size();

    auto max = a_size > b_size ? a_size : b_size;

    vec_tuple.reserve(max);

    for (size_t i=0; i < max; ++i) {

        typename T::value_type a_val{}, b_val{};
        
        std::cout << "i=" << i << ": ";

        if (i < a_size) {
            a_val = a[i];
            //tup = std::tuple_cat(a[i]);
            std::cout << "a[" << i << "]: " << a[i] << ", ";
        } else {
            std::cout << "null ";
            //tup = std::tuple_cat(T{});
        }

        if (i < b_size) {
            b_val = b[i];
            //tup = std::tuple_cat(a[i]);
            std::cout << "b[" << i << "]: " << b[i] << ", ";
        } else {
            std::cout << "null ";
            //tup = std::tuple_cat(T{});
        }

        vec_tuple[i] = std::make_tuple(a_val, b_val); //std::tuple<T...> tup;
        std::cout << '\n';
    }

    return vec_tuple;
}*/

//template<std::input_iterator... T>
/*template<typename... T>
void zip(const T&&... args) {

    std::vector vec{args...};

    for (const auto& arg : vec)
        std::cout << "arg: " << arg << '\n';
}*/

//template<std::input_iterator... T>
/*
template<typename... T>
void zip(const T&&... args) {

    std::vector vec{args...};

    std::vector<std::tuple<T...>> ret_vec_tuple;

    size_t max{};

    for (const auto& arg : vec) {
        auto vec_size = arg.size();
        if (vec_size > max)
            max = vec_size;
    }

    ret_vec_tuple.reserve(max);

    for (size_t i=0; i < max; ++i) {
        std::tuple<T...> tup;
        for (const auto& arg : vec) {
            if (i < arg.size()) {
                tup = std::tuple_cat(arg[i]);
                std::cout << "i=" << i << ", arg[" << i << "]: " << arg[i] << ' ';
            } else {
                std::cout << "null ";
                tup = std::tuple_cat(T{});
            }
        }

        std::cout << '\n';
    }
}
*/

}
