#pragma once

#include <concepts>

namespace gym::spaces {

template<typename T>
concept space = requires(T& t) {

    { t.sample() } -> std::same_as<T::data_type>;
};

}
