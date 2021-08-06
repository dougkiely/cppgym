#pragma once

#include <array>

namespace gym::utils {

// TODO: implement excluding endpoint
// TODO: implement returning step/interval

template<typename T, std::uint64_t size>
constexpr auto linspace(const T& start, const T& stop) -> std::array<T, size> {
    std::array<T, size> arr;

    constexpr auto partitions{size-1};

    const T step{(stop - start) / partitions};

    if constexpr (size > 0)
        arr[0] = start;
    else
        return arr;

    if constexpr (size > 1)
        arr[size-1] = stop;
    else
        return arr;

    for (std::size_t i=1; i < partitions; ++i)
        arr[i] = start + (static_cast<T>(i) * step);

    /*if constexpr (size % 2 == 1) {
        auto mid_val = std::midpoint(start, stop);
        std::size_t mid_idx = partitions / 2;
        //std::cout << "mid_idx: " << mid_idx << ", mid_val: " << mid_val << '\n';
        arr[mid_idx] = mid_val;
    }*/

    return arr;
}

}
