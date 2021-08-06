#include <cmath>
#include <coroutine>
#include <concepts>

#include "generator_gcc.hpp"

/*template<std::unsigned_integral T>
concept decay_rate = requires(T rate) {
    rate > 0;
};*/

template<std::floating_point T>
cppcoro::generator<T> decay(T start, T end, std::unsigned_integral auto rate) {
    T current{start};
    for (size_t step=0; ; ++step) {
        if (current > end)
            current = end + (current - end) * std::exp(-1.0f * step / rate);
        co_yield current;
    }
}
