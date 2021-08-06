#pragma once

#include <concepts>
#include <random>

namespace gym::spaces {

template<std::integral T=uint64_t>
class discrete {

    public:
        using value_type = T;
        //using shape_type = std::tuple<>;

        discrete(size_t n) : n_(n), uniform_dist_(0, n_ - 1) {}

        discrete(const discrete& other)=delete;
        discrete& operator=(const discrete& other)=delete;
        discrete(discrete&& other)=default;
        discrete& operator=(discrete&& other)=default;

        inline value_type sample() {
            return uniform_dist_(rand_gen_);
        }

        inline bool contains(value_type x) {
            return x >= 0 && x < n_;
        }

        size_t n() const {
            return n_;
        }

    private:
        size_t n_{};

        std::mt19937 rand_gen_{1}; // TODO: what about seed value? openai gym uses seed() function from space
        std::uniform_int_distribution<value_type> uniform_dist_;
};

}
