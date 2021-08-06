#pragma once

#include <array>
#include <concepts>
#include <iostream> //debug
#include <random>

namespace gym::spaces {

template<typename T, uint64_t N=1>
class box {

    public:
        using value_type = std::array<T, N>;

        //box(T low, T high, std::pair<uint32_t> shape) {}

        //template<typename x=std::enable_if<N, 1>(true)>
        box(T low, T high)
            : low_(std::move(value_type {low})),
              high_(std::move(value_type {high})) {
            
            for (uint64_t i=0; i < N; ++i)
                bounded_below_[i] = -std::numeric_limits<T>::infinity() < low_[i];

            for (uint64_t i=0; i < N; ++i)
                bounded_above_[i] = std::numeric_limits<T>::infinity() > high_[i];

            // cartpole uses bounded interval
            std::array<T, N> bounded;
            for (uint64_t i=0; i < N; ++i)
                bounded[i] = true; //bounded_below_[i] & bounded_above_[i];
        }
        
        box(const value_type& low, const value_type& high)
            //: low_(std::move(low)), high_(std::move(high)) {}
            : low_(low), high_(high) {

            /*std::cout << "box ctor: low: ";
            for (const auto& l : low)
                std::cout << l << ", ";
            std::cout << '\n';

            std::cout << "box ctor: high: ";
            for (const auto& h : high)
                std::cout << h << ", ";
            std::cout << '\n';

            std::cout << "box ctor: low_: ";
            for (const auto& l : low_)
                std::cout << l << ", ";
            std::cout << '\n';

            std::cout << "box ctor: high_: ";
            for (const auto& h : high_)
                std::cout << h << ", ";
            std::cout << '\n';*/
        }

        box(const box& other)=delete;
        box& operator=(box& other)=delete;
        box(box&& other)=default;
        box& operator=(box&& other)=default;

        /**
         * random sample inside of the box
         *
         * each sample has form of the interval:
         *  [a, b] : uniform distribution
         *  [a, infinity) : shifted exponential distribution
         *  (-infinity, b] : shifted negative exponential distribution
         *  (-infinity, infinity) : normal distribution
         *
         **/
        value_type sample() {

            std::array<T, N> sample_data;

            // cartpole uses uniform distribution on a real (floating point) type
            for (uint64_t i=0; i < N; ++i) {
                std::uniform_real_distribution<T> uniform_dist(low_[i], high_[i]);
                sample_data[i] = uniform_dist(rand_gen_);
            }

            return sample_data;
        }

        const value_type& low() const {
            return low_; // TODO: is this good? low_ is not defaulted to 0! update: defaulted to {}
        }

        const value_type& high() const {
            return high_; // TODO: is this good? high_ is not default to 0! update: defaulted to {}
        }

        uint64_t n() const { // TODO: is this function correct?
            return N;
        }

    private:
        std::mt19937 rand_gen_{1}; // TODO: what about this seed value? openai gym hardcodes it to 100

        value_type low_{};
        value_type high_{};
        std::array<bool, N> bounded_below_;
        std::array<bool, N> bounded_above_;
};

}
