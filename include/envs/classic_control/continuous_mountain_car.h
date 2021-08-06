#pragma once

#include <cmath>
#include <random>
#include <tuple>

#include "spaces/box.h"
#include "mountain_car.h" // import mountain_car_window class

namespace gym::envs::classic_control {

template<std::floating_point T=float>
class continuous_mountain_car {
    private:
        const T min_action_{-1.0};
        const T max_action_{1.0};

    public:
        static const uint64_t action_size{1};
        static const uint64_t observation_size{2};

        using action_space_type = gym::spaces::box<T, action_size>;
        using action_type = action_space_type::value_type;
        using observation_space_type = gym::spaces::box<T, observation_size>;
        using observation_type = observation_space_type::value_type;
        using reward_type = T;
        using step_return_type = std::tuple<observation_type, reward_type, bool>;

        action_space_type action_space{{min_action_}, {max_action_}};
        observation_space_type observation_space{low_state_, high_state_};

        continuous_mountain_car() : continuous_mountain_car(0.0) {}

        continuous_mountain_car(const T goal_velocity) : goal_velocity_(goal_velocity) {
            //seed();
            reset();
        }

        uint32_t seed(uint32_t value) {
            rand_gen_.seed(value);
            return value;
        }

        auto step(const action_type& action) -> step_return_type {
            auto act = action[0];
            if (act < min_action_ || act > max_action_) // TODO: assert
                return {state_, 0.0, true};

            auto& [position, velocity] = state_;
            auto force = std::fmin(std::fmax(act, min_action_), max_action_);

            velocity += force * power_ - gravity_ * std::cos(3 * position);
            std::clamp(velocity, -max_speed_, max_speed_);

            position += velocity;
            std::clamp(position, min_position_, max_position_);

            if (position == min_position_ && velocity < 0.0)
                velocity = 0.0;

            bool done = position >= goal_position_ && velocity >= goal_velocity_;

            reward_type reward{0.0};
            if (done)
                reward = 100.0;
            reward -= std::pow(static_cast<float>(act), 2.f) * 0.1f;

            return {state_, reward, done};
        }

        observation_type reset() {
            return state_ = {uniform_dist_(rand_gen_), 0.0};
        }

        using render_mode = mountain_car<T>::render_mode;

        template<render_mode mode=render_mode::human>
        auto render() const {
            static mountain_car_window window{min_position_, max_position_, goal_position_};
            return window.template render<mode>(state_);
        }

        void close() {}

    private:
        std::mt19937 rand_gen_{};
        std::uniform_real_distribution<T> uniform_dist_{-0.6f, -0.4f};

        const T min_position_{-1.2f};
        const T max_position_{0.6f};
        const T max_speed_{0.7f};
        const T goal_position_{0.45f}; // was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        const T goal_velocity_{};
        const T power_{0.0015f};
        const T gravity_{0.0025f};

        const observation_type low_state_{min_position_, -max_speed_};
        const observation_type high_state_{max_position_, max_speed_};

        observation_type state_{0.0, 0.0};
};

}
