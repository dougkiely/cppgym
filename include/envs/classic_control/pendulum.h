#pragma once

#include <array>
#include <concepts>
#include <cmath>
#include <numbers>
#include <random>
#include <tuple>

#include "spaces/box.h"
#include "rendering.h"

namespace gym::envs::classic_control {

template<std::floating_point T=float>
class pendulum_window;

template<std::floating_point T=float>
class pendulum {
    public:
        static const uint64_t action_size{1};
        static const uint64_t observation_size{3};

        using action_space_type = gym::spaces::box<T, action_size>;
        using action_type = action_space_type::value_type;
        using observation_space_type = gym::spaces::box<T, observation_size>;
        using observation_type = observation_space_type::value_type;
        using reward_type = T;
        using step_return_type = std::tuple<observation_type, reward_type, bool>;

        action_space_type action_space{{min_torque_}, {max_torque_}};
        observation_space_type observation_space{low_, high_};

        pendulum()=default;
        pendulum(const T g) : g_(g) {}

        uint32_t seed(uint32_t value) {
            rand_gen_.seed(value);
            return value;
        }

        auto step(const action_type& action) -> step_return_type {
            auto& [theta, theta_dot] = state_;

            const auto u = std::clamp(action[0], min_torque_, max_torque_);
            last_u_ = u; // for rendering
            const auto costs = std::pow(angle_normalize(theta), 2) + 0.1 * std::pow(theta_dot, 2) + 0.001 * std::pow(u, 2);

            theta_dot = theta_dot + (-3 * g_ / (2 * l_) * std::sin(theta + pi) + 3.0 / (m_ * std::pow(l_, 2)) * u) * dt_;
            theta = theta + theta_dot * dt_;
            theta_dot = std::clamp(theta_dot, min_speed_, max_speed_);

            return {std::move(get_obs()), -costs, false};
        }

        observation_type reset() {
            for (auto& x : state_)
                x = uniform_dist_(rand_gen_);

            last_u_ = {};

            return std::move(get_obs());
        }

        enum class render_mode {
            human,
            rgb_array
        };

        template<render_mode mode=render_mode::human>
        auto render() const {
            static pendulum_window<T> window{};
            return window.template render<mode>(state_, last_u_);
        }

        void close() {}

    protected:
        const T pi{std::numbers::pi_v<T>};

        observation_type get_obs() const {
            const auto& [theta, theta_dot] = state_;
            return {std::cos(theta), std::sin(theta), theta_dot};
        }

        T angle_normalize(const T& x) const {
            return std::fmod(x + pi, 2.0 * pi) - pi;
        }

    private:
        std::mt19937 rand_gen_{};
        std::uniform_real_distribution<T> uniform_dist_{pi, 1.0};

        const T max_speed_{8.0};
        const T min_speed_{-max_speed_};
        const T max_torque_{2.0};
        const T min_torque_{-max_torque_};
        const T dt_{0.05f};
        const T g_{10.0};
        const T m_{1.0};
        const T l_{1.0};

        T last_u_{};

        const observation_type low_{-1.0, -1.0, min_speed_};
        const observation_type high_{1.0, 1.0, max_speed_};

        std::array<T, 2> state_{0.0, 0.0};
};

template<std::floating_point T>
class pendulum_window {
    public:
        pendulum_window() {

            viewer_.set_bounds(-2.2, 2.2, -2.2, 2.2);

            rod_.set_color(1.0, 0.3, 0.0); // 0.8, 0.3, 0.3
            rod_.add_attr(pole_tf_);
            viewer_.add_geom(rod_);

            axle_.set_color(0.3, 0.3, 0.3); // 0.0, 0.0, 0.0
            viewer_.add_geom(axle_);

            img_.add_attr(img_tf_);
        }

        template<pendulum<T>::render_mode mode = pendulum<T>::render_mode::human>
        //auto render(const pendulum<T>::observation_state& state, const T& last_u) {
        auto render(const std::array<T, 2>& state, const T& last_u) {

            viewer_.add_onetime(img_);
            pole_tf_.set_rotation(state[0] + pi / 2);
            if (last_u)
                img_tf_.set_scale(point_2d{-last_u / 2, std::abs(last_u) / 2});

            return viewer_.render<mode == pendulum<T>::render_mode::rgb_array>();
        }

    protected:
        const T pi{std::numbers::pi_v<T>};

    private:
        const unsigned int screen_width_{500};
        const unsigned int screen_height_{500};

        Capsule rod_{make_capsule(1.0, 0.2)};
        Transform pole_tf_{};
        FilledPolygon axle_{make_circle(0.05)};
        Image img_{"clockwise.png", 1.0, 1.0};
        Transform img_tf_{};

        Viewer<> viewer_{"C++ Gym - Pendulum", screen_width_, screen_height_};
};

}
