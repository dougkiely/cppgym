#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream> //debug
#include <random>
#include <tuple>
#include <vector>

#include "core.h"
#include "spaces/box.h"
#include "spaces/discrete.h"
#include "utils/algo.h"
#include "utils/linalg.h"
#include "rendering.h"

namespace gym::envs::classic_control {

template<std::floating_point T=float>
class mountain_car_window;

template<std::floating_point T=float>
class mountain_car {
    public:
        static const uint64_t action_size{3};
        static const uint64_t observation_size{2};

        using action_space_type = gym::spaces::discrete<>;
        using action_type = action_space_type::value_type;
        using observation_space_type = gym::spaces::box<T, observation_size>;
        using observation_type = observation_space_type::value_type;
        using reward_type = T;
        using step_return_type = std::tuple<observation_type, reward_type, bool>;

        action_space_type action_space{action_size};
        observation_space_type observation_space{low_, high_};

        mountain_car()=default;
        mountain_car(const T goal_velocity) : goal_velocity_(goal_velocity) {}

        mountain_car(const mountain_car&)=delete;
        mountain_car& operator=(const mountain_car&)=delete;
        mountain_car(mountain_car&&)=default;
        mountain_car& operator=(mountain_car&&)=default;

        /*uint32_t seed() {
            std::seed_seq{1,2,3,4,5}; // TODO: increase entropy of the generated random numbers
        }*/

        uint32_t seed(uint32_t value) {
            rand_gen_.seed(value);
            return value;
        }

        auto step(const action_type& action) -> step_return_type {
            if (!action_space.contains(action))// TODO: assert
                return {state_, 0.0, true};

            auto& [position, velocity] = state_;

            // convert action values {0, 1, 2} to a direction {-1 (reverse), 0 (neutral), +1 (forward)}
            T action_direction{static_cast<T>(static_cast<int>(action) - 1)};

            velocity += action_direction * force_ + std::cos(3 * position) * (-gravity_);
            velocity = std::clamp(velocity, -max_speed_, max_speed_);
            position += velocity;
            position = std::clamp(position, min_position_, max_position_);

            if (position == min_position_ && velocity < 0.0)
                velocity = 0.0;

            const bool done = position >= goal_position_ && velocity >= goal_velocity_;

            const reward_type reward{-1.0};

            return {state_, reward, done};
        }

        observation_type reset() {
            return state_ = {uniform_dist_(rand_gen_), 0.0};
        }

        enum class render_mode {
            human,
            rgb_array
        };

        template<render_mode mode=render_mode::human>
        auto render() const {
            static mountain_car_window window{min_position_, max_position_, goal_position_};
            return window.template render<mode>(state_);
        }

        void close() {}

        using value_type = T;

    private:
        std::mt19937 rand_gen_{100};
        std::uniform_real_distribution<T> uniform_dist_{-0.6f, -0.4f};

        constexpr static T min_position_{-1.2f};
        constexpr static T max_position_{ 0.6f};
        constexpr static T max_speed_{0.07f};
        constexpr static T goal_position_{0.5f};
        T goal_velocity_{0.0f};

        constexpr static T force_{0.001f};
        constexpr static T gravity_{0.0025f};

        constexpr static observation_type low_{min_position_, -max_speed_};
        constexpr static observation_type high_{max_position_, max_speed_}; 

        observation_type state_{0.0, 0.0};
};

template<std::floating_point T>
class mountain_car_window {
    public:
        mountain_car_window(const T min_position, const T max_position, const T goal_position)
            : min_position_(min_position), max_position_(max_position),
                goal_position_(goal_position),
                  world_width_(max_position_ - min_position_),
                    scale_(static_cast<T>(screen_width_) / world_width_) {

            using namespace gym::utils;

            const size_t size_{100};

            const std::array<T, size_> xs{linspace<T, size_>(min_position_, max_position_)};
            const std::array<T, size_> ys{calc_height(xs)};

            const auto xys = zip<std::vector<point_2d<T>>>(xs, ys, [min_position=min_position_, scale=scale_](const T& x, const T& y){
                return point_2d{(x - min_position) * scale, y * scale};
            });

            track_.set_vertices(xys);
            track_.set_color(0.0f, 0.0f, 1.0f);
            track_.set_width(4);
            viewer_.add_geom(track_);

            car_.set_color(0.0f, 0.0f, 0.0f);
            car_.add_attr(car_clearance_transform_);
            car_.add_attr(car_transform_);
            viewer_.add_geom(car_);

            front_wheel_.set_color(0.5f, 0.5f, 0.5f);
            front_wheel_.add_attr(front_wheel_transform_);
            front_wheel_.add_attr(car_transform_);
            viewer_.add_geom(front_wheel_);

            back_wheel_.set_color(0.5f, 0.5f, 0.5f);
            back_wheel_.add_attr(back_wheel_transform_);
            back_wheel_.add_attr(car_transform_);
            viewer_.add_geom(back_wheel_);

            const auto flag_x = (goal_position_ - min_position_) * scale_;
            const auto flag_y1 = calc_height(goal_position_) * scale_;
            const auto flag_y2 = flag_y1 + 50;
            flag_pole_.set_points({flag_x, flag_y1}, {flag_x, flag_y2});
            flag_pole_.set_color(0.25f, 0.25f, 0.25f);
            viewer_.add_geom(flag_pole_);

            flag_.set_vertices({{flag_x, flag_y2}, {flag_x, flag_y2 - 10}, {flag_x + 25, flag_y2 - 5}});
            flag_.set_color(0.0f, 1.0f, 0.0f);
            viewer_.add_geom(flag_);
        }

        template<mountain_car<T>::render_mode mode>
        auto render(mountain_car<T>::observation_type const& state) {
            const auto position{state[0]};
            car_transform_.set_translation(point_2d{(position - min_position_) * scale_, calc_height(position) * scale_});
            car_transform_.set_rotation(std::cos(3 * position));
            
            return viewer_.render<mode == mountain_car<T>::render_mode::rgb_array>();
        }

protected:
        T calc_height(const T x) const {
            return std::sin(3 * x) * 0.45f + 0.55f;
        }

        template<typename U>
        U calc_height(const U& xs) const {
            U ret;

            // TODO: test with vector and array
            std::transform(std::begin(xs), std::end(xs), std::begin(ret), [this](T x) -> T {
                return calc_height(x);
            });

            return ret;
        }

    private:
        const unsigned int screen_width_{600};
        const unsigned int screen_height_{400};

        T min_position_;
        T max_position_;
        T goal_position_;

        const T world_width_;//{max_position_ - min_position_};
        const T scale_;//{static_cast<T>(screen_width_) / world_width_};
        constexpr static T car_width_{40.0};
        constexpr static T car_height_{20.0};

        PolyLine track_;

        constexpr static T clearance_{10.0};

        constexpr static T l_{-car_width_ / 2};
        constexpr static T r_{ car_width_ / 2};
        constexpr static T t_{ car_height_};
        constexpr static T b_{0.0};

        FilledPolygon car_{{l_, b_}, {l_, t_}, {r_, t_}, {r_, b_}};
        Transform car_clearance_transform_{{0.0, clearance_}};
        Transform car_transform_;

        FilledPolygon front_wheel_{make_circle(car_height_ / 2.5f)};
        Transform front_wheel_transform_{{car_width_ / 4, clearance_}};

        FilledPolygon back_wheel_{make_circle(car_height_ / 2.5f)};
        Transform back_wheel_transform_{{-car_width_ / 4, clearance_}};

        Line flag_pole_;
        FilledPolygon flag_;

        Viewer<> viewer_{"C++ Gym - Mountain Car", screen_width_, screen_height_};
};

}
