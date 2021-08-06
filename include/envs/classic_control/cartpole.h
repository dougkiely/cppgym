#pragma once

#include <cmath>
#include <iostream> //debug
#include <limits>
#include <numbers>
#include <random>
#include <tuple>
#include <vector>

#include "core.h" // env_base
#include "spaces/box.h"
#include "spaces/discrete.h"
#include "rendering.h"

namespace gym::envs::classic_control {

// TODO: implement copy and move constructors and assignment operators
// TODO: consider optimizing integral/floating-point types based on architecture

template<std::floating_point T=float>
class cartpole_window;

template<std::floating_point T=float>
class cartpole {
    public:
        static const uint64_t action_size{2};
        static const uint64_t observation_size{4};

        using action_space_type = gym::spaces::discrete<>;
        using action_type = action_space_type::value_type;
        using observation_space_type = gym::spaces::box<T, observation_size>;
        using observation_type = observation_space_type::value_type;
        using reward_type = T;
        using step_return_type = std::tuple<observation_type, reward_type, bool>;

        action_space_type action_space{action_size};
        observation_space_type observation_space{low_, high_};

        cartpole()=default;
        cartpole(const cartpole&)=delete;
        cartpole& operator=(const cartpole&)=delete;
        cartpole(cartpole&&)=default;
        cartpole& operator=(cartpole&&)=default;

        /*virtual ~cartpole() {
            close();
        }*/

        /*uint32_t seed() {
            std::seed_seq{1,2,3,4,5}; // TODO: increase entropy of the generated random numbers
        }*/

        uint32_t seed(uint32_t value) {
            rand_gen_.seed(value);
            return value; // TODO: do we actually need to return the seed value like this? or have another function like get_seed()?
        }

        auto step(const action_type& action) -> step_return_type {
            if (!action_space.contains(action)) { // TODO: assert
                return {state_, 0.0, true};
            }

            auto& [x, x_dot, theta, theta_dot] = state_;
            T force = action == 1 ? force_mag_ : -force_mag_;
            T costheta = std::cos(theta);
            T sintheta = std::sin(theta);

            T temp = (force + polemass_length_ * std::pow(theta_dot, 2.0f) * sintheta) / total_mass_;
            T thetaacc = (gravity_ * sintheta - costheta * temp) / (length_ * (4.0f / 3.0f - masspole_ * std::pow(costheta, 2.0f) / total_mass_));
            T xacc = temp - polemass_length_ * thetaacc * costheta / total_mass_;

            // update the four state variables, using euler's method
            x += tau_ * x_dot;
            x_dot += tau_ * xacc;
            theta += tau_ * theta_dot;
            theta_dot += tau_ * thetaacc;

            const bool done = x < -x_threshold_ || x > x_threshold_ ||
                              theta < -theta_threshold_radians_ ||
                              theta > theta_threshold_radians_;

            reward_type reward{0.0};

            if (!done) {
                reward = 1.0;
            } else if (steps_beyond_done_ == -1) {
                // pole just fell!
                steps_beyond_done_ = 0;
                reward = 1.0;
            } else {
                if (steps_beyond_done_ == 0)
                    std::cout << "Error: Calling step() after done == true. Call reset() after done == true. Any further steps are undefined behavior.\n";
                ++steps_beyond_done_;
            }

            return {state_, reward, done}; // TODO:, info=std::unordered_map<std::string, float>
        }

        observation_type reset() {
            for (auto& x : state_)
                x = uniform_dist_(rand_gen_);

            steps_beyond_done_ = -1;

            return state_;
        }

        enum class render_mode {
            human,
            rgb_array
        };

        //bool render(/*mode="human"*/) {
        //std::vector<uint8_t> render(/*mode="rgb_array"*/) {
        template<render_mode mode=render_mode::human>
        auto render() const {
            static cartpole_window<T> window{x_threshold_, length_}; // TODO: is this static relative to (within the scope of) this object instance only?
            return window.template render<mode>(state_);
        }

        void close() {}

        using value_type = T;

    private:
        std::mt19937 rand_gen_{1};
        std::uniform_real_distribution<T> uniform_dist_{-0.5, 0.5};

        const T gravity_{9.8f};
        const T masscart_{1.0f};
        const T masspole_{0.1f};
        const T total_mass_{masspole_ + masscart_};
        const T length_{0.5f}; // actually half the pole's length
        const T polemass_length_{masspole_ * length_};
        const T force_mag_{10.0f};
        const T tau_{0.02f}; // seconds between state updates

        // angle at which to fail the episode
        constexpr static T theta_threshold_radians_{12 * 2 * std::numbers::pi_v<T> / 360};
        constexpr static T x_threshold_{2.4f};

        constexpr static observation_type low_{-x_threshold_ * 2.0f,
                                               std::numeric_limits<T>::lowest(),
                                               -theta_threshold_radians_ * 2.0f,
                                               std::numeric_limits<T>::lowest()};

        constexpr static observation_type high_{x_threshold_ * 2.0f,
                                                std::numeric_limits<T>::max(),
                                                theta_threshold_radians_ * 2.0f,
                                                std::numeric_limits<T>::max()};

        observation_type state_{0.0, 0.0, 0.0, 0.0};

        int steps_beyond_done_{-1};
};

template<std::floating_point T>
class cartpole_window {
    public:
        cartpole_window(const T x_threshold, const T length) : x_threshold_(x_threshold), length_(length) {
            cart_.add_attr(cart_transform_);
            viewer_.add_geom(cart_);

            pole_.set_color(1.0f, 0.0f, 1.0f); //0.8f, 0.6f, 0.4f
            pole_.add_attr(pole_transform_);
            pole_.add_attr(cart_transform_);
            viewer_.add_geom(pole_);

            axle_.add_attr(pole_transform_);
            axle_.add_attr(cart_transform_);
            axle_.set_color(1.0f, 0.5f, 1.0f); //0.5f, 0.5f, 0.8f
            viewer_.add_geom(axle_);
 
            track_.set_color(1.0f, 1.0f, 0.0f); //0.0f, 0.0f, 0.0f
            viewer_.add_geom(track_);
        }

        template<cartpole<T>::render_mode mode>
        auto render(cartpole<T>::observation_type const& state) {
            auto cart_x = state[0] * scale_ + static_cast<T>(screen_width_) / 2.0f; // middle of cart
            cart_transform_.set_translation(point_2d{cart_x, cart_y_});
            pole_transform_.set_rotation(-state[2]);

            return viewer_.render<mode == cartpole<T>::render_mode::rgb_array>(); /*return_rgb_array=mode=="rgb_array"*/
        }

    private:
        const unsigned int screen_width_{600u};
        const unsigned int screen_height_{400u};

        T x_threshold_;
        T length_;
        
        const T world_width_{x_threshold_ * 2.0f};
        const T scale_{static_cast<T>(screen_width_) / world_width_};

        constexpr static T cart_y_{100.0f}; // top of cart
        constexpr static T pole_width_{10.0f};
        const T pole_len_{scale_ * (2 * length_)};
        constexpr static T cart_width_{50.0};
        constexpr static T cart_height_{30.0};

        constexpr static T p_l_{-pole_width_ / 2};
        constexpr static T p_r_{ pole_width_ / 2};
        const            T p_t_{ pole_len_ - pole_width_ / 2};
        constexpr static T p_b_{-pole_width_ / 2};

        const T c_l_{-cart_width_ / 2};
        const T c_r_{ cart_width_ / 2};
        const T c_t_{ cart_height_ / 2};
        const T c_b_{-cart_height_ / 2};

        const T axle_offset_{cart_height_ / 4};

        FilledPolygon cart_{{c_l_, c_b_}, {c_l_, c_t_}, {c_r_, c_t_}, {c_r_, c_b_}};
        FilledPolygon pole_{{p_l_, p_b_}, {p_l_, p_t_}, {p_r_, p_t_}, {p_r_, p_b_}};
        Transform cart_transform_;
        Transform pole_transform_{{0.0f, axle_offset_}};
        FilledPolygon axle_{make_circle<FilledPolygon>(pole_width_ / 2)};
        Line track_{{0.0f, cart_y_}, {static_cast<T>(screen_width_), cart_y_}};

        Viewer<> viewer_{"C++ Gym - Cart Pole", screen_width_, screen_height_};
};

}
