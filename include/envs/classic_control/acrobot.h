#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <iostream> //debug
#include <numbers>
#include <random>

#include "spaces/box.h"
#include "spaces/discrete.h"
#include "rendering.h"

namespace gym::envs::classic_control {

template<typename T>
concept scalar = std::integral<T> || std::floating_point<T>;

template<scalar T>
T wrap(T x, const T m, const T M) {
    const auto diff = M - m;

    while (x > M)
        x -= diff;

    while (x < m)
        x += diff;

    return x;
}

template<scalar T>
T bound(const T x, const T m, const T M) {
    // bound x between min (m) and Max (M)
    return std::min(std::max(x, m), M);
}

template<scalar T>
T bound(const T x, const std::array<T, 2> m) {
    return bound(x, m[0], m[1]);
}

template<std::floating_point T=float>
class acrobot_window;

template<std::floating_point T=float>
class acrobot {
    public:
        static const uint64_t action_size{3};
        static const uint64_t observation_size{6};

        using action_space_type = gym::spaces::discrete<>;
        using action_type = action_space_type::value_type;
        using observation_space_type = gym::spaces::box<T, observation_size>;
        using observation_type = observation_space_type::value_type;
        using reward_type = T;
        using step_return_type = std::tuple<observation_type, reward_type, bool>;

        action_space_type action_space{action_size};
        observation_space_type observation_space{low_, high_};

        /*acrobot() {
            seed();
        }*/

        /*virtual ~acrobot() {
            close();
        }*/

        /*void seed() {

        }*/

        auto step(const action_type& action) -> step_return_type {
            //if (!action_space.contains(action)) // TODO: assert
            //    return {get_ob(), 0.0, true};

            auto torque = AVAIL_TORQUE_[action];

            // add noise to the force action
            if (torque_noise_max_ > 0.0)
                torque += torque_noise_dist_(rand_gen_);

            // now, augment the state with our force action so it can be passed to _dsdt
            const std::array<T, 5> s_augmented{state_[0], state_[1], state_[2], state_[3], torque};

            const auto ns_tmp = rk4<2, 5>(/*dsdt, */s_augmented, {0.0, dt_});
            // only care about final timestep of integration returned by integrator
            const auto ns_final = std::array{ns_tmp[1]};
            const auto ns = std::array{ns_final[0], ns_final[1], ns_final[2], ns_final[3]}; // omit action

            state_[0] = wrap(ns[0], -pi, pi);
            state_[1] = wrap(ns[1], -pi, pi);
            state_[2] = bound(ns[2], -MAX_VEL_1_, MAX_VEL_1_);
            state_[3] = bound(ns[3], -MAX_VEL_2_, MAX_VEL_2_);

            std::cout << "step: state_[0]: " << state_[0] << ", state_[1]: " << state_[1] << ", state_[2]: " << state_[2] << ", state[3]: " << state_[3] << '\n';

            const bool terminal{is_terminal()};
            const reward_type reward{terminal ? 0.0f : -1.0f};
            return {get_ob(), reward, terminal};
        }

        observation_type reset() {
            for (auto& x : state_)
                x = uniform_dist_(rand_gen_);

            return std::move(get_ob());
        }

        enum class render_mode {
            human,
            rgb_array
        };

        template<render_mode mode=render_mode::human>
        auto render() const {
            static acrobot_window<T> window(LINK_LENGTH_1_, LINK_LENGTH_2_);
            return window.template render<mode>(state_);
        }

        void close() const {}

    protected:
        using internal_observation_type = std::array<T, 4>;

        observation_type get_ob() const {
            const auto& [s0, s1, s2, s3] = state_;
            return {std::cos(s0), std::sin(s0), std::cos(s1), std::sin(s1), s2, s3};
        }

        bool is_terminal() const {
            return -std::cos(state_[0]) - std::cos(state_[1] + state_[0]) > 1.0;
        }

        auto dsdt(const std::array<T, 5>& s_augmented/*, const T t*/) const -> std::array<T, 5> {

            const T m1{LINK_MASS_1_};
            const T m2{LINK_MASS_2_};
            const T l1{LINK_LENGTH_1_};
            const T lc1{LINK_COM_POS_1_};
            const T lc2{LINK_COM_POS_2_};
            const T I1{LINK_MOI_};
            const T I2{LINK_MOI_};

            const T g{9.8};
            const auto& [theta1, theta2, dtheta1, dtheta2, action] = s_augmented;

            const auto d1 = m1 * std::pow(lc1, 2) + m2 *
                (std::pow(l1, 2) + std::pow(lc2, 2) + 2 * l1 * lc2 * std::cos(theta2)) + I1 + I2;
            const auto d2 = m2 * (std::pow(lc2, 2) + l1 * lc2 * std::cos(theta2)) + I2;
            const auto phi2 = m2 * lc2 * g * std::cos(theta1 + theta1 - pi / 2.0);
            const auto phi1 = -m2 * l1 * lc2 * std::pow(dtheta2, 2) * std::sin(theta2)
                        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * std::sin(theta2)
                        + (m1 * lc1 + m2 * l1) * g * std::cos(theta1 - pi / 2.0) + phi2;

            T ddtheta2;

            if constexpr (book_or_nips_ == dynamics_equation::nips)
                // the following line is consistent with the description in the paper
                ddtheta2 = (action + d2 / d1 * phi1 - phi2) /
                            (m2 * std::pow(lc2, 2) + I2 - std::pow(d2, 2) / d1);
            else
                // the following line is consistent with the code implementation and the book
                ddtheta2 = (action + d2 / d1 * phi1 - m2 * l1 * lc2 * std::pow(dtheta1, 2) * std::sin(theta2) - phi2)
                        / (m2 * std::pow(lc2, 2) + I2 - std::pow(d2, 2) / d1);

            const T ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

            std::cout << "dsdt: dtheta1: " << dtheta1 << ", dtheta2: " << dtheta2 <<
                       ", ddtheta1: " << ddtheta1 << ", ddtheta2: " << ddtheta2 << '\n';
            return {dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0};
        }

        template<size_t row_size, size_t col_size>
        auto rk4(/*std::invocable<const std::array<T, 5>&> auto derivs,*/
            std::array<T, col_size> y0, const std::array<T, row_size>& t) const
                //requires
                    /*std::is_invocable_r_v<std::array<T, 5>&, decltype(derivs),
                                            std::array<T, 5>>
                        && *//*std::is_bounded_array_v<decltype(y0)>
                            && std::is_bounded_array_v<decltype(t)>*/
        {
            auto yout = std::array<std::array<T, col_size>, row_size> {};

            yout[0] = y0;

            //for (typename decltype(t)::size_type i=0; i < t.size() - 1; ++i) {
            for (std::size_t i=0; i < t.size() - 1; ++i) {

                const auto thist = t[i];
                const auto dt = t[i + 1] - thist;
                const auto dt2 = dt / 2.0;
                y0 = yout[i];

                using y0_type = decltype(y0);
                using y0_size_type = y0_type::size_type;
                const y0_size_type y0_size = y0.size();

                y0_type y0_tmp{};

                //const auto k1 = derivs(y0, thist);
                const auto k1 = dsdt(y0);

                for (y0_size_type y=0; y < y0_size; ++y)
                    y0_tmp[y] = y0[y] + dt2 * k1[y];

                //const auto k2 = derivs(y0_tmp, thist + dt2);
                const auto k2 = dsdt(y0_tmp);

                for (y0_size_type y=0; y < y0_size; ++y)
                    y0_tmp[y] = y0[y] + dt2 * k2[y];

                //const auto k3 = derivs(y0_tmp, thist + dt2);
                const auto k3 = dsdt(y0_tmp);

                for (y0_size_type y=0; y < y0_size; ++y)
                    y0_tmp[y] = y0[y] + dt * k3[y];

                //const auto k4 = derivs(y0_tmp, thist + dt);
                const auto k4 = dsdt(y0_tmp);

                for (y0_size_type y=0; y < y0_size; ++y)
                    yout[i + 1][y] = y0[y] + dt / 6.0 * (k1[y] + 2 * k2[y] + 2 * k3[y] + k4[y]);
            }

            std::cout << "rk4: yout:\n";
            for (size_t i=0; i < row_size; ++i) {
                for (size_t j=0; j < col_size; ++j)
                    std::cout << yout[i][j] << ' ';
                std::cout << '\n';
            }

            return yout;
        }

    protected:
        const T pi{std::numbers::pi_v<T>};

    private:
        std::mt19937 rand_gen_{1};
        std::uniform_real_distribution<T> uniform_dist_{-0.1, 0.1};

        const T dt_{0.2};

        const T LINK_LENGTH_1_{1.0};  // [m]
        const T LINK_LENGTH_2_{1.0};  // [m]
        const T LINK_MASS_1_{1.0};    // [kg] mass of link 1
        const T LINK_MASS_2_{1.0};    // [kg] mass of link 2
        const T LINK_COM_POS_1_{0.5}; // [m] position of the center of mass of link 1
        const T LINK_COM_POS_2_{0.5}; // [m] position of the center of mass of link 2
        const T LINK_MOI_{1.0};       // moments of inertia for both links

        const T MAX_VEL_1_{4 * pi};
        const T MAX_VEL_2_{9 * pi};

        const std::array<T, 3> AVAIL_TORQUE_{-1.0, 0.0, 1.0};

        const T torque_noise_max_{0.0};
        const T torque_noise_min_{-torque_noise_max_};
        std::uniform_real_distribution<T> torque_noise_dist_{torque_noise_min_, torque_noise_max_};

        // use dynamics equations from the nips paper or the book
        enum class dynamics_equation {
            book,
            nips
        };

        static constexpr dynamics_equation book_or_nips_{dynamics_equation::book};

        const observation_type high_{1.0, 1.0, 1.0, 1.0, MAX_VEL_1_, MAX_VEL_2_};
        const observation_type low_{-1.0, -1.0, -1.0, -1.0, -MAX_VEL_1_, -MAX_VEL_2_};

        internal_observation_type state_{0.0, 0.0, 0.0, 0.0};
};

template<std::floating_point T>
class acrobot_window {
    public:
        acrobot_window(const T link_length_1, const T link_length_2) : LINK_LENGTH_1_(link_length_1), LINK_LENGTH_2_(link_length_2) {

            viewer_.set_bounds(-bound_, bound_, -bound_, bound_);

            line_.set_color(0.7, 0.7, 0.7);

            viewer_.add_geom(line_);

            for (std::size_t i=0; i < SIZE; ++i) {
                link_[i].set_color(1.0, 0.3, 0.0); // 0.0, 0.8, 0.8
                link_[i].add_attr(j_tf_[i]);

                circ_[i].set_color(0.3, 0.3, 0.3); // 0.8, 0.8, 0.0
                circ_[i].add_attr(j_tf_[i]);
            }
        }

        template<acrobot<T>::render_mode mode=acrobot<T>::render_mode::human>
        //auto render(const acrobot<T>::internal_observation_type& state) {
        auto render(const std::array<T, 4>& state) {

            const auto s0{state[0]};
            const auto s1{state[1]};

            const auto p1 = std::array<T, SIZE> {-LINK_LENGTH_1_ * std::cos(s0),
                                                  LINK_LENGTH_1_ * std::sin(s0)};

            const auto p2 = std::array<T, SIZE>
                                          {p1[0] - LINK_LENGTH_2_ * std::cos(s0 + s1),
                                           p1[1] + LINK_LENGTH_2_ * std::sin(s0 + s1)};

            const auto xys = std::array<std::array<T, SIZE>, 3> {
                                       {std::array<T, SIZE> {0.0, 0.0},
                                        std::array<T, SIZE> {p1[1], p1[0]},
                                        std::array<T, SIZE> {p2[1], p2[0]}}};

            const auto thetas = std::array<T, SIZE> {s0 - pi / 2, s0 + s1 - pi / 2};

            for (std::size_t i=0; i < SIZE; ++i) {
                const T l {0.0};
                const T r {link_lengths_[i]};
                const T t {0.1};
                const T b {-0.1};

                const auto x {xys[i][0]};
                const auto y {xys[i][1]};
                const auto th {thetas[i]};

                j_tf_[i].set_translation(point_2d{x,y});
                j_tf_[i].set_rotation(th);

                link_[i].set_vertices({{l,b}, {l, t}, {r, t}, {r, b}});

                viewer_.add_onetime(link_[i]);
                viewer_.add_onetime(circ_[i]);
            }

            return viewer_.render<mode == acrobot<T>::render_mode::rgb_array>();
        }

    protected:
        const T pi{std::numbers::pi_v<T>};
 
    private:
        const unsigned int screen_width_{500};
        const unsigned int screen_height_{500};

        const T LINK_LENGTH_1_;
        const T LINK_LENGTH_2_;

        const std::array<T, 2> link_lengths_{LINK_LENGTH_1_, LINK_LENGTH_2_};

        const T bound_{LINK_LENGTH_1_ + LINK_LENGTH_2_ + 0.2f}; // 2.2 for default

        Line line_{{-2.2, 1}, {2.2, 1}};

        static const std::size_t SIZE{2};
        std::array<Transform, SIZE> j_tf_{};
        std::array<FilledPolygon, SIZE> link_{};
        std::array<FilledPolygon, SIZE> circ_{make_circle(0.1), make_circle(0.1)};

        Viewer<> viewer_{"C++ Gym - Acrobot", screen_width_, screen_height_};
};

}
