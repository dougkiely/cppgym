#pragma once

#include <array>
#include <concepts>
//#include <iostream> //debug
#include <memory>
#include <ostream>
#include <string_view>

//#include "envs/classic_control/acrobot.h"
#include "envs/classic_control/cartpole.h"
//#include "envs/classic_control/continuous_mountain_car.h"
#include "envs/classic_control/mountain_car.h"
//#include "envs/classic_control/pendulum.h"
#include "wrappers/time_limit.h"

namespace gym {

static std::array<std::tuple<std::string, std::string>, 6> env_registry{
    std::tuple{"CartPole-v0", "cp-v0"},
    std::tuple{"CartPole-v1", "cp-v1"},
    std::tuple{"MountainCar-v0", "mc-v0"},
    std::tuple{"MountainCarContinuous-v0", "mcc-v0"},
    std::tuple{"Pendulum-v0", "p-v0"},
    std::tuple{"Acrobot-v1", "a-v1"}
};

template<std::size_t N1, std::size_t N2>
constexpr bool is_same_str(const char (&str1)[N1], const char (&str2)[N2]) {
    if (N1 != N2)
        return false;

    for (std::size_t i=0; i < N1; ++i)
        if (str1[i] != str2[i])
            return false;

    return true;
}

template<std::size_t N>
struct string_literal {
    constexpr string_literal(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }
    char value[N];

    template<std::size_t Size>
    constexpr bool is_same(const char (&str)[Size]) const {
        return is_same_str(value, str);
    }

    template<std::size_t Size>
    friend std::ostream& operator<<(std::ostream &, const string_literal<Size>&);
};

template<std::size_t N>
std::ostream& operator<<(std::ostream &os, const string_literal<N>& literal) {
    os << literal.value;
    return os;
}

template<string_literal str>
void view_string_literal() {
    //std::cout << "view_literal<string_literal>: size: " << sizeof(str) << ", value: " << str.value << '\n';
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "Acrobot-v1"))
auto make() -> gym::env auto {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return time_limit<acrobot<T>>(55u); //500u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "CartPole-v0"))
auto make() -> gym::env auto {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return time_limit<cartpole<T>>(77u); //200u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "CartPole-v1"))
auto make() -> gym::env auto {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return time_limit<cartpole<T>>(555u); //500u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "MountainCar-v0"))
auto make(/*int argc, char** argv*/) -> gym::env auto {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    /*const auto width{600u};
    const auto height{400u};
    const auto x{0};
    const auto y{0};

    init_display(argc, argv, width, height, x, y);*/

    return time_limit<mountain_car<T>>(44u); //200u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "MountainCarContinuous-v0"))
auto make() -> gym::env auto {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return time_limit<continuous_mountain_car<T>>(99u); //999u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "Pendulum-v0"))
auto make() -> gym::env auto {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return time_limit<pendulum<T>>(88u); //200u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "MountainCar-v0"))
auto make_unique() -> std::unique_ptr<gym::env auto> {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return std::make_unique<time_limit<mountain_car<>>>(45u); //200u
}

template<string_literal env_name, std::floating_point T=float> requires (is_same_str(env_name.value, "MountainCarContinuous-v0"))
auto make_unique() -> std::unique_ptr<gym::env auto> {

    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    return std::make_unique<time_limit<continuous_mountain_car<>>>(45u); //200u
}

/*template<string_literal env_name, std::floating_point T=float>
auto make() -> gym::env auto {
    return make<env_name, T>(0, nullptr);
}*/

/*template<typename T>
concept MountainCar_v0 = std::same_as<T, decltype(string_literal("MountainCar-v0"))> && requires (T &t) {
    t.env_name;
};

template<MountainCar_v0 literal>
auto make() -> gym::env auto {
    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    std::cout << "make MountainCar: size: " << sizeof(literal) << ", env_name: " << literal.env_name << '\n';

    return time_limit<mountain_car<>>(55u); //200u
}*/

auto make(std::string_view env_name) -> gym::env auto {

    using namespace std::string_literals;
    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    //for (const auto& e : envs)
    if (env_name == "MountainCar-v0")
        //return std::move(mountain_car<>()); // works
        //return time_limit<mountain_car<>>(200u); // works
        //return make<mountain_car<>, 0>();
        return time_limit<mountain_car<>>(200u);

    //if (env_name == "CartPole-v0")
        //return time_limit<cartpole<>>(200u);

    /*if (env_name == "CartPole-v1")
        return make<cartpole<>, 1>();

    if (env_name == "MountainCarContinuous-v0")
        return make<continuous_mountain_car<>, 0>();

    if (env_name == "Pendulum-v0")
        return make<pendulum<>, 0>();

    if (env_name == "Acrobot-v1")
        return make<acrobot<>, 1>();*/

    throw std::runtime_error("Unknown environment name: "s + env_name.data());
}

auto make_unique(std::string_view env_name) -> std::unique_ptr<gym::env auto> {

    using namespace std::string_literals;
    using namespace gym::envs::classic_control;
    using namespace gym::wrappers;

    //for (const auto& e : envs)
    if (env_name == "MountainCar-v0")
        //return std::move(mountain_car<>()); // works
        //return time_limit<mountain_car<>>(200u); // works
        //return make<mountain_car<>, 0>();
        return std::make_unique<time_limit<mountain_car<>>>(200u);

    //if (env_name == "CartPole-v0")
        //return std::make_unique<time_limit<cartpole<>>>(200u);

    /*if (env_name == "CartPole-v1")
        return make<cartpole<>, 1>();

    if (env_name == "MountainCarContinuous-v0")
        return make<continuous_mountain_car<>, 0>();

    if (env_name == "Pendulum-v0")
        return make<pendulum<>, 0>();

    if (env_name == "Acrobot-v1")
        return make<acrobot<>, 1>();*/

    throw std::runtime_error("Unknown environment name: "s + env_name.data());
}

}
