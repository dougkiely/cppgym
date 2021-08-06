#pragma once

#include <concepts>

namespace gym {

//using bigint = int64_t; // TODO: what data type should this be?

//template<typename T>
//concept env = std::default_initializable<T> && requires(T& t, Action& action) {
//
//    { t.step(T::action_type) } -> std::convertible_to<std::tuple<typename T::observation_type, typename T::reward_type, bool>>;
//
//    { reset() } -> std::convertible_to<typename T::observation_type>;
//
//    render(/*mode=human*/);
//
//    close();
//
//    //{ seed(/*seed=none*/); } -> std::convertible_to<std::array<bigint, 8>>;
//};

template<typename T>
concept env = std::default_initializable<T> && requires(T& t, const T::action_type& action, uint32_t seed, typename T::render_mode mode) {

    { t.action_size } -> std::convertible_to<uint64_t>;
    { t.observation_size } -> std::convertible_to<uint64_t>;

    typename T::action_space_type;
    typename T::action_type;
    typename T::observation_space_type;
    typename T::observation_type;
    typename T::reward_type;
    typename T::step_return_type;

    std::same_as<decltype(t.action_space), typename T::action_space_type>;
    std::same_as<decltype(t.observation_space), typename T::observation_space_type>;

    { t.step(action) } -> std::same_as<typename T::step_return_type>;

    { t.reset() } -> std::same_as<typename T::observation_type>;

    //t.render(); // TODO: what about template parameter and return type?

    t.close();

    { t.seed(seed) } -> std::convertible_to<decltype(seed)>;
};

}
