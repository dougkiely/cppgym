#pragma once

#include "core.h"

namespace gym::wrappers {

template<gym::env E>
class time_limit {
    public:
        const uint64_t action_size;
        const uint64_t observation_size;

        using action_space_type = E::action_space_type;
        using action_type = E::action_type;
        using observation_space_type = E::observation_space_type;
        using observation_type = E::observation_type;
        using reward_type = E::reward_type;
        using step_return_type = E::step_return_type;

        action_space_type& action_space;
        observation_space_type& observation_space;

        time_limit() : time_limit(200u) {}

        time_limit(std::size_t max_episode_steps) :
            action_size(env_.action_size),
            observation_size(env_.observation_size),
            action_space(env_.action_space),
            observation_space(env_.observation_space),
            max_episode_steps_(max_episode_steps) {}

        time_limit(const time_limit&)=delete;
        time_limit& operator=(const time_limit&)=delete;
        time_limit(time_limit&&)=default;
        time_limit& operator=(time_limit&&)=default;

        auto step(const action_type& action) -> step_return_type {
            // assert(elapsed_steps_);

            auto [obs, reward, done] = env_.step(action);
            ++elapsed_steps_;

            if (elapsed_steps_ >= max_episode_steps_) {
                // TODO: info TimeLimit.truncated = true
                done = true;
            }

            return {std::move(obs), reward, done};
        }

        auto reset() -> observation_type {
            elapsed_steps_ = 0;
            return std::move(env_.reset());
        }

        using render_mode = E::render_mode;

        auto render() {
            env_.render();
        }

        void close() {
            env_.close();
        }

        uint32_t seed(uint32_t value) {
            return env_.seed(value);
        }

    private:
        E env_;
        std::size_t max_episode_steps_{0};
        std::size_t elapsed_steps_{0};
};

}
