#include <algorithm>
#include <array>
#include <chrono> //debug
#include <concepts>
#include <thread> //debug

#include "envs/classic_control/mountain_car.h"

template<typename E, std::size_t discrete_bin_count, std::size_t action_size, std::floating_point T=float>
class Q_learning_agent {
    public:
        using action_type = E::action_type;
        using observation_type = E::observation_type;
        using reward_type = E::reward_type;
        using policy_type = std::array<std::array<action_type,
                                discrete_bin_count+1>, discrete_bin_count+1>;
        using Q_table_type = std::array<
                               std::array<
                                 std::array<T, action_size>, // store Q-value of each action
                               discrete_bin_count+1>,
                             discrete_bin_count+1>;

        Q_learning_agent(E& env, const T learning_rate, const T discount_factor, const T epsilon_decay_rate, const T epsilon_min)
            : env_(env), alpha_(learning_rate), gamma_(discount_factor),
                epsilon_decay_rate_(epsilon_decay_rate), epsilon_min_(epsilon_min) {

            for (std::size_t i=0; i < Q_.size(); ++i)
                for (std::size_t j=0; j < Q_[i].size(); ++j)
                    for (std::size_t k=0; k < Q_[i][j].size(); ++k)
                        Q_[i][j][k] = 0; 

            const auto obs_high = env.observation_space.high();
            obs_low_ = env.observation_space.low();

            bin_width_[0] = (obs_high[0] - obs_low_[0]) / discrete_bin_count;
            bin_width_[1] = (obs_high[1] - obs_low_[1]) / discrete_bin_count;
        }

        auto discretize(const observation_type& obs) const -> std::array<int, 2> {
            return {static_cast<int>((obs[0] - obs_low_[0]) / bin_width_[0]),
                    static_cast<int>((obs[1] - obs_low_[1]) / bin_width_[1])};
        }

        action_type get_action(const observation_type& obs) {
            const auto discretized_obs = discretize(obs);

            // perform epsilon-greedy action selection
            if (epsilon_ > epsilon_min_)
                epsilon_ -= epsilon_decay_rate_;

            if (uniform_dist_(rand_gen_) > epsilon_)
                return argmax(Q_[discretized_obs[0]][discretized_obs[1]]);
            else
                return env_.action_space.sample();
        }

        void learn(const observation_type& obs, const action_type& action, const reward_type& reward, const observation_type& next_obs) {
            const auto discretized_obs = discretize(obs);
            const auto discretized_next_obs = discretize(next_obs);
            const auto td_target = reward + gamma_ * std::ranges::max(Q_[discretized_next_obs[0]][discretized_next_obs[1]]);
            const auto td_error = td_target - Q_[discretized_obs[0]][discretized_obs[1]][action];
            Q_[discretized_obs[0]][discretized_obs[1]][action] += alpha_ * td_error;
        }

        T get_epsilon() const {
            return epsilon_;
        }

        policy_type get_policy() const {
            policy_type policy;

            for (std::size_t i=0; i < Q_.size(); ++i)
                for (std::size_t j=0; j < Q_[i].size(); ++j)
                    policy[i][j] = argmax(Q_[i][j]);

            return policy;
        }

    protected:
        auto argmax(std::array<T, action_size> c) const {
            auto max = std::ranges::max_element(c);
            return std::ranges::distance(std::begin(c), max);
        }

    private:
        E& env_;
        observation_type obs_low_;

        T alpha_;
        T gamma_;

        std::mt19937 rand_gen_;
        std::uniform_real_distribution<T> uniform_dist_{0.0, 1.0};

        T epsilon_{1.0};
        T epsilon_decay_rate_;
        T epsilon_min_;

        observation_type bin_width_;

        Q_table_type Q_{};
};

template<typename A, typename E>
auto train(A& agent, E& env, size_t max_episodes) -> A::policy_type {
    typename E::reward_type max_reward = -std::numeric_limits<typename E::reward_type>::infinity();

    for (size_t e=0; e < max_episodes; ++e) {
        typename E::reward_type total_reward{};

        auto obs = env.reset();
        //env.render();

        while (true) {
            const auto action = agent.get_action(obs);
            const auto [next_obs, reward, done] = env.step(action);
            agent.learn(obs, action, reward, next_obs);
            obs = next_obs;
            total_reward += reward;
            //env.render();
            if (done)
                break;
        }

        if (total_reward > max_reward)
            max_reward = total_reward;

        std::cout << "Episode: " << (e+1) << ", total reward: " << total_reward << ", max reward: " << max_reward
                    << ", epsilon: " << agent.get_epsilon() << '\n';
    }

    // policy
    return std::move(agent.get_policy());
}

template<typename A, typename E>
auto test(const A& agent, E& env, const typename A::policy_type& policy) -> E::reward_type {
    typename E::reward_type total_reward{};

    auto obs = env.reset();
    env.render();

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto discretized_obs = agent.discretize(obs);
        const auto action = policy[discretized_obs[0]][discretized_obs[1]];
        const auto [next_obs, reward, done] = env.step(action);
        const auto ch = getchar();
        obs = next_obs;
        total_reward += reward;
        env.render();
        if (done)
            break;
    }

    return total_reward;
}

int main(int argc, char** argv) {

    using namespace gym::envs::classic_control;

    init_display(argc, argv, 600u, 400u, 0, 0);

    using T = float;

    constexpr size_t max_episodes{50000}; // 50,000
    constexpr size_t steps_per_episode{200}; // mountain car env max steps is 200
    constexpr size_t max_steps{max_episodes * steps_per_episode};
    constexpr T alpha_learning_rate{0.05f};
    constexpr T gamma_discount_factor{0.98f};
    constexpr unsigned int discrete_bin_count{30}; //30

    constexpr T epsilon_min{0.05f};
    constexpr T epsilon_decay_rate{500 * epsilon_min / max_steps};

    mountain_car env;

    constexpr std::size_t action_size{decltype(env)::action_size};

    auto agent = Q_learning_agent<decltype(env), discrete_bin_count, action_size>(
                    env, alpha_learning_rate, gamma_discount_factor, epsilon_decay_rate, epsilon_min);

    const auto learned_policy = train(agent, env, max_episodes);

    using reward_type = decltype(env)::reward_type;

    reward_type total_rewards{};
    reward_type max_reward{-std::numeric_limits<reward_type>::infinity()};

    constexpr size_t test_episode_count{10};
 
    for (size_t i=0; i < test_episode_count; ++i) {
        reward_type reward = test(agent, env, learned_policy);
        if (reward > max_reward)
            max_reward = reward;
        total_rewards += reward;
        reward_type avg_reward{total_rewards / static_cast<reward_type>(i+1u)};

        std::cout << "Test Episode: " << (i+1) << ", reward: " << reward << ", max_reward: " << max_reward
                    << ", avg reward: " << avg_reward <<  ", total_rewards: " << total_rewards << '\n';
    }
}
