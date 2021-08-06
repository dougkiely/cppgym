#include <algorithm>
#include <array>
//#include <chrono> //debug
#include <concepts>
#include <memory>
//#include <thread> //debug

#include "envs/classic_control/cartpole.h"

template<typename E, std::size_t discrete_bin_count, std::size_t action_size, std::floating_point T=float>
class Q_learning_agent {
    public:
        using action_type = E::action_type;
        using observation_type = E::observation_type;
        using reward_type = E::reward_type;
        using policy_type = int; /*std::array<
                              std::array<
                                std::array<
                                  std::array<action_type, discrete_bin_count+1>, // store the action with the highest Q-value
                                discrete_bin_count+1>,
                              discrete_bin_count+1>,
                            discrete_bin_count+1>;*/
        using Q_table_type = std::array<       // 1st dimension (5 dimensions total)
                               std::array<     // 2nd dimension
                                 std::array<   // 3rd dimension
                                   std::array< // 4th dimension
                                     std::array<T, action_size>, // store Q-value of each action in 5th dimension
                                   discrete_bin_count+1>,
                                 discrete_bin_count+1>,
                               discrete_bin_count+1>,
                             discrete_bin_count+1>;

        Q_learning_agent(/*E& env, */const T learning_rate, const T discount_factor, const T epsilon_decay_rate, const T epsilon_min)
            : /*env_(env), */alpha_(learning_rate), gamma_(discount_factor),
                epsilon_decay_rate_(epsilon_decay_rate), epsilon_min_(epsilon_min)//,
                  //Q_(std::make_unique<Q_table_type>())
        {

            /*Q_.reset(new Q_table_type());

            for (std::size_t i=0; i < (*Q_).size(); ++i)
                for (std::size_t j=0; j < (*Q_)[i].size(); ++j)
                    for (std::size_t k=0; k < (*Q_)[i][j].size(); ++k)
                        for (std::size_t l=0; l < (*Q_)[i][j][k].size(); ++l)
                            for (std::size_t m=0; m < (*Q_)[i][j][k][l].size(); ++m)
                                (*Q_)[i][j][k][l][m] = 4.4f;

            const auto obs_high = env.observation_space.high();
            obs_low_ = env.observation_space.low();

            bin_width_[0] = (obs_high[0] - obs_low_[0]) / discrete_bin_count;
            bin_width_[1] = (obs_high[1] - obs_low_[1]) / discrete_bin_count;
            bin_width_[2] = (obs_high[2] - obs_low_[2]) / discrete_bin_count;
            bin_width_[3] = (obs_high[3] - obs_low_[3]) / discrete_bin_count;*/
        }

        auto discretize(const observation_type& obs) const -> std::array<int, 4> {
            return {static_cast<int>((obs[0] - obs_low_[0]) / bin_width_[0]),
                    static_cast<int>((obs[1] - obs_low_[1]) / bin_width_[1]),
                    static_cast<int>((obs[2] - obs_low_[2]) / bin_width_[2]),
                    static_cast<int>((obs[3] - obs_low_[3]) / bin_width_[3])};
        }

        action_type get_action(const observation_type& obs) {
            const auto discretized_obs = discretize(obs);

            // perform epsilon-greedy action selection
            if (epsilon_ > epsilon_min_)
                epsilon_ -= epsilon_decay_rate_;

            if (uniform_dist_(rand_gen_) > epsilon_)
                return argmax((*Q_)[discretized_obs[0]][discretized_obs[1]][discretized_obs[2]][discretized_obs[3]]);
            else
                return 0; //env_.action_space.sample();
        }

        void learn(const observation_type& obs, action_type action, reward_type reward, const observation_type& next_obs) {
            auto discretized_obs = discretize(obs);
            auto& Q_actions = (*Q_)[discretized_obs[0]][discretized_obs[1]][discretized_obs[2]][discretized_obs[3]];

            const auto discretized_next_obs = discretize(next_obs);
            auto& Q_next_actions = (*Q_)[discretized_next_obs[0]][discretized_next_obs[1]][discretized_next_obs[2]][discretized_next_obs[3]];

            for (size_t i=0; i < action_size; ++i)
                std::cout << "action[" << i << "] q-value: " << Q_actions[i] << '\n';
                //std::cout << "action[" << i << "] q-value: " << Q_[discretized_obs[0]][discretized_obs[1]][discretized_obs[2]][discretized_obs[3]][action] << '\n';

            const auto td_target = reward + gamma_ * std::ranges::max(Q_next_actions);
            const auto td_error = td_target - Q_actions[action];
            Q_actions[action] += alpha_ * td_error;
        }

        T get_epsilon() const {
            return epsilon_;
        }

        auto get_policy() const -> std::unique_ptr<policy_type> {
            auto policy = std::make_unique<policy_type>();

            /*for (std::size_t i=0; i < (*Q_).size(); ++i)
                for (std::size_t j=0; j < (*Q_)[i].size(); ++j)
                    for (std::size_t k=0; k < (*Q_)[i][j].size(); ++k)
                        for (std::size_t l=0; l < (*Q_)[i][j][k].size(); ++l)
                            (*policy)[i][j][k][l] = argmax((*Q_)[i][j][k][l]);*/

            return policy;
        }

    protected:
        auto argmax(std::array<T, action_size> c) const {
            auto max = std::ranges::max_element(c);
            return std::ranges::distance(std::begin(c), max);
        }

    private:
        //E& env_;
        observation_type obs_low_;

        T alpha_;
        T gamma_;

        std::mt19937 rand_gen_;
        std::uniform_real_distribution<T> uniform_dist_{0.0, 1.0};

        T epsilon_{1.0};
        T epsilon_decay_rate_;
        T epsilon_min_;

        observation_type bin_width_;

        std::unique_ptr<Q_table_type> Q_; // store Q-value of each action
};

template<typename A, typename E>
auto train(A& agent, E& env, const size_t max_episodes) -> std::unique_ptr<typename A::policy_type> {
    typename E::reward_type max_reward = -std::numeric_limits<typename E::reward_type>::infinity();

    for (size_t e=0; e < max_episodes; ++e) {
        typename E::reward_type total_reward{};

        auto obs = env.reset();
        //env.render();

        while (true) {
            auto action = agent.get_action(obs);
            auto [next_obs, reward, done] = env.step(action);
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
auto test(A& agent, E& env, const typename A::policy_type* const policy) -> E::reward_type {
    typename E::reward_type total_reward{};

    auto obs = env.reset();
    env.render();

    while (true) {
        //std::this_thread::sleep_for(std::chrono::milliseconds(50));
        //const auto discretized_obs = agent.discretize(obs);
        //const auto action = (*policy)[discretized_obs[0]][discretized_obs[1]][discretized_obs[2]][discretized_obs[3]];
        const auto [i0, i1, i2, i3] = agent.discretize(obs);
        const auto action = (*policy); //(*policy)[i0][i1][i2][i3];
        const auto [next_obs, reward, done] = env.step(action);
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

    cartpole env;
    //auto env = std::make_unique<cartpole<>>();

    auto obs_high = env.observation_space.high();
    auto obs_low  = env.observation_space.low();

    std::cout << "cartpole.observation_space.high: ";
    for (const auto& obs : obs_high)
        std::cout << obs << ", ";
    std::cout << '\n';
    std::cout << "cartpole.observation_space.low : ";
    for (const auto& obs : obs_low)
        std::cout << obs << ", ";
    std::cout << '\n';

    constexpr size_t max_episodes{50000}; // 50,000
    constexpr size_t steps_per_episode{200}; // cartpole env max steps is 200
    constexpr size_t max_steps{max_episodes * steps_per_episode};
    constexpr T alpha_learning_rate{0.05f};
    constexpr T gamma_discount_factor{0.98f};
    constexpr unsigned int discrete_bin_count{30}; //30

    constexpr T epsilon_min{0.05f};
    constexpr T epsilon_decay_rate{500 * epsilon_min / max_steps};

    /*cartpole env;
    //auto env = std::make_unique<cartpole<>>();

    auto obs_high = env.observation_space.high();
    auto obs_low  = env.observation_space.low();

    std::cout << "cartpole.observation_space.high: ";
    for (const auto& obs : obs_high)
        std::cout << obs << ", ";
    std::cout << '\n';
    std::cout << "cartpole.observation_space.low : ";
    for (const auto& obs : obs_low)
        std::cout << obs << ", ";
    std::cout << '\n';*/

    constexpr std::size_t action_size{2}; //{decltype(env)::action_size};

    //auto agent = Q_learning_agent<cartpole<>/*decltype(env)*/, discrete_bin_count, action_size>(
    //                /*(*env.get()), */alpha_learning_rate, gamma_discount_factor, epsilon_decay_rate, epsilon_min);
/*
    const auto learned_policy = train(agent, (*env.get()), max_episodes);

    using reward_type = typename cartpole<>::reward_type; //decltype(env)::reward_type;

    reward_type total_rewards{};
    reward_type max_reward{-std::numeric_limits<reward_type>::infinity()};

    constexpr size_t test_episode_count{10};
 
    for (size_t i=0; i < test_episode_count; ++i) {
        reward_type reward = test(agent, (*env), learned_policy.get());
        if (reward > max_reward)
            max_reward = reward;
        total_rewards += reward;
        reward_type avg_reward{total_rewards / static_cast<reward_type>(i+1u)};

        std::cout << "Test Episode: " << (i+1) << ", reward: " << reward << ", max_reward; " << max_reward
                    << ", avg reward: " << avg_reward <<  ", total_rewards: " << total_rewards << '\n';
    }
*/
}
