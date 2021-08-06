#include <chrono>
#include <concepts>
#include <deque>
#include <iostream>
#include <type_traits>

#include <torch/torch.h>

#include "core.h"
#include "envs/classic_control/cartpole.h"

#include "epsilon_generator.h"

class DQNImpl : public torch::nn::Module {
    public:
        DQNImpl(size_t input_layer_size, size_t hidden_layer_size, size_t output_layer_size)
            : fc1_(input_layer_size, hidden_layer_size),
              fc2_(hidden_layer_size, output_layer_size) {}

        torch::Tensor forward(torch::Tensor x) {
            auto output = torch::tanh(fc1_(x));
            return fc2_(output);
        }

    private:
        torch::nn::Linear fc1_;
        torch::nn::Linear fc2_;
};

TORCH_MODULE(DQN);

template<typename Env, typename Device, typename Network, typename Criterion, typename Optimizer, std::floating_point T=float>
class dqn_agent {
public:
    using action_type = Env::action_type;
    using observation_type = Env::observation_type;
    using reward_type = Env::reward_type;

    dqn_agent(Env& env, Device& device, Network& dqn, Criterion criterion, Optimizer optimizer, const T gamma_discount_factor, cppcoro::generator<T> epsilon_decay)
        : env_(env),
          device_(device),
          dqn_(dqn), //dqn_(std::move(dqn)),
          criterion_(std::move(criterion)),
          optimizer_(std::move(optimizer)),
          gamma_discount_factor_(gamma_discount_factor),
          epsilon_decay_(std::move(epsilon_decay))
    {
        dqn_->to(device);
    }

    action_type get_action(observation_type& obs) {

        auto select_action = [this](observation_type& obs) -> action_type {
            torch::NoGradGuard no_grad;
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto state = torch::from_blob(obs.data(), {static_cast<long>(obs.size())}, options); // TODO: any way to avoid static_cast<>?
            state.to(device_);
            auto q_values = dqn_->forward(state);
            auto tensors = torch::max(q_values, 0);
            auto action_tensor = std::get<1>(tensors);
            //std::cout << "get_action: action_tensor:\n";
            //std::cout << "ndimension: " << action_tensor.ndimension() << '\n';
            //std::cout << action_tensor << '\n';
            action_type action = action_tensor.item().toLong();
            //std::cout << "action_tensor value: " << action << '\n';
            return action;
        };

        if (training_) {
            static typename decltype(epsilon_decay_)::iterator epsilon_it = epsilon_decay_.begin();

            const auto random_value = torch::rand(1)[0].item<T>();

            const auto epsilon = *epsilon_it;

            ++epsilon_it;

            if (random_value > epsilon)
                return select_action(obs);

            return env_.action_space.sample();
        }

        return select_action(obs);
    }

    void learn(const observation_type& obs, const action_type& action, const reward_type& reward, const observation_type& next_obs, const bool done) {

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto state_tensor = torch::from_blob(const_cast<float*>(obs.data()), {static_cast<long>(obs.size())}, options);
        state_tensor.to(device_);

        auto next_state_tensor = torch::from_blob(const_cast<float*>(next_obs.data()), {static_cast<long>(next_obs.size())}, options);
        next_state_tensor.to(device_);

        auto reward_tensor = torch::tensor({reward});
        reward_tensor.to(device_);

        torch::Tensor target;

        if (done) {
            target = torch::tensor({reward});
        } else {
            auto new_state_values = dqn_->forward(next_state_tensor).detach();
            auto max_new_state_values = torch::max(new_state_values);
            target = reward + gamma_discount_factor_ * max_new_state_values;
        }

        auto predicted = dqn_->forward(state_tensor)[action].view(-1);

        auto loss = criterion_->forward(predicted, target);

        optimizer_.zero_grad();
        loss.backward();
        optimizer_.step();
    }

    bool is_training_mode() {
        return training_;
    }

    void set_training_mode(bool training) {
        training_ = training;
    }

private:
    Env& env_;
    Device& device_;
    Network& dqn_;
    Criterion criterion_;
    Optimizer optimizer_;
    T gamma_discount_factor_;
    cppcoro::generator<T> epsilon_decay_;
    bool training_{true};
};

template<typename A, gym::env E>
void train(A& agent, E& env, std::unsigned_integral auto max_episodes, std::unsigned_integral auto max_steps_per_episode) {

    using r = E::reward_type;

    r total_rewards{};
    typename std::deque<r>::size_type max_recent_count{100};
    std::deque<r> recent_rewards;
    recent_rewards.resize(max_recent_count);

    for (size_t e=0; e < max_episodes; ++e) {

        auto obs = env.reset();

        for (size_t i=0; i < max_steps_per_episode; ++i) {

            const auto action = agent.get_action(obs);
            const auto [next_obs, reward, done] = env.step(action);
            agent.learn(obs, action, reward, next_obs, done);
            obs = next_obs;

            total_rewards += reward;

            if (done)
                break;
        }

        if (recent_rewards.size() == max_recent_count)
            recent_rewards.pop_front();
        recent_rewards.push_back(total_rewards);
    }

    auto mean_reward = total_rewards / max_episodes;
    auto mean_reward_recent = std::reduce(recent_rewards.begin(), recent_rewards.end()) / recent_rewards.size();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mean Reward: " << mean_reward << '\n';
    std::cout << "Mean Reward (last " << recent_rewards.size() << " episodes)" << mean_reward_recent << '\n';
}

template<typename A, gym::env E>
auto test(/*const*/A& agent, E& env/*, std::unsigned_integral auto max_steps*/) -> E::reward_type {
    using namespace std::chrono_literals;

    agent.set_training_mode(true);

    typename E::reward_type total_reward{};

    auto obs = env.reset();
    env.render();

    while (true) {
        std::this_thread::sleep_for(50ms);
        const auto action = agent.get_action(obs);
        const auto [next_obs, reward, done] = env.step(action);
        obs = next_obs;
        total_reward += reward;
        env.render();
        if (done)
            break;
    }

    agent.set_training_mode(false);

    return total_reward;
}

/*template<std::floating_point T>
T next_epsilon(std::unsigned_integral auto steps) {
    
}*/

int main(int argc, char** argv) {

    using namespace std::string_literals;
    using namespace gym::envs::classic_control;

    init_display(argc, argv, 600u, 400u, 0, 0);

    const bool use_cuda = torch::cuda::is_available();

    std::cout << "use_cuda: " << std::boolalpha << use_cuda << '\n';

    //auto device = torch::device(use_cuda ? "cuda:0"s : "cpu"s);
    torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU);

    std::cout << "device: " << device << '\n';

    using T = float;

    constexpr size_t episodes{700};
    constexpr size_t steps_per_episode{1000};
    constexpr T alpha_learning_rate{0.01f};
    constexpr T gamma_discount_factor{0.98f};
    constexpr size_t hidden_layer_size{64};

    constexpr T epsilon = 0.9f;
    constexpr T epsilon_min = 0.02f;
    constexpr size_t epsilon_decay_rate = 500;

    cartpole env;

    constexpr uint32_t seed{1}; 
    env.seed(seed);
    torch::manual_seed(seed);

    const auto input_layer_size = env.observation_space.n();
    const auto output_layer_size = env.action_space.n();

    DQN dqn(input_layer_size, hidden_layer_size, output_layer_size);

    auto epsilon_generator = decay(epsilon, epsilon_min, epsilon_decay_rate);

    auto optimizer = torch::optim::Adam(dqn->parameters(), torch::optim::AdamOptions(alpha_learning_rate));

    auto agent = dqn_agent(env, device, /*std::move(*/dqn/*)*/,
                           torch::nn::MSELoss(), std::move(optimizer),
                           gamma_discount_factor, std::move(epsilon_generator));

    train(agent, env, episodes, steps_per_episode);

    using reward_type = decltype(env)::reward_type;

    reward_type total_rewards{};
    reward_type max_reward{-std::numeric_limits<reward_type>::infinity()};

    constexpr size_t test_episode_count{10};

    for (size_t i=0; i < test_episode_count; ++i) {
        auto reward = test(agent, env);
        if (reward > max_reward)
            max_reward = reward;
        total_rewards += reward;
        reward_type avg_reward{total_rewards / static_cast<reward_type>(i+1u)};

        std::cout << "Test Episode: " << (i+1) << ", reward: " << reward << ", max_reward: " << max_reward
                    << ", avg reward: " << avg_reward << ", total_rewards: " << total_rewards << '\n';
    }
}
