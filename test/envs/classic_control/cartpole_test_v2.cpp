//#include <algorithm>
//#include <array>
//#include <chrono> //debug
//#include <concepts>
//#include <memory>
//#include <thread> //debug

#include "envs/classic_control/cartpole.h"
#include "spaces/box.h"
#include "spaces/discrete.h"

int main(int argc, char** argv) {

    using namespace gym::envs::classic_control;

    //init_display(argc, argv, 600u, 400u, 0, 0);

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

    return 0;
}
