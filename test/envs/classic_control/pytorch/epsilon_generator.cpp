#include <iostream>

#include "epsilon_generator.h"

int main() {

    size_t count{10};

    for (auto x : decay(0.9f, 0.02f, 500u)) {
        std::cout << x << '\n';
        if (--count == 0)
            break;
    }

    auto decay_rate = decay(0.9f, 0.02f, 500u);
    auto decay_rate_it = decay_rate.begin();

    for (size_t i=0; i < 500; ++i) {
        auto x = *decay_rate_it;
        std::cout << x << '\n';
        ++decay_rate_it;
    }
}
