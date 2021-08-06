#include <array>
#include <iostream>
#include <memory>

void test_stack_based() {

    std::cout << "begin: test_stack_based\n";

    using T = float;

    constexpr size_t size_{2};
    constexpr size_t dim_size_{31};

    using multi_array = std::array<
                          std::array<
                            std::array<
                              std::array<
                                std::array<T, size_>,
                              dim_size_>,
                            dim_size_>,
                          dim_size_>,
                        dim_size_>;

    multi_array marr{};

    for (size_t i=0; i < marr.size(); ++i)
        for (size_t j=0; j < marr[i].size(); ++j)
            for (size_t k=0; k < marr[i][j].size(); ++k)
                for (size_t l=0; l < marr[i][j][k].size(); ++l)
                    for (size_t m=0; m < marr[i][j][k][l].size(); ++m) {
                        std::cout << "i[" << i << "],j[" << j << "],k[" << k << "],l[" << l << "],m[" << m << "]=" << marr[i][j][k][l][m] << '\n';
                        marr[i][j][k][l][m] = 4.0f;
                        std::cout << " i[" << i << "],j[" << j << "],k[" << k << "],l[" << l << "],m[" << m << "]=" << marr[i][j][k][l][m] << '\n';
                    }

    std::cout << "end: test_stack_based\n";
}

void test_heap_based() {

    std::cout << "begin: test_heap_based\n";

    using T = float;

    constexpr size_t size_{2};
    constexpr size_t dim_size_{31};

    using multi_array = std::array<
                          std::array<
                            std::array<
                              std::array<
                                std::array<T, size_>,
                              dim_size_>,
                            dim_size_>,
                          dim_size_>,
                        dim_size_>;

    multi_array* marr = new multi_array{};

    for (size_t i=0; i < marr->size(); ++i)
        for (size_t j=0; j < (*marr)[i].size(); ++j)
            for (size_t k=0; k < (*marr)[i][j].size(); ++k)
                for (size_t l=0; l < (*marr)[i][j][k].size(); ++l)
                    for (size_t m=0; m < (*marr)[i][j][k][l].size(); ++m) {
                        std::cout << "i[" << i << "],j[" << j << "],k[" << k << "],l[" << l << "],m[" << m << "]=" << (*marr)[i][j][k][l][m] << '\n';
                        (*marr)[i][j][k][l][m] = 4.0f;
                        std::cout << " i[" << i << "],j[" << j << "],k[" << k << "],l[" << l << "],m[" << m << "]=" << (*marr)[i][j][k][l][m] << '\n';
                    }

    delete marr;

    std::cout << "end: test_heap_based\n";
}

void test_smart_pointer_based() {

    std::cout << "begin: test_smart_pointer_based\n";

    using T = float;

    constexpr size_t size_{2};
    constexpr size_t dim_size_{31};

    using multi_array = std::array<
                          std::array<
                            std::array<
                              std::array<
                                std::array<T, size_>,
                              dim_size_>,
                            dim_size_>,
                          dim_size_>,
                        dim_size_>;

    auto marr = std::make_unique<multi_array>();

    for (size_t i=0; i < marr->size(); ++i)
        for (size_t j=0; j < (*marr)[i].size(); ++j)
            for (size_t k=0; k < (*marr)[i][j].size(); ++k)
                for (size_t l=0; l < (*marr)[i][j][k].size(); ++l)
                    for (size_t m=0; m < (*marr)[i][j][k][l].size(); ++m) {
                        std::cout << "i[" << i << "],j[" << j << "],k[" << k << "],l[" << l << "],m[" << m << "]=" << (*marr)[i][j][k][l][m] << '\n';
                        (*marr)[i][j][k][l][m] = 4.0f;
                        std::cout << " i[" << i << "],j[" << j << "],k[" << k << "],l[" << l << "],m[" << m << "]=" << (*marr)[i][j][k][l][m] << '\n';
                    }

    std::cout << "end: test_smart_pointer_based\n";
}

int main() {

    std::cout << "begin: main\n";

    //test_stack_based();

    //test_heap_based();

    test_smart_pointer_based();

    std::cout << "end: main\n";

    return 0;
}
