#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>

#include <utils/product_iterator.hpp>

TEST_CASE("product iterator", "[utils]")
{
    using namespace utils;
    
    SECTION("We can allocate a product iterator with many degrees of freedom.")
    {
        size_t nterms = 1;
        std::vector<std::vector<size_t>> test(4);
        size_t nstart = 2;
        for(size_t i = 0; i < 4; ++i)
        {
            test[i].reserve(nstart+i);
            nterms *= nstart+i;
            for(size_t j=0; j<nstart+i; ++j)
            {
                test[i].push_back(j+i+1);
            }
        }


        //allocate an array that stores the result of this
        std::vector<std::vector<size_t>> res(nterms);
        for(size_t t = 0; t < nterms; ++t){res[t].resize(4);}
        size_t t = 0;
        for(size_t i = 0; i < test[0].size(); ++i)
        {
            for(size_t j = 0; j < test[1].size(); ++j)
            {
                for(size_t k=0; k < test[2].size(); ++k)
                {
                    for(size_t l=0; l < test[3].size(); ++l)
                    {
                        res[t][0] = test[0][i];
                        res[t][1] = test[1][j];
                        res[t][2] = test[2][k];
                        res[t][3] = test[3][l];
                        ++t;
                    }
                }
            }
        }

        SECTION("We can get the beginning product iterate")
        {
            product_iterator<size_t> prod_b = prod_begin(test);
            REQUIRE(prod_b.size() == 4);

            SECTION("The product iterator state points to the begin objects of each array in test")
            {
                for(size_t c = 0; c < 4; ++c)
                {
                    REQUIRE(prod_b.state()[c] == test[c].begin());
                }
            }

            size_t c = 0;
            SECTION("Can iterate over the prod iter object.");
            {
                for(auto a : *prod_b)
                {
                    REQUIRE(a == res[0][c]);
                    ++c;
                }
            }
            SECTION("Can access elements in the prod iter object.")
            {
                for(c = 0; c < 4; ++c)
                {
                    REQUIRE(prod_b[c] == res[0][c]);
                }
            }
        }

        SECTION("We can get the end product iterate")
        {
            product_iterator<size_t> prod_b = prod_end(test);
            REQUIRE(prod_b.size() == 4);

            SECTION("The product iterator state points to the begin objects of each array in test")
            {
                for(size_t c = 0; c < 4; ++c)
                {
                    REQUIRE(prod_b.state()[c] == test[c].end());
                }
            }
        }

        SECTION("We can iterate over a product iterator object")
        {
            t = 0;
            for(product_iterator<size_t> prod_iter = prod_begin(test); prod_iter != prod_end(test); ++prod_iter)
            {
                size_t c = 0;
                for(auto a : *prod_iter)
                {
                    REQUIRE(a == res[t][c]);
                    ++c;
                }
                ++t;
            }

        }
    }
}
