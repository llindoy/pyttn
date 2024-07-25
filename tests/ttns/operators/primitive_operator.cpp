#include <catch2/catch_template_test_macros.hpp>
#include <ttns_lib/operators/primitive_operator.hpp>

#include <memory>
#include <random>

#include "../helper_functions.hpp"

TEMPLATE_TEST_CASE("IdentityOperator", "[template][operator]", double, std::complex<double>)
{
    using value_type = TestType;
    using real_type = typename linalg::get_real_type<value_type>::type;
    using namespace ttns;
    size_t size = 16;

    std::mt19937 rng;
    //setup the identity operator
    std::shared_ptr<ops::primitive<value_type>> op = std::make_shared<ops::identity<value_type>>(size);

    REQUIRE( op->is_identity());
    REQUIRE( op->is_resizable());
    REQUIRE( op->size()==size);

    SECTION("Identity operators can be resize.")
    {
        size_t size_new = 24;
        op->resize(size_new);

        REQUIRE( op->size()==size_new );
    }

    SECTION("Identity operators can be cloned.")
    {
        std::shared_ptr<ops::primitive<value_type>> op2 = op->clone();
        REQUIRE( op->is_identity());
        REQUIRE( op->is_resizable());
        REQUIRE( op->size()==size);
    }

    SECTION("Apply the identity operator to a vector returns the same vector.")
    {
        
        //initialise a random matrix
        linalg::vector<value_type> A = init_random<value_type>::vector(size, rng);

        //initialise the result vector
        linalg::vector<value_type> B;

        SECTION("Apply the identity operator to vector without time values.")
        {
            op->apply(A, B);
            
            REQUIRE(vector_close(A, B, 1e-12));
        }

        real_type t = 1;
        real_type dt = 0.05;
        SECTION("Apply the identity operator to vector with time values.")
        {
            op->apply(A, B, t, dt);
            
            REQUIRE(vector_close(A, B, 1e-12));
        }
    }

    SECTION("Apply the identity operator to a matrix returns the same matrix.")
    {
        size_t d2 = 14;
        
        //initialise a random matrix
        linalg::matrix<value_type> A = init_random<value_type>::matrix(size, d2, rng);

        //initialise the result vector
        linalg::matrix<value_type> B;

        SECTION("Apply the identity operator to matrix without time values.")
        {
            op->apply(A, B);

            REQUIRE(matrix_close(A, B, 1e-12));
        }

        real_type t = 1;
        real_type dt = 0.05;
        SECTION("Apply the identity operator to matrix with time values.")
        {
            op->apply(A, B, t, dt);
            
            REQUIRE(matrix_close(A, B, 1e-12));
        }
    }
}

