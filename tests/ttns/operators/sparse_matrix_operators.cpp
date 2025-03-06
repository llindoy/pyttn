#include <catch2/catch_template_test_macros.hpp>
#include <ttns_lib/operators/matrix_operators.hpp>

#include <memory>
#include <random>
#include <complex>

#include "../helper_functions.hpp"

TEMPLATE_TEST_CASE("SparseMatrixOperator", "[template][operator]", double, std::complex<double>)
{
    using value_type = TestType;
    using real_type = typename linalg::get_real_type<value_type>::type;
    using namespace ttns;
    size_t size = 16;

    std::mt19937 rng;

    linalg::csr_matrix<value_type> mat = init_random<value_type>::csr_matrix(size, size, rng);
    //setup the identity operator
    using mat_op = ops::sparse_matrix_operator<value_type>;
    std::shared_ptr<ops::primitive<value_type>> op = std::make_shared<mat_op>(mat);

    REQUIRE( !op->is_identity());
    REQUIRE( !op->is_resizable());
    REQUIRE( op->size()==size);

    SECTION("sparse matrix operators cannot be resized.")
    {
        size_t size_new = 24;
        CHECK_THROWS(op->resize(size_new));
    }

    SECTION("sparse matrix operators can be cloned.")
    {
        std::shared_ptr<ops::primitive<value_type>> op2 = op->clone();
        std::shared_ptr<mat_op> A  = std::dynamic_pointer_cast<mat_op>(op2);

        REQUIRE( !op->is_identity());
        REQUIRE( op->size()==size);
        REQUIRE(matrix_close(A->mat(), mat, 1e-12));
    }

    SECTION("Apply the ssparse matrix operator to a vector returns the sparse matrix acting on the vector.")
    {
        //initialise a random matrix
        linalg::vector<value_type> A = init_random<value_type>::vector(size, rng);

        //initialise the result vector
        linalg::vector<value_type> B;

        SECTION("Apply the sparse matrix operator to vector without time values.")
        {
            op->apply(A, B);
            linalg::vector<value_type> C = mat*A;
            
            REQUIRE(vector_close(B, C, 1e-12));
        }

        real_type t = 1;
        real_type dt = 0.05;
        SECTION("Apply the sparse matrix operator to vector with time values.")
        {
            op->apply(A, B, t, dt);
            
            linalg::vector<value_type> C = mat*A;
            
            REQUIRE(vector_close(B, C, 1e-12));
        }
    }

    SECTION("Apply the sparse matrix operator to a matrix returns the sparse matrix acting on the matrix.")
    {
        size_t d2 = 14;
        
        //initialise a random matrix
        linalg::matrix<value_type> A = init_random<value_type>::matrix(size, d2, rng);

        //initialise the result vector
        linalg::matrix<value_type> B;

        SECTION("Apply the sparse matrix operator to matrix without time values.")
        {
            op->apply(A, B);
            linalg::matrix<value_type> C = mat*A;

            REQUIRE(matrix_close(B, C, 1e-12));
        }

        real_type t = 1;
        real_type dt = 0.05;
        SECTION("Apply the sparse matrix operator to matrix with time values.")
        {
            op->apply(A, B, t, dt);
            linalg::matrix<value_type> C = mat*A;
            
            REQUIRE(matrix_close(B, C, 1e-12));
        }
    }
}
