#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>
#include <memory>

#include <ttns_lib/sop/operator_dictionaries/spin_operator.hpp>
#include <ttns_lib/ttn/ttn.hpp>

#include "../../helper_functions.hpp"


TEST_CASE("spin operators", "[sop]")
{
    using namespace ttns;
    using namespace spin;

    using T = std::complex<double>;

    SECTION("We can build operators for spin S=1/2")
    {
        SECTION("We can build spin S_m operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_m<T>>();
        
            SECTION("We can build a spin S_m operator acting on a single mode.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 1);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(2,2);    Mres.fill_zeros();      Mres(1, 0) = T(1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(1.0)));
                    linalg::csr_matrix<T> Mres(coo, 2,2);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
        
            SECTION("We can build a spin S_m operator acting on the first of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(2, 0) = T(1.0);    Mres(3, 1) = T(1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 0, T(1.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 1, T(1.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
            SECTION("We can build a spin S_m operator acting on the second of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(1, 0) = T(1.0);    Mres(3, 2) = T(1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 1,  M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(1.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 2, T(1.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 1, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 1, M));
                }
            }
        }
        
        SECTION("We can build spin S_p operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_p<T>>();
        
            SECTION("We can build a spin S_p operator acting on a single mode.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 1);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(2,2);    Mres.fill_zeros();      Mres(0, 1) = T(1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(1.0)));
                    linalg::csr_matrix<T> Mres(coo, 2,2);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
        
            SECTION("We can build a spin S_p operator acting on the first of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(0, 2) = T(1.0);    Mres(1, 3) = T(1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 2, T(1.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 3, T(1.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
            SECTION("We can build a spin S_p operator acting on the second of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(0, 1) = T(1.0);    Mres(2, 3) = T(1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 1,  M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(1.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 3, T(1.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 1, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 1, M));
                }
            }
        }
        
        SECTION("We can build spin S_x operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_x<T>>();
        
            SECTION("We can build a spin S_x operator acting on a single mode.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 1);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(2,2);    Mres.fill_zeros();      Mres(1, 0) = T(1.0/2.0);      Mres(0, 1) = T(1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 2,2);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
        
            SECTION("We can build a spin S_x operator acting on the first of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(2, 0) = T(1.0/2.0);    Mres(3, 1) = T(1.0/2.0);    Mres(0, 2) = T(1.0/2.0);    Mres(1, 3) = T(1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 2, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 3, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 0, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 1, T(1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
            SECTION("We can build a spin S_x operator acting on the second of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(1, 0) = T(1.0/2.0);    Mres(3, 2) = T(1.0/2.0);      Mres(0, 1) = T(1.0/2.0);    Mres(2, 3) = T(1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 1,  M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 3, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 2, T(1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 1, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 1, M));
                }
            }
        }
        
        SECTION("We can build spin S_y operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_y<T>>();
        
            SECTION("We can build a spin S_y operator acting on a single mode.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 1);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(2,2);    Mres.fill_zeros();      Mres(1, 0) = T(0, 1.0/2.0);      Mres(0, 1) = T(0, -1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(0, 1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(0, -1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 2,2);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
        
            SECTION("We can build a spin S_y operator acting on the first of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(2, 0) = T(0, 1.0/2.0);    Mres(3, 1) = T(0, 1.0/2.0);    Mres(0, 2) = T(0, -1.0/2.0);    Mres(1, 3) = T(0, -1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 2, T(0, -1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 3, T(0, -1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 0, T(0, 1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 1, T(0, 1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        
            SECTION("We can build a spin S_y operator acting on the second of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(1, 0) = T(0.0, 1.0/2.0);    Mres(3, 2) = T(0.0, 1.0/2.0);      Mres(0, 1) = T(0.0, -1.0/2.0);    Mres(2, 3) = T(0.0, -1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 1,  M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(0.0,-1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(0.0, 1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 3, T(0.0,-1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 2, T(0.0, 1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 1, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 1, M));
                }
            }
        }
        
        SECTION("We can build spin S_z operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_z<T>>();
        
            SECTION("We can build a spin S_z operator acting on a single mode.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 1);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(2,2);    Mres.fill_zeros();      Mres(0, 0) = T(1.0/2.0);    Mres(1, 1) = T(-1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 0, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 1, T(-1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 2,2);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct diagonal matrix representation.")
                {
                    linalg::diagonal_matrix<T> Mres(2,2);   Mres(0, 0) = T(1/2.0);  Mres(1, 1) = T(-1/2.0);
                    linalg::diagonal_matrix<T> M;
                    op->as_diagonal(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
            }
        
        
            SECTION("We can build a spin S_z operator acting on the first of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(0, 0) = T(1.0/2.0);    Mres(1, 1) = T(1.0/2.0);    Mres(2, 2) = T(-1.0/2.0);   Mres(3, 3) = T(-1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 0, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 1, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 2, T(-1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 3, T(-1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct diagonal matrix representation.")
                {
                    linalg::diagonal_matrix<T> Mres(4, 4);   Mres(0, 0) = T(1/2.0);  Mres(1, 1) = T(1/2.0); Mres(2,2) = T(-1/2.0); Mres(3, 3) = T(-1/2.0);
                    linalg::diagonal_matrix<T> M;
                    op->as_diagonal(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
            }
        
            SECTION("We can build a spin S_z operator acting on the second of two modes.")
            {
        
                std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 2);
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(4,4);    Mres.fill_zeros();      Mres(0, 0) = T(1.0/2.0);    Mres(1, 1) = T(-1.0/2.0);  Mres(2, 2) = T(1.0/2.0);     Mres(3, 3) = T(-1.0/2.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 1,  M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 0, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 1, T(-1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 2, T(1.0/2.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(3, 3, T(-1.0/2.0)));
                    linalg::csr_matrix<T> Mres(coo, 4,4);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 1, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct diagonal matrix representation.")
                {
                    linalg::diagonal_matrix<T> Mres(4, 4);   Mres(0, 0) = T(1/2.0);  Mres(1, 1) = T(-1/2.0); Mres(2,2) = T(1/2.0); Mres(3, 3) = T(-1/2.0);
                    linalg::diagonal_matrix<T> M;
                    op->as_diagonal(basis, 1, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
            }
        }
    }

    SECTION("We can build operators for spins S=1")
    {
        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(3, 1);
        SECTION("We can build spin S_m operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_m<T>>();
        
            SECTION("We can build a spin S_m operator acting on a single mode.")
            {
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(3,3);    Mres.fill_zeros();      Mres(1, 0) = T(std::sqrt(2.0)); Mres(2, 1) = T(std::sqrt(2.0));
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 1, T(std::sqrt(2.0))));
                    linalg::csr_matrix<T> Mres(coo, 3,3);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        }
        
        SECTION("We can build spin S_p operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_p<T>>();
        
            SECTION("We can build a spin S_p operator acting on a single mode.")
            {
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(3,3);    Mres.fill_zeros();      Mres(0, 1) = T(std::sqrt(2.0)); Mres(1, 2) = T(std::sqrt(2.0));
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 2, T(std::sqrt(2.0))));
                    linalg::csr_matrix<T> Mres(coo, 3,3);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        }
        
        SECTION("We can build spin S_x operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_x<T>>();
        
            SECTION("We can build a spin S_x operator acting on a single mode.")
            {
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(3,3);    Mres.fill_zeros();      Mres(1, 0) = T(1.0/std::sqrt(2.0));      Mres(0, 1) = T(1.0/std::sqrt(2.0));    Mres(2, 1) = T(1.0/std::sqrt(2.0));      Mres(1, 2) = T(1.0/std::sqrt(2.0));
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(1.0/std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(1.0/std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 1, T(1.0/std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 2, T(1.0/std::sqrt(2.0))));
                    linalg::csr_matrix<T> Mres(coo, 3,3);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        }
        
        SECTION("We can build spin S_y operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_y<T>>();
        
            SECTION("We can build a spin S_y operator acting on a single mode.")
            {
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(3,3);    Mres.fill_zeros();      Mres(1, 0) = T(0, 1.0/std::sqrt(2.0));      Mres(0, 1) = T(0, -1.0/std::sqrt(2.0));    Mres(2, 1) = T(0, 1.0/std::sqrt(2.0));      Mres(1, 2) = T(0, -1.0/std::sqrt(2.0));
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 0, T(0, 1.0/std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 1, T(0,-1.0/std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 1, T(0, 1.0/std::sqrt(2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 2, T(0,-1.0/std::sqrt(2.0))));
                    linalg::csr_matrix<T> Mres(coo, 3,3);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It cannot be constructed as a diagonal matrix.")
                {
                    linalg::diagonal_matrix<T> M;
                    REQUIRE_THROWS(op->as_diagonal(basis, 0, M));
                }
            }
        }
        
        SECTION("We can build spin S_z operators")
        {
            std::shared_ptr<single_site_operator<T>> op = std::make_shared<S_z<T>>();
        
            SECTION("We can build a spin S_z operator acting on a single mode.")
            {
        
                SECTION("It has the correct dense matrix representation.")
                {
                    linalg::matrix<T> Mres(3, 3);    Mres.fill_zeros();      Mres(0, 0) = T(1.0);    Mres(2, 2) = T(-1.0);
                    linalg::matrix<T> M;
                    op->as_dense(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct sparse matrix representation.")
                {
                    using index_type = typename linalg::csr_matrix<T>::index_type;
                    std::vector<std::tuple<index_type, index_type, T>> coo;
                    coo.push_back(std::make_tuple<index_type, index_type, T>(0, 0, T(1.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(1, 1, T(0.0)));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(2, 2, T(-1.0)));
                    linalg::csr_matrix<T> Mres(coo, 3,3);
                    linalg::csr_matrix<T> M;
                    op->as_csr(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
        
                SECTION("It has the correct diagonal matrix representation.")
                {
                    linalg::diagonal_matrix<T> Mres(3,3);   Mres(0, 0) = T(1.0);  Mres(1, 1) = T(0);  Mres(2,2) = T(-1.0);
                    linalg::diagonal_matrix<T> M;
                    op->as_diagonal(basis, 0, M);
                    REQUIRE(matrix_close(M, Mres, 1e-12));
                }
            }
        }   
    }
}
