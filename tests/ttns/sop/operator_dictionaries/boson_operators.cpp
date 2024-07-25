#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>
#include <memory>

#include <ttns_lib/sop/operator_dictionaries/bosonic_operator.hpp>
#include <ttns_lib/ttn/ttn.hpp>

#include "../../helper_functions.hpp"


TEST_CASE("boson operators", "[sop]")
{
    using namespace ttns;
    using namespace boson;

    using T = std::complex<double>;
    size_t ndim = 4;

    SECTION("We can build bosonic creation operators")
    {
        std::shared_ptr<single_site_operator<T>> op = std::make_shared<creation<T>>();

        SECTION("We can build a bosonic creation operator acting on a single mode.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 1);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim,ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    Mres(i, i-1) = std::sqrt(i);
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;
                for(size_t i = 1; i < ndim; ++i)
                {
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i, i-1, T(std::sqrt(i)))); ;
                }
                
                linalg::csr_matrix<T> Mres(coo, ndim, ndim);
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


        SECTION("We can build a bosonic creation operator acting on the first of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        Mres((i*ndim)+j, (i-1)*ndim+j) = std::sqrt(i);
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((i*ndim)+j, (i-1)*ndim+j, T(std::sqrt(i))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

        SECTION("We can build a bosonic creation operator acting on the second of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        Mres((j*ndim)+i, j*ndim+(i-1)) = std::sqrt(i);
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+i, j*ndim+(i-1), T(std::sqrt(i))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

    SECTION("We can build bosonic annihilation operators")
    {
        std::shared_ptr<single_site_operator<T>> op = std::make_shared<annihilation<T>>();

        SECTION("We can build a bosonic annihilation operator acting on a single mode.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 1);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim,ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    Mres(i-1, i) = std::sqrt(i);
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;
                for(size_t i = 1; i < ndim; ++i)
                {
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i-1, i, T(std::sqrt(i)))); ;
                }
                
                linalg::csr_matrix<T> Mres(coo, ndim, ndim);
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


        SECTION("We can build a bosonic annihilation operator acting on the first of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        Mres((i-1)*ndim+j, (i)*ndim+j) = std::sqrt(i);
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((i-1)*ndim+j, i*ndim+j, T(std::sqrt(i))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

        SECTION("We can build a bosonic annihilation operator acting on the second of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        Mres((j*ndim)+i-1, j*ndim+i) = std::sqrt(i);
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+i-1, j*ndim+i, T(std::sqrt(i))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

    SECTION("We can build bosonic number operators")
    {
        std::shared_ptr<single_site_operator<T>> op = std::make_shared<number<T>>();

        SECTION("We can build a bosonic number operator acting on a single mode.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 1);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim,ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    Mres(i, i) = i;
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;
                for(size_t i = 1; i < ndim; ++i)
                {
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i, i, T(i))); ;
                }
                
                linalg::csr_matrix<T> Mres(coo, ndim, ndim);
                linalg::csr_matrix<T> M;
                op->as_csr(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct diagonal matrix representation.")
            {
                linalg::diagonal_matrix<T> Mres(ndim,ndim);    
                for(size_t i = 0; i < ndim; ++i)
                {
                    Mres(i, i) = i;
                }
                linalg::diagonal_matrix<T> M;
                op->as_diagonal(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }
        }


        SECTION("We can build a bosonic number operator acting on the first of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        Mres(i*ndim+j, i*ndim+j) = i;
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>(i*ndim+j, i*ndim+j, T(i)));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
                linalg::csr_matrix<T> M;
                op->as_csr(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }


            SECTION("It has the correct diagonal matrix representation.")
            {
                linalg::diagonal_matrix<T> Mres(ndim*ndim,ndim*ndim);      
                for(size_t i = 0; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        Mres((i*ndim)+j, i*ndim+j) = i;
                    }
                }
                linalg::diagonal_matrix<T> M;
                op->as_diagonal(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }
        }

        SECTION("We can build a bosonic number operator acting on the second of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        Mres((j*ndim)+i, j*ndim+i) = i;
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+i, j*ndim+i, T(i)));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
                linalg::csr_matrix<T> M;
                op->as_csr(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct diagonal matrix representation.")
            {
                linalg::diagonal_matrix<T> Mres(ndim*ndim,ndim*ndim); 
                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 0; i < ndim; ++i)
                    {
                        Mres((j*ndim)+i, j*ndim+i) = i;
                    }
                }
                linalg::diagonal_matrix<T> M;
                op->as_diagonal(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }
        }
    }

    SECTION("We can build bosonic position operators")
    {
        std::shared_ptr<single_site_operator<T>> op = std::make_shared<position<T>>();

        SECTION("We can build a bosonic position operator acting on a single mode.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 1);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim,ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    Mres(i, i-1) = std::sqrt(i/2.0);
                    Mres(i-1, i) = std::sqrt(i/2.0);
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;
                for(size_t i = 1; i < ndim; ++i)
                {
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i, i-1, T(std::sqrt(i/2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i-1, i, T(std::sqrt(i/2.0))));
                }
                
                linalg::csr_matrix<T> Mres(coo, ndim, ndim);
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


        SECTION("We can build a bosonic position operator acting on the first of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        Mres(i*ndim+j, (i-1)*ndim+j) = std::sqrt(i/2.0);
                        Mres((i-1)*ndim+j, i*ndim+j) = std::sqrt(i/2.0);
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>(i*ndim+j, (i-1)*ndim+j, T(std::sqrt(i/2.0))));
                        coo.push_back(std::make_tuple<index_type, index_type, T>((i-1)*ndim+j, i*ndim+j, T(std::sqrt(i/2.0))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

        SECTION("We can build a bosonic position operator acting on the second of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        Mres((j*ndim)+i, j*ndim+(i-1)) = std::sqrt(i/2.0);
                        Mres((j*ndim)+i-1, j*ndim+i) = std::sqrt(i/2.0);
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+i, j*ndim+(i-1), T(std::sqrt(i/2.0))));
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+(i-1), j*ndim+i, T(std::sqrt(i/2.0))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

    SECTION("We can build bosonic momentum operators")
    {
        std::shared_ptr<single_site_operator<T>> op = std::make_shared<momentum<T>>();

        SECTION("We can build a bosonic momentum operator acting on a single mode.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 1);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim,ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    Mres(i, i-1) = T(0, std::sqrt(i/2.0));
                    Mres(i-1, i) = T(0,-std::sqrt(i/2.0));
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;
                for(size_t i = 1; i < ndim; ++i)
                {
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i, i-1, T(0, std::sqrt(i/2.0))));
                    coo.push_back(std::make_tuple<index_type, index_type, T>(i-1, i, T(0,-std::sqrt(i/2.0))));
                }
                
                linalg::csr_matrix<T> Mres(coo, ndim, ndim);
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


        SECTION("We can build a bosonic momentum operator acting on the first of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        Mres(i*ndim+j, (i-1)*ndim+j) =T(0, std::sqrt(i/2.0));
                        Mres((i-1)*ndim+j, i*ndim+j) =T(0,-std::sqrt(i/2.0));
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 0, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t i = 1; i < ndim; ++i)
                {
                    for(size_t j = 0; j < ndim; ++j)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>(i*ndim+j, (i-1)*ndim+j, T(0, std::sqrt(i/2.0))));
                        coo.push_back(std::make_tuple<index_type, index_type, T>((i-1)*ndim+j, i*ndim+j, T(0,-std::sqrt(i/2.0))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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

        SECTION("We can build a bosonic momentum operator acting on the second of two modes.")
        {

            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(ndim, 2);

            SECTION("It has the correct dense matrix representation.")
            {
                linalg::matrix<T> Mres(ndim*ndim,ndim*ndim);    Mres.fill_zeros();      
                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        Mres((j*ndim)+i, j*ndim+i-1) = T(0, std::sqrt(i/2.0));
                        Mres((j*ndim)+i-1, j*ndim+i) = T(0,-std::sqrt(i/2.0));
                    }
                }
                linalg::matrix<T> M;
                op->as_dense(basis, 1, M);
                REQUIRE(matrix_close(M, Mres, 1e-12));
            }

            SECTION("It has the correct sparse matrix representation.")
            {
                using index_type = typename linalg::csr_matrix<T>::index_type;
                std::vector<std::tuple<index_type, index_type, T>> coo;

                for(size_t j = 0; j < ndim; ++j)
                {
                    for(size_t i = 1; i < ndim; ++i)
                    {
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+i, j*ndim+(i-1), T(0, std::sqrt(i/2.0))));
                        coo.push_back(std::make_tuple<index_type, index_type, T>((j*ndim)+(i-1), j*ndim+i, T(0,-std::sqrt(i/2.0))));
                    }
                }
                linalg::csr_matrix<T> Mres(coo, ndim*ndim, ndim*ndim);
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
}
