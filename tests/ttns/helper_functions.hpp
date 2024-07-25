#ifndef TESTS_TTNS_OPERATORS_HELPER_FUNCTIONS_HPP
#define TESTS_TTNS_OPERATORS_HELPER_FUNCTIONS_HPP

#include <linalg/linalg.hpp>
#include <random>
#include <algorithm>

template <typename T, typename RT>
bool vector_close(const linalg::vector<T>& A, const linalg::vector<T>& B, RT eps = RT(1e-12))
{
    if(A.shape() != B.shape()){return false;}
    for(size_t i = 0; i < A.shape(0); ++i)
    {
        if(std::abs(A(i) - B(i)) > eps){return false;}
    }
    return true;
}

template <typename T, typename RT>
bool matrix_close(const linalg::matrix<T>& A, const linalg::matrix<T>& B, RT eps = RT(1e-12))
{
    if(A.shape() != B.shape()){return false;}
    for(size_t i = 0; i < A.shape(0); ++i)
    {
        for(size_t j = 0; j<A.shape(1); ++j)
        {
            if(std::abs(A(i, j) - B(i, j)) > eps){return false;}
        }
    }
    return true;
}

template <typename T, typename RT>
bool matrix_close(const linalg::diagonal_matrix<T>& A, const linalg::diagonal_matrix<T>& B, RT eps = RT(1e-12))
{
    if(A.shape() != B.shape()){return false;}
    for(size_t i = 0; i < A.nnz(); ++i)
    {
        if(std::abs(A[i] - B[i]) > eps){return false;}
    }
    return true;
}

template <typename T, typename RT>
bool matrix_close(const linalg::csr_matrix<T>& A, const linalg::csr_matrix<T>& B, RT eps = RT(1e-12))
{
    if(A.nnz() != B.nnz() || A.nrows() != B.nrows() || A.ncols() != B.ncols()) {return false;}

    auto Ar = A.rowptr();   auto Br = B.rowptr();
    for(size_t i = 0; i < A.nrows(); ++i)
    {
        if(Ar[i] != Br[i]){return false;}
    }

       
    auto Ac = A.colind();   auto Bc = B.colind();
    auto Ab = A.buffer();   auto Bb = B.buffer();
    for(size_t i = 0; i < A.nnz(); ++i)
    {
        if(Ac[i] != Bc[i]){return false;}
        if(std::abs(Ab[i] - Bb[i]) > eps){return false;}
    }

    return true;
}


template <typename T>
class init_random
{
public:
    static linalg::vector<T> vector(size_t d1, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        linalg::vector<T> vec(d1, [&rng, &dist](size_t){return dist(rng);});
        return vec;
    }

    static linalg::matrix<T> matrix(size_t d1, size_t d2, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        linalg::matrix<T> mat(d1, d2, [&rng, &dist](size_t, size_t){return dist(rng);});
        return mat;
    }

    static linalg::diagonal_matrix<T> diagonal_matrix(size_t d1, size_t d2, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        linalg::diagonal_matrix<T> mat(d1, d2);
        for(size_t i = 0; i < (d1 < d2 ? d1 : d2); ++i)
        {
            mat[i] = dist(rng);
        }
        return mat;
    }

    static linalg::csr_matrix<T> csr_matrix(size_t d1, size_t d2, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        using coo_type = typename linalg::csr_matrix<T>::coo_type;
        using index_type = typename linalg::csr_matrix<T>::index_type;

        coo_type coo;

        //randomly generate the index types
        for(size_t i = 0; i < d1; ++i)
        {
            //get a vector of all the possible column indices
            std::vector<index_type> nd(d2);
            for(size_t j = 0; j < d2; ++j){nd[j] = static_cast<index_type>(j);}

            //shuffle the vector
            std::shuffle(std::begin(nd), std::end(nd), rng);

            //and add a random number of these terms to the coo array
            std::uniform_int_distribution<size_t> ud(1, d2);
            size_t nadd = ud(rng);
        
            for(size_t kx = 0; kx < nadd; ++kx)
            {
                index_type col = nd[kx];
                T data = dist(rng);
                coo.push_back(std::make_tuple(static_cast<index_type>(i), col, data));
            }
        }
    
        linalg::csr_matrix<T> mat(coo, d1, d2);
        return mat;
    }
};


template <typename T>
class init_random<std::complex<T>>
{
    using CT = std::complex<T>;
public:
    static linalg::vector<std::complex<T>> vector(size_t d1, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        linalg::vector<CT> vec(d1, [&rng, &dist](size_t){return CT(dist(rng), dist(rng));});
        return vec;
    }

    static linalg::matrix<CT> matrix(size_t d1, size_t d2, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        linalg::matrix<CT> mat(d1, d2, [&rng, &dist](size_t, size_t){return CT(dist(rng), dist(rng));});
        return mat;
    }

    static linalg::diagonal_matrix<CT> diagonal_matrix(size_t d1, size_t d2, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        linalg::diagonal_matrix<CT> mat(d1, d2);
        for(size_t i = 0; i < (d1 < d2 ? d1 : d2); ++i)
        {
            mat[i] = CT(dist(rng), dist(rng));
        }
        return mat;
    }

    static linalg::csr_matrix<CT> csr_matrix(size_t d1, size_t d2, std::mt19937& rng)
    {
        std::normal_distribution<T> dist(0, 1);
        using coo_type = typename linalg::csr_matrix<CT>::coo_type;
        using index_type = typename linalg::csr_matrix<CT>::index_type;

        coo_type coo;

        //randomly generate the index types
        for(size_t i = 0; i < d1; ++i)
        {
            //get a vector of all the possible column indices
            std::vector<index_type> nd(d2);
            for(size_t j = 0; j < d2; ++j){nd[j] = static_cast<index_type>(j);}

            //shuffle the vector
            std::shuffle(std::begin(nd), std::end(nd), rng);

            //and add a random number of these terms to the coo array
            std::uniform_int_distribution<size_t> ud(1, d2);
            size_t nadd = ud(rng);
        
            for(size_t kx = 0; kx < nadd; ++kx)
            {
                index_type col = nd[kx];
                CT data = CT(dist(rng), dist(rng));
                coo.push_back(std::make_tuple(static_cast<index_type>(i), col, data));
            }
        }
    
        linalg::csr_matrix<CT> mat(coo, d1, d2);
        return mat;
    }
};


#endif

