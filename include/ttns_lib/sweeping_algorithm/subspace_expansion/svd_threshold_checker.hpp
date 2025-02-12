#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SVD_THRESHOLD_CHECKER_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SVD_THRESHOLD_CHECKER_HPP

#include <linalg/linalg.hpp>
#include "../../ttn/orthogonality/decomposition_engine.hpp"

template <typename T, typename backend>
struct svd_threshold_checker;

template <tyename T>
struct svd_threshold_checker<T, linalg::blas_backend>
{
    using size_type = typename linalg::blas_backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

public:
    static inline size_type terms_to_add(const linalg::diagonal_matrix<T, linalg::blas_backend> &S, size_type neigs, real_type svd_scale,
                                         orthogonality::truncation_mode trunc_mode = orthogonality::truncation_mode::singular_values_truncation)
    {
        size_type nadd = 0;
        for (size_type i = 0; i < neigs; ++i)
        {
            real_type sv = 0;
            // check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
            if (trunc_mode == orthogonality::truncation_mode::singular_values_truncation)
            {
                if (linalg::real(S(i, i)) > 0)
                {
                    sv = std::sqrt(linalg::real(S(i, i))) * svd_scale;
                }
                if (!std::isnan(sv))
                {
                    if (sv > m_spawning_threshold)
                    {
                        ++nadd;
                    }
                }
            }
            else
            {
                if (linalg::real(S(i, i)) > 0)
                {
                    sv = linalg::real(S(i, i)) * svd_scale * svd_scale;
                }
                if (!std::isnan(sv))
                {
                    if (sv > m_spawning_threshold)
                    {
                        ++nadd;
                    }
                }
            }
        }
        return nadd;
    }
};

#ifdef PYTTN_BUILD_CUDA
template <tyename T>
struct svd_threshold_checker<T, linalg::cuda_backend>
{
    using size_type = typename linalg::cuda_backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

public:
    static inline size_type terms_to_add(const linalg::diagonal_matrix<T, linalg::blas_backend> &S, size_type neigs, real_type svd_scale,
                                         orthogonality::truncation_mode trunc_mode = orthogonality::truncation_mode::singular_values_truncation)
    {
        size_type nadd = 0;
        RAISE_EXCEPTION("SVD threshold checker not implemented for cuda backend.")
        return nadd;
    }
};

#endif //TTNS_LIB_SWEEPING_ALGORITHM_SVD_THRESHOLD_CHECKER_HPP
