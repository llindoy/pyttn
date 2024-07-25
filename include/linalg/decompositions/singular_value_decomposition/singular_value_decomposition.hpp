#ifndef LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_HPP
#define LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_HPP

#include <common/exception_handling.hpp>
#include "singular_value_decomposition_blas.hpp"
#include "singular_value_decomposition_cuda.hpp"

namespace linalg
{
//eigensolver object for hermitian matrix type that is not mutable
template <typename matrix_type, bool use_divide_and_conquer>
class singular_value_decomposition<matrix_type, use_divide_and_conquer, typename std::enable_if<is_dense_matrix<matrix_type>::value, void>::type> : public internal::dense_matrix_singular_value_decomposition<matrix_type, use_divide_and_conquer>
{
public:
    using base_type = internal::dense_matrix_singular_value_decomposition<matrix_type, use_divide_and_conquer>;
    template <typename ... Args>  singular_value_decomposition(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};
}   //namespace linalg

#endif //LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_HPP

