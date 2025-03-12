#ifndef LINALG_DECOMPOSITIONS_QR_HPP
#define LINALG_DECOMPOSITIONS_QR_HPP

#include "../../utils/exception_handling.hpp"
#include "qr_blas.hpp"

namespace linalg
{
template <typename matrix_type>
class qr<matrix_type, typename std::enable_if<is_dense_matrix<matrix_type>::value, void>::type> : public internal::dense_matrix_qr<matrix_type>
{
public:
    using base_type = internal::dense_matrix_qr<matrix_type>;
    template <typename ... Args>  qr(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};


template <typename matrix_type>
class lq<matrix_type, typename std::enable_if<is_dense_matrix<matrix_type>::value, void>::type> : public internal::dense_matrix_lq<matrix_type>
{
public:
    using base_type = internal::dense_matrix_lq<matrix_type>;
    template <typename ... Args>  lq(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}catch(...){throw;}
};
}   //namespace linalg

#endif //LINALG_DECOMPOSITIONS_QR_HPP




