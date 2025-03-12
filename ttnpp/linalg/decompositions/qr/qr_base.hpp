#ifndef LINALG_DECOMPOSITIONS_QR_BASE_HPP
#define LINALG_DECOMPOSITIONS_QR_BASE_HPP

#include "../decompositions_common.hpp"

namespace linalg
{
namespace internal
{
//a function for validating the size of the result type of the eigensolvers
struct qr_result_validation
{
    template <typename matrix_type, typename vecs_type>
    static inline typename std::enable_if<traits<vecs_type>::is_mutable, void>::type sim_trans(const matrix_type& mat, vecs_type& vecs)
    {
        if(vecs.shape(0) != mat.shape(0) || vecs.shape(1) != mat.shape(1))
        {
            //print out an info statement indicating that the eigensolver is resizing the eigenvector matrix.
            CALL_AND_HANDLE(vecs.resize(mat.shape(0), mat.shape(1)), "Failed to reshape the eigenvectors matrix.");   
        }
    }
    template <typename matrix_type, typename vecs_type>
    static inline typename std::enable_if<!traits<vecs_type>::is_mutable, void>::type sim_trans(const matrix_type& mat, vecs_type& vecs)
    {ASSERT(vecs.shape(0) == mat.shape(0) && vecs.shape(1) == mat.shape(1), "The input eigenvectors matrix does not have the correct shape.");}

};  //struct eigensolver_result_validation


template <typename T, typename backend> struct qr_helper;
template <typename matrix_type, typename enabler = void> class dense_matrix_qr;
template <typename matrix_type, typename enabler = void> class dense_matrix_lq;
}   //namespace internal
}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_QR_BASE_HPP

