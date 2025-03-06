#ifndef LINALG_DECOMPOSITIONS_TRIDIAGONALISATION_BASE_HPP
#define LINALG_DECOMPOSITIONS_TRIDIAGONALISATION_BASE_HPP

#include "../decompositions_common.hpp"

namespace linalg
{
namespace internal
{
//a function for validating the size of the result type of the eigensolvers
struct tridiagonalisation_result_validation
{
    template <typename matrix_type, typename vals_type>
    static inline typename std::enable_if<is_symtridiag_matrix_type<vals_type>::value && traits<vals_type>::is_resizable, void>::type tridiag(const matrix_type& mat, vals_type& eigs)
    {
        if((eigs.shape(0) != eigs.shape(1)) || (mat.shape(0) != eigs.size()))
        {
            //print out an info statement indicating that the eigensolver is resizing the eigenvalues array.
            CALL_AND_HANDLE(eigs.resize(mat.shape(0), mat.shape(1)), "Failed to reshape the eigenvalues array.");
        }
    }
    template <typename matrix_type, typename vals_type>
    static inline typename std::enable_if<is_symtridiag_matrix_type<vals_type>::value && !traits<vals_type>::is_resizable, void>::type tridiag(const matrix_type& mat, vals_type& eigs)
    {ASSERT(((eigs.shape(0) == eigs.shape(1)) && (mat.shape(0) == eigs.nelems())), "The input eigenvalues array does not have the correct shape.");}


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


}   //namespace internal
}   //namespace linalg


#endif //LINALG_DECOMPOSITIONS_TRIDIAGONALISATION_BASE_HPP//

