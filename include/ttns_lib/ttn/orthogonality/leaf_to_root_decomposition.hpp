#ifndef TTNS_LIB_TTN_ORTHOGONALITY_LEAF_TO_ROOT_DECOMPOSITION_ENGINE_HPP
#define TTNS_LIB_TTN_ORTHOGONALITY_LEAF_TO_ROOT_DECOMPOSITION_ENGINE_HPP

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>
#include "decomposition_engine.hpp"

namespace ttns
{
namespace orthogonality
{

template <typename T, typename backend> 
class leaf_to_root_decomposition_engine
{
    using hdata = ttn_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using real_type = typename tmp::get_real_type<T>::type;
    using dmat = linalg::diagonal_matrix<real_type, backend>;
    using size_type = typename backend::size_type;
public:
    //the base functions for acting on the underlying hdata objects
    static inline std::array<size_type, 2> maximum_matrix_dimension_node(const hdata& a, bool use_capacity = false)
    {
        std::array<size_type, 2> ret{{a.dimen(use_capacity), a.hrank(use_capacity)}};
        return ret;
    }

    template <typename engine>
    static inline size_type maximum_work_size_node(engine& eng, const hdata& A, mat U, mat& R, bool use_capacity = false)
    {
        try
        {
            if(use_capacity){ASSERT(A.capacity() <= U.capacity(), "The U matrix is not the same size as the input matrix."); }
            else{ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); }
            ASSERT(A.hrank(use_capacity)*A.hrank(use_capacity) <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
            
            CALL_AND_HANDLE(U.resize(A.dimen(use_capacity), A.hrank(use_capacity)), "Failed when resizing the U matrix so that it has the correct shape.");
            CALL_AND_HANDLE(R.resize(A.hrank(use_capacity), A.hrank(use_capacity)), "Failed when resizing the R matrix so that it has the correct shape.");
            CALL_AND_HANDLE(return eng.query_work_size(A.as_rank_2(use_capacity), U, R), "Failed to query work size of the decomposition engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate the work size required for computing the leaf-to-root decomposition.");
        }
    }

    //function for resizing the node objects for the leaf-to-root decomposition
    static inline void resize_r_matrix(const hdata& a, mat& r, bool use_capacity = false)
    {
        CALL_AND_RETHROW(r.resize(a.hrank(use_capacity), a.hrank(use_capacity)));
    }

    //evluate the leaf-to-root decomposition at a node.  Given the node tensor A returns the two matrices U, R such that
    //A = U R
    template <typename engine>
    static inline size_type evaluate(engine& eng, const hdata& a, mat& U, mat& R, real_type tol = real_type(0), size_type nchi = 0, bool rel_truncate = false, truncation_mode trunc_mode = truncation_mode::singular_values_truncation, bool save_svd = false)
    {
        const auto& A = a.as_matrix();   
        ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
        ASSERT(a.hrank()*a.hrank() <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
        
        CALL_AND_HANDLE(U.resize(A.shape(0), A.shape(1)), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(R.resize(A.shape(1), A.shape(1)), "Failed when resizing the R matrix so that it has the correct shape.");
        size_type bond_dimension = 0;
        CALL_AND_HANDLE(bond_dimension = eng(A, U, R, tol, nchi, rel_truncate, trunc_mode, save_svd), "Failed when using the decomposition engine to evaluate the decomposition.");
        return bond_dimension;
    }    

    //A = U R
    template <typename engine>
    static inline size_type evaluate(engine& eng, const hdata& a, mat& U, mat& R, dmat& S, real_type tol = real_type(0), size_type nchi = 0, bool rel_truncate = false, truncation_mode trunc_mode = truncation_mode::singular_values_truncation, bool save_svd = false)
    {
        const auto& A = a.as_matrix();   
        ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
        ASSERT(a.hrank()*a.hrank() <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
        
        CALL_AND_HANDLE(U.resize(A.shape(0), A.shape(1)), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(R.resize(A.shape(1), A.shape(1)), "Failed when resizing the R matrix so that it has the correct shape.");
        size_type bond_dimension = 0;
        CALL_AND_HANDLE(bond_dimension = eng(A, U, R, S, tol, nchi, rel_truncate, trunc_mode, save_svd), "Failed when using the decomposition engine to evaluate the decomposition.");
        return bond_dimension;
    }    

    template <typename engine>
    static inline void evaluate(engine& eng, const hdata& A)
    {
        CALL_AND_HANDLE(eng(A), "Failed when using the decomposition engine to evaluate the decomposition.");
    }    

    //applies the result of a nodes leaf-to-root decomposition to it's parent node.  This function computes the tensor X^{n}
    //obtained as A^{n-1}_I;j A^{n}_kji,l = U^{n-1}_I;j (R_{jj'} A^{n}_kj'i, l) = U^{n-1}_I;j X^{n}_kji,l
    static inline void apply_bond_matrix(hdata& a, hdata& pa, size_type mode, const mat& R, mat& pt)
    {
        pt.resize(pa.shape(0), pa.shape(1));
        ASSERT(pa.as_matrix().size() <= pt.capacity(), "The temporary working matrix is not large enough to store the result.");
        CALL_AND_HANDLE(pt.resize(pa.as_matrix().shape()), "Failed to resize the temporary working matrix so that it has the correct shape.");

        ASSERT(mode < pa.nmodes(), "The hierarchical tucker tensor is ill-formed.");

        auto pa_3 = pa.as_rank_3(mode);
        auto pt_3 = pt.reinterpret_shape(pa_3.shape());
        CALL_AND_HANDLE(pt_3 = contract(R, 1, pa_3, 1), "Failed to compute the requested contraction between the R matrix and the parent A tensor.");

        CALL_AND_HANDLE(pa.as_matrix() = pt, "Failed to copy temporary working matrix into hierarchical tucker tensor parent node matrix.");
        pt.resize(a.shape(0), a.shape(1));
    }

    //this function applies the transformation to both the present node and it's parent.
    static inline void apply_with_truncation(hdata& a, hdata& pa, size_type mode, const mat& R, mat& u)
    {
        CALL_AND_HANDLE(a.as_matrix() = u, "Failed to set the value of the hierarchial tucker tensor node matrix.");

        u.resize(pa.shape(0), pa.shape(1));

        ASSERT(pa.as_matrix().size() <= u.capacity(), "The uorary working matrix is not large enough to store the result.");
        CALL_AND_HANDLE(u.resize(pa.as_matrix().shape()), "Failed to resize the uorary working matrix so that it has the correct shape.");

        ASSERT(mode < pa.nmodes(), "The hierarchical tucker tensor is ill-formed.");

        auto pa_3 = pa.as_rank_3(mode);
        u.resize( (R.shape(0)*pa.shape(0))/R.shape(1), pa.shape(1));


        auto pt_3 = u.reinterpret_shape(pa_3.shape(0), R.shape(0), pa_3.shape(2));
        CALL_AND_HANDLE(pt_3 = contract(R, 1, pa_3, 1), "Failed to compute the requested contraction between the R matrix and the parent A tensor.");

        CALL_AND_HANDLE(pa.as_matrix() = u, "Failed to copy uorary working matrix into hierarchical tucker tensor parent node matrix.");
        pa.set_dim(mode, R.shape(0));
        u.resize(a.shape(0), a.shape(1));
    }

};  //class leaf_to_root_decomposition
}   //namespace orthogonality
}   //namespace ttns

#endif //TTNS_LIB_TTN_ORTHOGONALITY_LEAF_TO_ROOT_DECOMPOSITION_ENGINE_HPP//

