#ifndef TTNS_LIB_TTN_ORTHOGONALITY_ROOT_TO_LEAF_DECOMPOSITION_HPP
#define TTNS_LIB_TTN_ORTHOGONALITY_ROOT_TO_LEAF_DECOMPOSITION_HPP

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>
#include "decomposition_engine.hpp"

namespace ttns
{
namespace orthogonality
{

template <typename T, typename backend> 
class root_to_leaf_decomposition_engine
{
    using hdata = ttn_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;
    using real_type = typename tmp::get_real_type<T>::type;
    using dmat = linalg::diagonal_matrix<real_type, backend>;
    using size_type = typename backend::size_type;

public:
    //helper functions for determining the size of temporary buffers needed for the decomposition engine
    static inline std::array<size_type, 2> maximum_matrix_dimension_node(const hdata& a, bool use_capacity = false)
    {
        std::array<size_type, 2> ret{{use_capacity ? a.capacity() : a.size(), 1}};
        size_type max_dim = a.hrank(use_capacity);
        for(size_type mode=0; mode < a.nmodes(); ++mode)
        {
            if(a.dim(mode, use_capacity) > max_dim){max_dim = a.dim(mode, use_capacity);}
        }
        ret[0] /= max_dim;  ret[1] *= max_dim;
        return ret;
    }
    
    static inline std::array<size_type, 2> maximum_matrix_dimension_node(const std::vector<hdata>& a, bool use_capacity = false)
    {
        std::array<size_type, 2> retmax{{0, 0}};
        for(size_type i = 0; i < a.size(); ++i)
        {
            std::array<size_type, 2> ret = maximum_matrix_dimension_node(a[i], use_capacity);
            if(ret[0] > retmax[0]){retmax[0] = ret[0];}
            if(ret[1] > retmax[1]){retmax[1] = ret[1];}
        }
        return retmax;
    }

    template <typename engine>
    static inline size_type maximum_work_size_node(engine& eng, const hdata& A, mat& U, mat& R, bool use_capacity = false)
    {
        size_type max_work_size;
        
        //determine the maximum dimension of the hierarchical tucker decomposition object
        size_type max_dim = A.hrank(use_capacity);
        if(A.nmodes()>1)
        {
            for(size_type mode=0; mode < A.nmodes(); ++mode)
            {
                if(A.dim(mode) > max_dim){max_dim = A.dim(mode, use_capacity);}
            }
        }

        //check that the result arrays have the correct capacity so that we can make sure we get the correct result
        if(use_capacity)
        {
            ASSERT(A.capacity() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
        }
        else            
        {
            ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
        }
        ASSERT(max_dim*max_dim <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");
        
        CALL_AND_HANDLE(U.resize(A.dimen(use_capacity), A.hrank(use_capacity)), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(R.resize(A.hrank(use_capacity), A.hrank(use_capacity)), "Failed when resizing the R matrix so that it has the correct shape.");

        //first check the default ordering decomposition
        CALL_AND_HANDLE(max_work_size = eng.query_work_size(A.as_rank_2(use_capacity), U, R), "Failed to query work size of the decomposition engine.");

        //now we test all of the possible reordering sizes 
        if(A.nmodes() != 1)
        {
            for(size_type mode=0; mode<A.nmodes(); ++mode)
            {
                int d1 = A.hrank(use_capacity)*A.dimen(use_capacity)/A.dim(mode, use_capacity);     int d2 = A.dim(mode, use_capacity);
                auto Ar = A.as_rank_2(use_capacity).reinterpret_shape(d1, d2);
                auto Ur = U.reinterpret_shape(d1, d2);

                CALL_AND_HANDLE(R.resize(d2, d2), "Failed when resizing the R matrix so that it has the correct shape.");

                size_type work_size;
                CALL_AND_HANDLE(work_size = eng.query_work_size(Ar, Ur, R), "Failed to query work size of the decomposition engine.");

                if(work_size > max_work_size){max_work_size = work_size;}
            }
        }
        //and return the maximum worksize
        return max_work_size;
    }

    template <typename engine>
    static inline size_type maximum_work_size_node(engine& eng, const std::vector<hdata>& A, std::vector<mat>& U, std::vector<mat>& R, bool use_capacity = false)
    {
        size_type maxdim = 0;
        for(size_type i = 0; i < A.size(); ++i)
        {
            size_type mi = maximum_work_size_node(eng, A[i], U[i], R[i], use_capacity);
            if(mi > maxdim){maxdim=mi;}
        }
        return maxdim;
    }

    //function for resizing the node objects for the root to leaf decomposition
    static inline void resize_r_matrix(const hdata& a, mat& r, bool use_capacity = false)
    {
        size_type max_dim = a.hrank(use_capacity);
        if(a.nmodes()>1)
        {
            for(size_type mode = 0; mode<a.nmodes(); ++mode){max_dim = max_dim < a.dim(mode, use_capacity) ? a.dim(mode, use_capacity) : max_dim;}
        }
        r.resize(max_dim,max_dim);
    }

    template <typename engine>
    static inline size_type evaluate(engine& eng, const hdata& A, mat& U, mat& R, mat& tm, size_type mode, real_type tol = real_type(0), size_type nchi = 0, bool rel_truncate = false, truncation_mode trunc_mode = truncation_mode::singular_values_truncation, bool save_svd = false)
    {
        ASSERT(mode < A.nmodes(), "Failed to perform the root to leaf decomposition.  The node to be decomposed does not have the requested mode.");

        size_type d1 = (A.hrank()*A.dimen())/A.dim(mode);     size_type d2 = A.dim(mode);

        ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
        ASSERT(A.size() <= tm.capacity(), "The U matrix is not the same size as the input matrix."); 
        ASSERT(d2*d2 <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");

        CALL_AND_HANDLE(U.resize(A.size(0), A.size(1)), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(tm.resize(d1, d2), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(R.resize(d2, d2), "Failed when resizing the R matrix so that it has the correct shape.");

        //indices useful for reordering the A tensor
        size_type id1 = 1;                  size_type id2 = A.hrank()*A.dimen();
        for(size_type i=0; i<=mode; ++i){id1 *= A.dim(i);}
        id2 /= id1;

        size_type bond_dimension = 0;
        //First we create a reordering of the matrix such that we have the index of interest as the fastest index of the matrix.
        CALL_AND_HANDLE(
        {
            //first we reinterpret 
            auto atens = A.as_matrix().reinterpret_shape(id1, id2);
            auto utens = U.reinterpret_shape(id2, id1);

            //now we permute the dimensions of this reinterpreted shape rank 3 tensor so that the middle dimension becomes the last dimension.
            utens = trans(atens);
        }
        , "Failed to unpack the matricisation back into the full rank tensor.");

        //now we reinterpret the reordered tensors as the correctly sized matrix and perform the singular values decomposition
        CALL_AND_HANDLE(
        {
            auto umat = U.reinterpret_shape(d1, d2);
            bond_dimension = eng(umat, tm, R, tol, nchi, rel_truncate, trunc_mode, save_svd);
        }
        , "Failed when evaluating the decomposition of the resultant matricisation.");


        size_type id1p = (R.shape(0)*id1/R.shape(1));
        U.resize(R.shape(0)*A.shape(0)/R.shape(1), A.size(1));
        //now it is necessary to undo the permutation done before so that we can store the transformed U tensor
        CALL_AND_HANDLE(
        {
            auto utens = U.reinterpret_shape(id1p, id2);
            auto ttens = tm.reinterpret_shape(id2, id1p);

            utens = trans(ttens);
        }
        ,"Failed when repacking the matricisation.");
        
        //Finally we need to transpose the R matrix as we want its transpose for the remainder of the code.
        CALL_AND_HANDLE(eng.transposeV(R), "Failed to compute the inplace transpose of the R matrix.");
        
        return bond_dimension;
    }

    //evaluate the root to leaf decomposition of a node for a given logical index.  Given the node tensor A and a mode index mode, returns the two matrices U, R
    template <typename engine>
    static inline void evaluate(engine& eng, const hdata& A, mat& U, size_type mode)
    {
        ASSERT(mode < A.nmodes(), "Failed to perform the root to leaf decomposition.  The node to be decomposed does not have the requested mode.");

        size_type d1 = (A.hrank()*A.dimen())/A.dim(mode);     size_type d2 = A.dim(mode);

        ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 

        CALL_AND_HANDLE(U.resize(A.size(0), A.size(1)), "Failed when resizing the U matrix so that it has the correct shape.");

        //indices useful for reordering the A tensor
        size_type id1 = 1;                  size_type id2 = A.hrank()*A.dimen();
        for(size_type i=0; i<=mode; ++i){id1 *= A.dim(i);}
        id2 /= id1;

        //First we create a reordering of the matrix such that we have the index of interest as the fastest index of the matrix.
        //This is done by performing the operations:
        //  
        //
        CALL_AND_HANDLE(
        {
            //first we reinterpret 
            auto atens = A.as_matrix().reinterpret_shape(id1, id2);
            auto utens = U.reinterpret_shape(id2, id1);

            //now we permute the dimensions of this reinterpreted shape rank 3 tensor so that the middle dimension becomes the last dimension.
            utens = trans(atens);
        }
        , "Failed to unpack the matricisation back into the full rank tensor.");

        //now we reinterpret the reordered tensors as the correctly sized matrix and perform the singular values decomposition
        CALL_AND_HANDLE(
        {
            auto umat = U.reinterpret_shape(d1, d2);
            eng(umat);
        }
        , "Failed when evaluating the decomposition of the resultant matricisation.");
    }

    template <typename engine>
    static inline size_type evaluate(engine& eng, const hdata& A, mat& U, mat& R, dmat& S, mat& tm, size_type mode, real_type tol = real_type(0), size_type nchi = 0, bool rel_truncate = false, truncation_mode trunc_mode = truncation_mode::singular_values_truncation, bool save_svd = false)
    {
        ASSERT(mode < A.nmodes(), "Failed to perform the root to leaf decomposition.  The node to be decomposed does not have the requested mode.");

        size_type d1 = (A.hrank()*A.dimen())/A.dim(mode);     size_type d2 = A.dim(mode);

        ASSERT(A.size() <= U.capacity(), "The U matrix is not the same size as the input matrix."); 
        ASSERT(A.size() <= tm.capacity(), "The U matrix is not the same size as the input matrix."); 
        ASSERT(d2*d2 <= R.capacity(), "The R matrix is not large enough to store the result of the decomposition.");

        CALL_AND_HANDLE(U.resize(A.size(0), A.size(1)), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(tm.resize(d1, d2), "Failed when resizing the U matrix so that it has the correct shape.");
        CALL_AND_HANDLE(R.resize(d2, d2), "Failed when resizing the R matrix so that it has the correct shape.");

        //indices useful for reordering the A tensor
        size_type id1 = 1;                  size_type id2 = A.hrank()*A.dimen();
        for(size_type i=0; i<=mode; ++i){id1 *= A.dim(i);}
        id2 /= id1;

        size_type bond_dimension = 0;
        //First we create a reordering of the matrix such that we have the index of interest as the fastest index of the matrix.
        CALL_AND_HANDLE(
        {
            //first we reinterpret 
            auto atens = A.as_matrix().reinterpret_shape(id1, id2);
            auto utens = U.reinterpret_shape(id2, id1);

            //now we permute the dimensions of this reinterpreted shape rank 3 tensor so that the middle dimension becomes the last dimension.
            utens = trans(atens);
        }
        , "Failed to unpack the matricisation back into the full rank tensor.");

        //now we reinterpret the reordered tensors as the correctly sized matrix and perform the singular values decomposition
        CALL_AND_HANDLE(
        {
            auto umat = U.reinterpret_shape(d1, d2);
            bond_dimension = eng(umat, tm, R, S, tol, nchi, rel_truncate, trunc_mode, save_svd);
        }
        , "Failed when evaluating the decomposition of the resultant matricisation.");


        size_type id1p = (R.shape(0)*id1/R.shape(1));
        U.resize(R.shape(0)*A.shape(0)/R.shape(1), A.size(1));
        //now it is necessary to undo the permutation done before so that we can store the transformed U tensor
        CALL_AND_HANDLE(
        {
            auto utens = U.reinterpret_shape(id1p, id2);
            auto ttens = tm.reinterpret_shape(id2, id1p);

            utens = trans(ttens);
        }
        ,"Failed when repacking the matricisation.");
        
        //Finally we need to transpose the R matrix as we want its transpose for the remainder of the code.
        CALL_AND_HANDLE(eng.transposeV(R), "Failed to compute the inplace transpose of the R matrix.");
        
        return bond_dimension;
    }

    static inline void apply_to_node(hdata& a, const mat& u)
    {
        ASSERT(a.as_matrix().shape() == u.shape(), "The U matrix and hierarchical tucker tensor node matrix are not the same size.");
        CALL_AND_HANDLE(a.as_matrix() = u, "Failed to set the value of the hierarchial tucker tensor node matrix.");
    }

    //applies the result of the root to leaf decomposition of a nodes parent to the node.  
    static inline void apply_bond_matrix(hdata& a, const mat& pr, mat& tm)
    {
        mat& A = a.as_matrix();   

        ASSERT(A.size() <= tm.capacity(), "The temporary working matrix is not large enough to store the result.");
        ASSERT(pr.shape(0) == pr.shape(1) && pr.shape(0) == A.shape(1), "The parent's R matrix is not the correct size.");
        CALL_AND_HANDLE(tm.resize(A.shape()), "Failed to resize the temporary working matrix so that it has the correct shape.");

        CALL_AND_HANDLE(tm = A*pr, "Failed to evaluate contraction with parents R matrix.");
        CALL_AND_HANDLE(A = tm, "Failed to set the value of the hierarchical tucker tensor node matrix.");
    }

    static inline void apply_with_truncation(hdata& a, hdata& c, size_type bond_index, const mat& pr, mat& tm)
    {
        CALL_AND_HANDLE(a.as_matrix() = tm, "Failed to set the value of the hierarchial tucker tensor node matrix.");

        a.set_dim(bond_index, pr.shape(1));
        mat& A = c.as_matrix();   

        ASSERT(A.size() <= tm.capacity(), "The temporary working matrix is not large enough to store the result.");
        ASSERT( pr.shape(0) == A.shape(1), "The parent's R matrix is not the correct size.");
        CALL_AND_HANDLE(tm.resize(A.shape(0), pr.shape(1)), "Failed to resize the temporary working matrix so that it has the correct shape.");

        CALL_AND_HANDLE(tm = A*pr, "Failed to evaluate contraction with parents R matrix.");
        CALL_AND_HANDLE(A = tm, "Failed to set the value of the hierarchical tucker tensor node matrix.");
    }
};  //class root_to_leaf_decomposition
}   //namespace orthogonality
}   //namespace ttns

#endif //TTNS_LIB_TTN_ORTHOGONALITY_ROOT_TO_LEAF_DECOMPOSITION_HPP

