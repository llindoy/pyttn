#ifndef TTNS_LIB_SWEEPING_ALGORITHM_TWO_SITE_ENERGY_VARIATIONS_HELPER_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_TWO_SITE_ENERGY_VARIATIONS_HELPER_HPP

#include "../../common/kronecker_product_operator_helper.hpp"
#include "../environment/sum_of_product_operator_env.hpp"


namespace ttns
{

template <typename T, typename backend = linalg::blas_backend> 
class two_site_variations;

template <typename U>
class two_site_variations<linalg::complex<U>, linalg::blas_backend>
{
public:
    using value_type = linalg::complex<U>;
    using real_type = U;
    using T = linalg::complex<U>;
    using backend = linalg::blas_backend;
    using size_type = typename backend::size_type;

    using environment_type = sop_environment<T, backend>;
    using spo_core = typename environment_type::spo_core;
    using mfo_core = typename environment_type::mfo_core;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_value_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;
    using sop_node_type = typename environment_type::sop_node_type;

    using hnode = ttn_node<value_type, backend>;
    using hdata = ttn_node_data<value_type, backend>;
    using vec = linalg::vector<value_type, backend>;
    using mat = linalg::matrix<value_type, backend>;
    using triad = std::vector<mat>;
    using rank_4 = std::vector<linalg::tensor<value_type, 3, backend>>;

public:
    two_site_variations() {}

    two_site_variations(const two_site_variations& o) = default;
    two_site_variations(two_site_variations&& o) = default;
    two_site_variations& operator=(const two_site_variations& o) = default;
    two_site_variations& operator=(two_site_variations&& o) = default;


public:
    static inline size_type get_nterms(const sttn_node_data<T>& h)
    {
        size_type two_site_energy_terms = 0;
        for(size_type ind=0; ind < h.nterms(); ++ind)
        {
            if(!h[ind].is_identity_mf() && !h[ind].is_identity_spf())
            {
                ++two_site_energy_terms;
            }
        }       
        return two_site_energy_terms;
    }

    static inline void set_indices(const sttn_node_data<T>& h, linalg::vector<size_type>& hinds, linalg::vector<T>& coeffs)
    {
        size_type two_site_energy_terms = 0;
        for(size_type ind=0; ind < h.nterms(); ++ind)
        {
            if(!h[ind].is_identity_mf() && !h[ind].is_identity_spf())
            {
                hinds[two_site_energy_terms] = ind;
                coeffs[two_site_energy_terms] = h[ind].coeff();
                ++two_site_energy_terms;
            }
        }       
    }

    //computes the action of the Hamiltonian acting on the SPFs associated with the lower of the two nodes used in the two-site expansion and stores
    //each of the terms in the array res.  Here we also include the r-term contracted into this term as generally the lower terms will have smaller
    //bond dimension
    static inline void construct_two_site_energy_terms_lower(hnode& A, const sop_node_type& hinf, const env_node_type& h, const env_type& hprim, triad& res, triad& HA, triad& temp, const linalg::vector<size_type>& hinds, const mat& rmat, bool apply_projector = true)
    {
        try
        {
            const auto& a = A().as_matrix();  
            //compute the action of the Hamiltonian on the lower of the two nodes and store the result in res
#ifdef USE_OPENMP
#ifdef PARALLELISE_HAMILTONIAN_SUM
        #pragma omp parallel for default(shared) schedule(dynamic, 1) num_threads(HA.size())
#endif
#endif
            for(size_type r = 0; r < hinds.size(); ++r)
            {
                size_type ind = hinds[r];
                ASSERT(!hinf()[ind].is_identity_mf() && !hinf()[ind].is_identity_spf(), "Invalid index.");

                size_type ti = omp_get_thread_num();
                CALL_AND_HANDLE(HA[ti].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                CALL_AND_HANDLE(temp[ti].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                CALL_AND_HANDLE(res[r].fill_zeros(), "Failed to fill array with zeros.");

                using spo_core = single_particle_operator_engine<T, backend>;
                if(A.is_leaf())
                {
                    for(size_type i = 0; i < hinf()[ind].nspf_terms(); ++i)
                    {
                        auto& indices = hinf()[ind].spf_indexing()[i][0];
                        CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(a, HA[ti]), "Failed to apply leaf operator.");
                        CALL_AND_HANDLE(res[r] += hinf()[ind].spf_coeff(i)*HA[ti], "Failed to apply matrix product to obtain result.");
                    }
                }
                else
                {
                    for(size_type i = 0; i < hinf()[ind].nspf_terms(); ++i)
                    {
                        CALL_AND_HANDLE(spo_core::kron_prod(h, hinf(), ind, i, A(), temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                        CALL_AND_HANDLE(res[r] += hinf()[ind].spf_coeff(i)*HA[ti], "Failed to apply matrix product to obtain result.");
                    }
                }

                //now we multiply res[r] by rmat to get the correct factor in place
                CALL_AND_HANDLE(HA[ti] = res[r]*rmat, "Failed to apply r-tensor."); 
                CALL_AND_HANDLE(res[r] = HA[ti], "Failed to copy HA ti back to res."); 
                
                if(apply_projector)
                {
                    //now apply the orthogonal complement projector to this result and reaccumulate it in res
                    CALL_AND_HANDLE(temp[ti].resize(A().size(1), A().size(1)), "Failed to resize temporary array.");
                    CALL_AND_HANDLE(temp[ti] = adjoint(a)*res[r], "Failed to compute matrix element.");
                    CALL_AND_HANDLE(res[r] -= a*temp[ti], "Failed to subtract off the projected contribution to the Hamiltonian.");
                    CALL_AND_HANDLE(temp[ti].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute the one site objects used in the construction of the two site Hamiltonian.");
        }
    }

    static inline void construct_two_site_energy_terms_upper(hnode& A, const sop_node_type& hinf, const env_node_type& h, rank_4& res, triad& HA, triad& temp, triad& temp2, const linalg::vector<size_type>& hinds, bool apply_projector = true)
    {
        try
        {
            size_type mode = h.child_id();
            const auto& h_p = h.parent();
            const auto& hinf_p = hinf.parent();
            //compute the action of the Hamiltonian on the upper of the two nodes and store the result in res
#ifdef USE_OPENMP
#ifdef PARALLELISE_HAMILTONIAN_SUM
        #pragma omp parallel for default(shared) schedule(dynamic, 1) num_threads(HA.size())
#endif
#endif
            for(size_type r = 0; r < hinds.size(); ++r)
            {
                size_type ind = hinds[r];
                ASSERT(!hinf()[ind].is_identity_mf() && !hinf()[ind].is_identity_spf(), "Invalid index.");

                size_type ti = omp_get_thread_num();
                CALL_AND_HANDLE(HA[ti].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                CALL_AND_HANDLE(temp[ti].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                CALL_AND_HANDLE(res[r].fill_zeros(), "Failed to fill array with zeros.");

                auto rmat = res[r].reinterpret_shape(A().shape(0), A().shape(1));
    
                for(size_type it = 0; it < hinf()[ind].nmf_terms(); ++it)
                {
                    size_type pi = hinf()[ind].mf_indexing()[it].parent_index();

                    CALL_AND_HANDLE(mfo_core::kron_prod(h, hinf(), ind, it, A(), HA[ti], temp[ti]), "Failed to evaluate action of kronecker product operator.");
                    if(!hinf_p()[pi].is_identity_mf())
                    {
                        CALL_AND_HANDLE(rmat += hinf()[ind].mf_coeff(it)*temp[ti]*trans(h_p().mf(pi)), "Failed to apply action of parent mean field operator.");
                    }
                    else
                    {
                        rmat += hinf()[ind].mf_coeff(it)*temp[ti];
                    }
                }

                if(apply_projector)
                {
                    //now apply the orthogonal complement projector to this result and reaccumulate it in res
                    CALL_AND_HANDLE(temp[ti] = conj(A().as_matrix()), "Failed to compute conjugate of the A matrix.");

                    //check that this is the correct projection
                    try
                    {
                        auto _A = A().as_rank_3(mode);
                        auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                        CALL_AND_HANDLE(HA[ti].resize(_A.shape(1), _A.shape(1)), "Failed to resize temporary array.");
                        CALL_AND_HANDLE(HA[ti] = (contract(res[r], 0, 2, _temp, 0, 2).bind_workspace(temp2[ti])), "Failed when evaluating the final contraction.");
                        CALL_AND_HANDLE(res[r] -= contract(HA[ti], 1, _A, 1), "Failed to contract the final res array with matrix.");
                        CALL_AND_HANDLE(HA[ti].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                    }                                         
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                    }   
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute the one site objects used in the construction of the two site Hamiltonian.");
        }
    }

    static inline void construct_two_site_energy(const linalg::vector<T>& hcoeff, const triad& h2s1, const rank_4& h2s2, mat& temp,  mat& res)
    {
        try
        {
            ASSERT(h2s1.size() == h2s2.size(), "Incorrect site terms.");
            ASSERT(h2s1.size() == hcoeff.size(), "Incorrect site terms.");
            if(h2s1.size() == 0)
            {
                CALL_AND_HANDLE(res.fill_zeros(), "Failed to fill the two site energy object with zeros.");
                return;
            }

            CALL_AND_HANDLE(temp.fill_zeros(), "Failed to fill the two site energy object with zeros.");
            auto ttens =  temp.reinterpret_shape(h2s2[0].size(0), h2s1[0].size(0), h2s2[0].size(2));
            auto rtens =  res.reinterpret_shape(h2s1[0].size(0), h2s2[0].size(0), h2s2[0].size(2));
            for(size_type r = 0; r < h2s1.size(); ++r)
            {
                auto contraction = contract(hcoeff[r]*h2s1[r], 1, h2s2[r], 1);
                CALL_AND_HANDLE(ttens += contraction, "Failed to contract element into res.");
            }
            rtens = linalg::transpose(ttens, {1, 0, 2});
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute two site energy.");
        }
    }
    

    //function for evaluating the two matrices required to evaluate the singular vectors (either left or right) of the projected two site Hamiltonian onto 
    //TODO: set up openmp parallelisation of this function call.
    template <typename vtype, typename rtype>
    inline void operator()(const vtype& v, const linalg::vector<T>& hcoeff, triad& op1, rank_4& op2, size_type nterms, mat& t1, mat& t2, mat& temp2, bool MconjM, rtype& res) const
    {       
        try
        {
            if(op2.size() == 0){res = 0.0*v;}
            else
            {
                res = 0.0*v;
                CALL_AND_HANDLE(t2.resize(op2[0].shape(1), 1), "Failed to resize t1 vector to the required size");
                auto t2vec = t2.reinterpret_shape(t2.shape(0));
                if(MconjM)
                {
                    CALL_AND_HANDLE(t1.resize(op1[0].shape(0), 1), "Failed to resize t2 vector to the required size");
                    t1.fill_zeros();
                    auto t1vec = t1.reinterpret_shape(t1.shape(0));

                    auto vtens = v.reinterpret_shape(op2[0].shape(0), 1, op2[0].shape(2));
                    auto rtens = res.reinterpret_shape(op2[0].shape(0), 1, op2[0].shape(2));
                    // t1 = M v

                    for(size_type r=0; r < nterms; ++r)
                    {
                        //first apply op2 to the v matrix
                        CALL_AND_HANDLE(t2 = contract(op2[r], 0, 2, vtens, 0, 2), "Failed to contract op2 with the tensor representation of the input array.");

                        //now apply op1 to the t2 vector. This is simply a matrix vector product and we add the result to t1
                        CALL_AND_HANDLE(t1vec += hcoeff[r]*op1[r]*t2vec, "Failed to compute the t1 vector.");
                    }

                    CALL_AND_HANDLE(temp2.resize(op1[0].shape(0), op1[0].shape(1)), "Failed to reshape temp array.");
                    // res = Mconj t1
                    for(size_type r=0; r < nterms; ++r)
                    {
                        CALL_AND_HANDLE(t2vec = linalg::conj(hcoeff[r])*(linalg::trans(op1[r])*linalg::conj(t1vec)).bind_conjugate_workspace(temp2), "Failed to compute the t2prime vector.");
                        CALL_AND_HANDLE(rtens += contract(op2[r], 1, t2, 0), "Failed to contract op2 with the matrix representation of the t2 vector.");
                    }
                    res = linalg::conj(res);
                }
                //M Mconj
                else
                {
                    CALL_AND_HANDLE(t1.resize(op2[0].shape(0)*op2[0].shape(2), 1), "Failed to resize t2 vector to the required size");
                    CALL_AND_HANDLE(temp2.resize(op2[0].shape(0)*op2[0].shape(1), op2[0].shape(2)), "Failed to reshape temp array.");
                    auto t1tens = t1.reinterpret_shape(op2[0].shape(0), 1, op2[0].shape(2));
                      
                    t1.fill_zeros();

                    //t1 = v M 
                    for(size_type r=0; r < nterms; ++r)
                    {
                        CALL_AND_HANDLE(t2vec = hcoeff[r]*linalg::trans(op1[r])*v, "Failed to apply op1 to the input vector.");
                        CALL_AND_HANDLE(t1tens += contract(op2[r], 1, t2, 0), "Failed to contract op2 with the matrix representation of the t2 vector.");
                    }

                    //r = t1 Mconj
                    for(size_type r=0; r < nterms; ++r)
                    {
                        //first apply op2 to the v matrix
                        CALL_AND_HANDLE(t2 = contract(linalg::conj(op2[r]), 0, 2, t1tens, 0, 2, temp2), "Failed to contract op2 with the tensor representation of the input array.");

                        //now apply op1 to the t2 vector. This is simply a matrix vector product and we add the result to t1
                        CALL_AND_HANDLE(res += linalg::conj(hcoeff[r])*(linalg::conj(op1[r])*t2vec), "Failed to contract op1 with temporary array.");
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply two-site energy variance object.");
        }
    }

public:
    //TODO: Need to move this implementation to the linear algebra functions and implement cuda and numpy specific version
    static inline void expand_matrix(mat& r, mat& temp, size_type iadd)
    {
        CALL_AND_HANDLE(temp.resize(r.shape(0), r.shape(1)), "Failed to resize temporary array.");
        CALL_AND_HANDLE(temp = r, "Failed to copy array into temporary buffer.");
        CALL_AND_HANDLE(r.resize(r.shape(0)+iadd, r.shape(1)+iadd), "Failed to resize matrix.");
    
        r.fill_zeros();
        for(size_type i = 0; i < temp.shape(0); ++i)
        {
            for(size_type j = 0; j < temp.shape(1); ++j)
            {
                r(i, j) = temp(i, j);
            }
        }
    }
};

}   //namespace ttns

#endif  //TTNS_TWO_SITE_ENERGY_VARIATIONS_HELPER_HPP//

