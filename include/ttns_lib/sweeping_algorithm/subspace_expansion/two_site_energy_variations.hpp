#ifndef TTNS_LIB_SWEEPING_ALGORITHM_TWO_SITE_ENERGY_VARIATIONS_HELPER_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_TWO_SITE_ENERGY_VARIATIONS_HELPER_HPP

#include "../../common/kronecker_product_operator_helper.hpp"
#include "../environment/sum_of_product_operator_env.hpp"

namespace ttns
{

template <typename T, typename backend = linalg::blas_backend> 
class two_site_variations;

template <typename U>
class two_site_variations<complex<U>, linalg::blas_backend>
{
public:
    using value_type = complex<U>;
    using real_type = U;
    using T = complex<U>;
    using backend = linalg::blas_backend;
    using size_type = typename backend::size_type;

    using environment_type = sop_environment<T, backend>;
    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_value_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;

    using hnode = ttn_node<value_type, backend>;
    using hdata = ttn_node_data<value_type, backend>;
    using vec = linalg::vector<value_type, backend>;
    using mat = linalg::matrix<value_type, backend>;
    using triad = std::vector<mat>;
    using rank_4 = std::vector<linalg::tensor<value_type, 3, backend>>;
    using matnode = typename tree<mat>::node_type;

public:
    two_site_variations() : m_r_dist(0, 1) {}
    two_site_variations(size_type seed) : m_r_dist(0, 1), m_rng(seed) {}
    two_site_variations(const std::mt19937& rng) : m_r_dist(0, 1), m_rng(rng) {}
    two_site_variations(std::mt19937&& rng) : m_r_dist(0, 1), m_rng(std::move(rng)) {}

    two_site_variations(const two_site_variations& o) = default;
    two_site_variations(two_site_variations&& o) = default;
    two_site_variations& operator=(const two_site_variations& o) = default;
    two_site_variations& operator=(two_site_variations&& o) = default;


public:
    void set_rng(size_t seed){m_rng = std::mt19937(seed);}
    void set_rng(const std::mt19937& rng){m_rng = rng;}

    static inline size_type get_nterms(const env_value_type& h)
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

    static inline void set_indices(const env_value_type& h, linalg::vector<size_type>& hinds)
    {
        size_type two_site_energy_terms = 0;
        for(size_type ind=0; ind < h.nterms(); ++ind)
        {
            if(!h[ind].is_identity_mf() && !h[ind].is_identity_spf())
            {
                hinds[two_site_energy_terms] = ind;
                ++two_site_energy_terms;
            }
        }       
    }

    //TODO: Need to modify this code to account for the new Hamiltonian structure used

    //computes the action of the Hamiltonian acting on the SPFs associated with the lower of the two nodes used in the two-site expansion and stores
    //each of the terms in the array res.  Here we also include the r-term contracted into this term as generally the lower terms will have smaller
    //bond dimension
    static inline void construct_two_site_energy_terms_lower(hnode& A, const env_node_type& h, const env_type& hprim, triad& res, triad& HA, triad& temp, const linalg::vector<size_type>& hinds, const mat& rmat, bool apply_projector = true)
    {
        try
        {
            for(size_type i = 0; i < HA.size(); ++i)
            {
                CALL_AND_HANDLE(HA[i].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                CALL_AND_HANDLE(temp[i].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
            }
    
            const auto& a = A().as_matrix();  
            //compute the action of the Hamiltonian on the lower of the two nodes and store the result in res
            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
            for(size_type r = 0; r < hinds.size(); ++r)
            {
                size_type ind = hinds[r];
                ASSERT(!h()[ind].is_identity_mf() && !h()[ind].is_identity_spf(), "Invalid index.");

                size_type ti = omp_get_thread_num();
                CALL_AND_HANDLE(res[r].fill_zeros(), "Failed to fill array with zeros.");

                if(A.is_leaf())
                {
                    for(size_type i = 0; i < h()[ind].nspf_terms(); ++i)
                    {
                        auto& indices = h()[ind].spf_indexing()[i][0];
                        CALL_AND_HANDLE(hprim(indices[0], indices[1]).apply(a, HA[ti]), "Failed to apply leaf operator.");
                        CALL_AND_HANDLE(res[r] += h()[ind].spf_coeff(i)*HA[ti], "Failed to apply matrix product to obtain result.");
                    }
                }
                else
                {
                    using kpo = kronecker_product_operator<value_type, backend>;
                    for(size_type i = 0; i < h()[ind].nspf_terms(); ++i)
                    {
                        CALL_AND_HANDLE(kpo::apply(h, ind, i, A(), temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                        CALL_AND_HANDLE(res[r] += h()[ind].spf_coeff(i)*HA[ti], "Failed to apply matrix product to obtain result.");
                    }
                }

                //now we multiply res[r] by rmat to get the correct factor in place
                CALL_AND_HANDLE(HA[ti] = res[r]*rmat, "Failed to apply r-tensor."); 
                CALL_AND_HANDLE(res[r] = HA[ti], "Failed to copy HA ti bacck to res."); 
                
                if(apply_projector)
                {
                    //now apply the orthogonal complement projector to this result and reaccumulate it in res
                    CALL_AND_HANDLE(temp[ti].resize(A().size(1), A().size(1)), "Failed to resize temporary array.");
                    CALL_AND_HANDLE(temp[ti] = adjoint(a)*HA[ti], "Failed to compute matrix element.");
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

    static inline void construct_two_site_energy_terms_upper(hnode& A, const env_node_type& h, rank_4& res, triad& HA, triad& temp, triad& temp2, const linalg::vector<size_type>& hinds, bool apply_projector = true)
    {
        try
        {
            for(size_type i = 0; i < HA.size(); ++i)
            {
                CALL_AND_HANDLE(HA[i].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
                CALL_AND_HANDLE(temp[i].resize(A().size(0), A().size(1)), "Failed to resize temporary array.");
            }
    
            size_type mode = h.child_id();
            const auto& h_p = h.parent();
            //compute the action of the Hamiltonian on the upper of the two nodes and store the result in res
            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
            for(size_type r = 0; r < hinds.size(); ++r)
            {
                size_type ind = hinds[r];
                ASSERT(!h()[ind].is_identity_mf() && !h()[ind].is_identity_spf(), "Invalid index.");

                size_type ti = omp_get_thread_num();
                CALL_AND_HANDLE(res[r].fill_zeros(), "Failed to fill array with zeros.");

                auto rmat = res[r].reinterpret_shape(A().shape(0), A().shape(1));
    
                using mfo_core = mean_field_operator_engine<value_type, backend>;
                for(size_type it = 0; it < h()[ind].nmf_terms(); ++it)
                {
                    size_type pi = h()[ind].mf_indexing()[it].parent_index();

                    CALL_AND_HANDLE(mfo_core::kron_prod(h, ind, it, A(), HA[ti], temp[ti]), "Failed to evaluate action of kronecker product operator.");
                    if(!h_p()[pi].is_identity_mf())
                    {
                        CALL_AND_HANDLE(rmat += h()[ind].mf_coeff(it)*temp[ti]*trans(h_p()[pi].mf()), "Failed to apply action of parent mean field operator.");
                    }
                    else
                    {
                        rmat += h()[ind].mf_coeff(it)*temp[ti];
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

    static inline void construct_two_site_energy(const triad& h2s1, const rank_4& h2s2, mat& temp,  mat& res)
    {
        try
        {
            ASSERT(h2s1.size() == h2s2.size(), "Incorrect site terms.");
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
                auto contraction = contract(h2s1[r], 1, h2s2[r], 1);
                CALL_AND_HANDLE(ttens += contraction, "Failed to contract element into res.");
            }
            //now we need to permute the first two indices of this array
            for(size_type i = 0; i < rtens.shape(0); ++i)
            {
                for(size_type j = 0; j < rtens.shape(1); ++j)
                {
                    for(size_type k = 0; k < rtens.shape(2); ++k)
                    {
                        rtens(i, j, k) = ttens(j, i, k);
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute two site energy.");
        }
    }
    

    //function for evaluating the two matrices required to evaluate the singular vectors (either left or right) of the projected two site Hamiltonian onto 
    template <typename vtype, typename rtype>
    inline void operator()(const vtype& v, triad& op1, rank_4& op2, size_type nterms, mat& t1, mat& t2, mat& temp2, bool MconjM, rtype& res) const
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
                        CALL_AND_HANDLE(t1vec += op1[r]*t2vec, "Failed to compute the t1 vector.");
                    }

                    CALL_AND_HANDLE(temp2.resize(op1[0].shape(0), op1[0].shape(1)), "Failed to reshape temp array.");
                    // res = Mconj t1
                    for(size_type r=0; r < nterms; ++r)
                    {
                        CALL_AND_HANDLE(t2vec = (linalg::trans(op1[r])*linalg::conj(t1vec)).bind_conjugate_workspace(temp2), "Failed to compute the t2prime vector.");
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
                        CALL_AND_HANDLE(t2vec = linalg::trans(op1[r])*v, "Failed to apply op1 to the input vector.");
                        CALL_AND_HANDLE(t1tens += contract(op2[r], 1, t2, 0), "Failed to contract op2 with the matrix representation of the t2 vector.");
                    }

                    //r = t1 Mconj
                    for(size_type r=0; r < nterms; ++r)
                    {
                        //first apply op2 to the v matrix
                        CALL_AND_HANDLE(t2 = contract(linalg::conj(op2[r]), 0, 2, t1tens, 0, 2, temp2), "Failed to contract op2 with the tensor representation of the input array.");

                        //now apply op1 to the t2 vector. This is simply a matrix vector product and we add the result to t1
                        CALL_AND_HANDLE(res += linalg::conj(op1[r])*t2vec, "Failed to contract op1 with temporary array.");
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

    //generate the orthogonal trial vectors for the parent node.  Here the v
    template <typename vec2_type>
    inline void generate_orthogonal_trial_vector(const hdata& a, size_type mode, vec2_type&& x)
    {
        bool generate_vector = true;
        auto atens = a.as_rank_3(mode);

        ASSERT(x.capacity() >= atens.shape(0)*atens.shape(2), "Invalid xvec size.");
        CALL_AND_HANDLE(x.resize(atens.shape(0)*atens.shape(2)), "Failed to resize x object.");

        //if we already dealing with a full rank space we cannot generate an additional orthogonal vector
        if(atens.shape(0)*atens.shape(2) == atens.shape(1)){return;}

        while(generate_vector)
        {
            generate_orthonormal(x);
            bool has_orthogonal_component = true;
            for(size_type k=0; k < atens.size(1) && has_orthogonal_component; ++k)
            {
                //subtract the projection of x onto hdata away from x
                remove_projection(atens, k, x);

                //compute the norm of the x tensor
                real_type norm = std::sqrt(std::real(linalg::dot_product(linalg::conj(x), x)));

                if(norm > 1e-14){x/=norm;}
                else{has_orthogonal_component = false;}
            }
            if(has_orthogonal_component)
            {
                generate_vector = false;
            }
        }
    }

public:
    static inline void expand_tensor(hdata& a, mat& temp, size_type mode, size_type iadd, std::vector<size_type>& dims)
    {
        try
        {
            CALL_AND_HANDLE(temp.resize(a.shape(0), a.shape(1)), "Failed to resize temporary array.");
            CALL_AND_HANDLE(temp = a.as_matrix(), "Failed to copy array into temporary buffer.");

            auto atens = a.as_rank_3(mode);
            auto ttens = temp.reinterpret_shape(atens.shape(0), atens.shape(1), atens.shape(2));

            CALL_AND_HANDLE(ttens = atens, "Failed to store tensor into temporary.");
            CALL_AND_HANDLE(expand_tensor_internal(a, mode, iadd, dims), "Failed to expand tensor.");

            a.as_matrix().fill_zeros();
            auto atens_resized = a.as_rank_3(mode);
    
            fill_tensor(ttens, atens_resized);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to expand tensor.");
        }
    }

    //vec a matrix who's rows are the vectors to add
    static inline void expand_tensor(hdata& a, mat& temp, size_type mode, const mat& vec, size_type iadd, std::vector<size_type>& dims)
    {
        try
        {
            CALL_AND_HANDLE(temp.resize(a.shape(0), a.shape(1)), "Failed to resize temporary array.");
            CALL_AND_HANDLE(temp = a.as_matrix(), "Failed to copy array into temporary buffer.");

            ASSERT(vec.shape(0) >= iadd, "We have fewer vectors stored than are required.");

            auto atens = a.as_rank_3(mode);
            ASSERT(vec.shape(1) == atens.shape(0)*atens.shape(2), "Invalid tensor size.");
            auto ttens = temp.reinterpret_shape(atens.shape(0), atens.shape(1), atens.shape(2));

            CALL_AND_HANDLE(ttens = atens, "Failed to store tensor into temporary.");
            CALL_AND_HANDLE(expand_tensor_internal(a, mode, iadd, dims), "Failed to expand tensor.");

            a.as_matrix().fill_zeros();
            auto atens_resized = a.as_rank_3(mode);
            auto vtens = vec.reinterpret_capacity(iadd, atens.shape(0), atens.shape(2));
    
            fill_tensor(vtens, iadd, ttens, atens_resized);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to expand tensor.");
        }
    }

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
protected:


    template <typename vec2_type>
    void generate_orthonormal(vec2_type&& x)
    {
        for(size_t i = 0; i < x.size(); ++i)
        {
            x(i) = value_type(m_r_dist(m_rng), m_r_dist(m_rng));
        }
        real_type norm = std::sqrt(std::real(linalg::dot_product(linalg::conj(x), x)));
        x/=norm;
    }


    static inline void expand_tensor_internal(hdata& a, size_type mode, size_type iadd, std::vector<size_type>& dims)
    {
        if(mode == a.nmodes())
        {
            ASSERT(a.hrank()+iadd <= a.max_hrank(), "Cannot expand tensor object. Insufficient memory has been allocated.");
            a.resize(a.hrank()+iadd, a.dims());
        }
        else
        {
            dims = a.dims();
            ASSERT(dims[mode]+iadd <= a.max_dim(mode), "Cannot expand tensor object. Insufficient memory has been allocated.");
            dims[mode] += iadd;
            a.resize(a.hrank(), dims);
        }
    }


    template <typename Utype, typename vt>
    static value_type dot(const Utype& u, size_type k, vt& x)
    {
        try
        {
            ASSERT(k < u.shape(1), "k index out of bound.");
            auto xmat = x.reinterpret_shape(u.shape(0), u.shape(2));
            value_type dot = 0;
            for(size_t i = 0; i < u.shape(0); ++i)
            {
                for(size_t j=0; j < u.shape(2); ++j)
                {
                    dot += u(i, k, j)*xmat(i, j);
                }
            }
            return dot;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute non-contiguous dot product.");
        }
    }

    template <typename Utype, typename vt>
    static void remove_projection(const Utype& u, size_type k, vt& x)
    {
        try
        {
            value_type uix = dot(u, k, x);

            auto xmat = x.reinterpret_shape(u.shape(0), u.shape(2));
            for(size_t i = 0; i < u.shape(0); ++i)
            {
                for(size_t j=0; j < u.shape(2); ++j)
                {
                    xmat(i, j) -= uix*u(i, k, j);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute non-contiguous dot product.");
        }
    }

    template <typename A, typename B> 
    static void fill_tensor(const B& b, A& a)
    {
        ASSERT(a.shape(0) == b.shape(0) && a.shape(2) == b.shape(2) && a.shape(1) >= b.shape(1), "invalid tensor sizes.");
        for(size_type i = 0; i < b.shape(0); ++i)
        {
            for(size_type j = 0; j < b.shape(1); ++j)
            {
                for(size_type k=0; k < b.shape(2); ++k)
                {
                    a(i, j, k) = b(i, j, k);
                }
            }
        }
    }
    template <typename A, typename B, typename V> 
    static void fill_tensor(const V& v, size_type iadd, const B& b, A& a)
    {
        ASSERT(a.shape(0) == b.shape(0) && a.shape(2) == b.shape(2) && b.shape(1) + iadd <= a.shape(1), "invalid tensor sizes.");
        ASSERT(v.shape(1) == b.shape(0) && v.shape(2) == b.shape(2) && v.shape(0) >= iadd, "invalid tensor sizes.");
        
        for(size_type i = 0; i < b.shape(0); ++i)
        {
            for(size_type j = 0; j < b.shape(1); ++j)
            {
                for(size_type k=0; k < b.shape(2); ++k)
                {
                    a(i, j, k) = b(i, j, k);
                }
            }
            for(size_type j = 0; j < iadd; ++j)
            {
                size_type aj = j+b.shape(1);
                for(size_type k=0; k < b.shape(2); ++k)
                {
                    a(i, aj, k) = v(j, i, k);
                }
            }
        }
    }
protected:
    std::normal_distribution<real_type> m_r_dist;
    std::mt19937 m_rng;
};

}   //namespace ttns

#endif  //TTNS_TWO_SITE_ENERGY_VARIATIONS_HELPER_HPP//

