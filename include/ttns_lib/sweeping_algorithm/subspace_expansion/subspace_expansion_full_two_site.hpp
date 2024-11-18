#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_FULL_TWO_SITE_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_FULL_TWO_SITE_HPP

#include "two_site_energy_variations.hpp"
#include "../environment/sum_of_product_operator_env.hpp"
#include "../../ttn/orthogonality/decomposition_engine.hpp"

namespace ttns
{

template <typename T, typename backend>
class subspace_expansion_full_two_site
{
public:
    using twosite = two_site_variations<T, backend>;
    using vec_type = linalg::vector<T, backend>;
    using mat_type = linalg::matrix<T, backend>;
    using triad_type = std::vector<mat_type>;
    
    using environment_type = sop_environment<T, backend>;
    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_type = typename environment_type::environment_type;

    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

    using dmat_type = linalg::diagonal_matrix<real_type, backend>;

    using hnode = ttn_node<T, backend>;
    using hdata = ttn_node_data<T, backend>;

    using bond_matrix_type = typename ttn<T, backend>::bond_matrix_type;

public:
    subspace_expansion_full_two_site() : m_twosite() {}
    subspace_expansion_full_two_site(const ttn<T, backend>& A, const env_type& ham, size_type seed = 0)  : m_twosite(seed)
    {
        CALL_AND_HANDLE(initialise(A, ham), "Failed to construct subspace_expansion_full_two_site.");
    }   
    subspace_expansion_full_two_site(const subspace_expansion_full_two_site& o) = default;
    subspace_expansion_full_two_site(subspace_expansion_full_two_site&& o) = default;

    subspace_expansion_full_two_site& operator=(const subspace_expansion_full_two_site& o) = default;
    subspace_expansion_full_two_site& operator=(subspace_expansion_full_two_site&& o) = default;

    void initialise(const ttn<T, backend>& A, const env_type& sop)
    {
        try
        {
            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

            size_type maxcapacity = 0;  size_type maxnmodes = 0;
            size_type mtwosite_capacity = 0;    size_type mtwosite_size = 0;
            for(const auto& a : A)
            {
                size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}
                size_type nmodes = a().nmodes();    if(nmodes > maxnmodes){maxnmodes = nmodes;}

                if(!a.is_root())
                {
                    for(size_type mode = 0;  mode < nmodes;  ++mode)
                    {
                        auto aptens_c = a.parent()().as_rank_3(mode, true);
                        auto aptens_s = a.parent()().as_rank_3(mode);

                        size_type c2s = a().max_dimen()*aptens_c.shape(0)*aptens_c.shape(1);
                        size_type s2s = a().max_dimen()*aptens_s.shape(0)*aptens_s.shape(1);
                        if(c2s > mtwosite_capacity){mtwosite_capacity=c2s;}
                        if(s2s > mtwosite_size){mtwosite_size=s2s;}
                    }
                }
            }

            size_type max_two_site_energy_terms = 0;

            for(const auto& hinf : sop.contraction_info())
            {
                size_type two_site_energy_terms = twosite::get_nterms(hinf());
                if(two_site_energy_terms > max_two_site_energy_terms){max_two_site_energy_terms = two_site_energy_terms;}
            }

            m_twosite_energy.reallocate(mtwosite_capacity);
            m_twosite_temp.reallocate(mtwosite_capacity);
            m_twosite_energy.resize(1, mtwosite_size);
            m_twosite_temp.resize(1, mtwosite_size);

            CALL_AND_HANDLE(m_2s_1.resize(max_two_site_energy_terms), "Failed to resize two site spf buffer.");
            CALL_AND_HANDLE(m_2s_2.resize(max_two_site_energy_terms), "Failed to resize two site mf buffer.");

            for(size_type i = 0; i < max_two_site_energy_terms; ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].reallocate(maxcapacity), "Failed to reallocate two site spf buffer.");
                CALL_AND_HANDLE(m_2s_2[i].reallocate(maxcapacity), "Failed to reallocate two site mf buffer.");
            }

            m_maxcapacity = maxcapacity;
            m_inds.resize(maxnmodes);
            m_coeffs.resize(maxnmodes);
            m_dim.resize(maxnmodes);

            m_rvec.reallocate(maxcapacity);
            m_trvec.reallocate(maxcapacity);
            m_trvec2.reallocate(maxcapacity);

            m_onesite_expansions = 0;
            m_twosite_expansions = 0;

        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void clear()
    {
        try
        {
            for(size_type i = 0; i < m_2s_1.size(); ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].clear(), "Failed to clear the rvec object.");
                CALL_AND_HANDLE(m_2s_2[i].clear(), "Failed to clear the rvec object.");
            }
            CALL_AND_HANDLE(m_inds.clear(), "Failed to clear temporary inds array.");
            CALL_AND_HANDLE(m_coeffs.clear(), "Failed to clear temporary coefficients array.");
            CALL_AND_HANDLE(m_rvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec2.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_1.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_2.clear(), "Failed to clear the rvec object.");
            m_onesite_expansions = 0;
            m_twosite_expansions = 0;
            m_twosite_energy.clear();
            m_twosite_temp.clear();
            m_svd.clear();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    template <typename BufType>
    bool down(hnode& A1, hnode& A2, bond_matrix_type& r, const dmat_type& pops, env_node_type& h, const env_type& op, std::mt19937& rng, BufType& buf, real_type svd_scale)
    {
        try
        {
            bool unoccupied_spawning = m_unoccupied_threshold > 0;
            bool subspace_spawning = m_spawning_threshold > 0;

            if(A2.is_root() || A1.is_root() ){return false;}
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A2().dim(mode) >= A2().max_dim(mode) || A1().hrank() >= A1().max_hrank()){return false;}

            //check if the max dim variable has been set and if the current bond dimension is greater or equal to this value don't attempt subspace expansion
            if(m_max_dim != 0 && A1().hrank() >= m_max_dim){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A2().max_dim(mode) - A2().dim(mode);
            size_type max_add3 = m_max_dim - A1().hrank();
            max_add = max_add < max_add3 ? max_add : max_add3;
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            auto A2tens = A2().as_rank_3(mode);
            size_type max_dimension = A2tens.shape(0)*A2tens.shape(2);
            //size_type curr_dim = A2().dim(mode);

            //if the matricisation of the tensor we are currently trying to expand is square then we don't even attempt to expand
            if(A2tens.shape(1) >= max_dimension){return false;}

            real_type scale_factor = 1.0;
            size_type n_unocc = get_nunoccupied(pops, scale_factor);
            svd_scale /= scale_factor;
            if(m_only_apply_when_no_unoccupied && n_unocc >= m_minimum_unoccupied){return false;}

            //get the maximum number of elements that we can add
            size_type max_add2 = max_dimension - A2tens.shape(1);

            //we can add min(max_add, max_add2) elements
            max_add = max_add < max_add2 ? max_add : max_add2;

            //now determine the number of unoccupied functions, and using this determine the minimum number of terms we need to add
            size_type required_terms = 0;
            if(n_unocc < m_minimum_unoccupied)
            {
                required_terms = m_minimum_unoccupied - n_unocc;
            }
            //and limit the required number of terms to the maximum number of terms we can add
            if(required_terms > max_add){required_terms = max_add;}

            size_type nadd = 0;
            //if the other tensor is not square then we are in a regime where the two-site algorithm will provide a search direction so we should attempt it
            if(A1().shape(1) < A1().shape(0) && A1().shape(1) != 1 && subspace_spawning)
            {
                const auto& hinf = op.contraction_info()[h.id()];
                //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
                size_type nterms = twosite::get_nterms(hinf()); 
                ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
                CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
                CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");
                CALL_AND_HANDLE(m_coeffs.resize(nterms), "Failed to resize coefficients array.");

                twosite::set_indices(hinf(), m_inds, m_coeffs); 

                //compute the one site objects that are used to compute the two site projected energy
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, hinf, h, op, m_2s_1, buf.HA, buf.temp, m_inds, r, true), 
                                "Failed to construct the component of the two site energy acting on the lower site.");
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, hinf,  h, m_2s_2, buf.HA, buf.temp, buf.temp2, m_inds, true), 
                                "Failed to construct the component of the two site energy acting on the upper site.");

                CALL_AND_HANDLE(m_twosite_energy.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");
                CALL_AND_HANDLE(m_twosite_temp.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");

                CALL_AND_HANDLE(twosite::construct_two_site_energy(m_coeffs, m_2s_1, m_2s_2, m_twosite_temp, m_twosite_energy), "Failed to construct two site energy object.");
                CALL_AND_HANDLE(m_svd(m_twosite_energy, m_S, m_twosite_temp, m_V), "Failed to compute svd.");

                for(size_type i = 0; i < m_S.size(); ++i)
                {
                    real_type sv = 0;
                    //check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
                    if(m_trunc_mode == orthogonality::truncation_mode::singular_values_truncation)
                    {
                        if(std::real(m_S(i, i)) > 0){sv = std::sqrt(std::real(m_S(i, i)))*svd_scale;}
                        if(!std::isnan(sv))
                        {
                            if(sv > m_spawning_threshold){++nadd;}
                        }
                    }
                    else
                    {
                        if(std::real(m_S(i, i)) > 0){sv = std::real(m_S(i, i))*svd_scale*svd_scale;}
                        if(!std::isnan(sv))
                        {
                            if(sv > m_spawning_threshold){++nadd;}
                        }
                    }
                }

                if(nadd > max_add){nadd = max_add;}
#ifdef ALLOW_EVAL_BUT_DONT_APPLY
                if(nadd != 0 && !m_eval_but_dont_apply)
#else
                if(nadd != 0)
#endif
                {
                    //expand the A1 tensor zero padding
                    CALL_AND_HANDLE(A1().expand_bond(A1.nmodes(), nadd, buf.temp[0]), "Failed to expand A1 tensor."); 

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], nadd), "Failed to expand R matrix.");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(A2().expand_bond(mode, nadd, buf.temp[0], m_V), "Failed to expand A1 tensor."); 

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    subspace_expanded = true;
                }
                ++m_twosite_expansions;
            } 
#ifdef ALLOW_EVAL_BUT_DONT_APPLY
            if(nadd < required_terms && unoccupied_spawning && !m_eval_but_dont_apply)
#else
            if(nadd < required_terms && unoccupied_spawning)
#endif
            {
                size_type add = required_terms-nadd;

                //expand the A1 tensor zero padding
                CALL_AND_HANDLE(A1().expand_bond(A1.nmodes(), nadd, buf.temp[0]), "Failed to expand A1 tensor."); 

                //expand the r-matrix
                CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], nadd), "Failed to expand R matrix.");

                //expand the A2 tensor padding with the right singular vectors
                CALL_AND_HANDLE(A2().expand_bond(mode, nadd, buf.temp[0], rng), "Failed to expand A1 tensor."); 

                //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                //stored in these matrices as they will be updated before they are used for anything.
                CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");
            
                nadd += add;
                subspace_expanded = true;
                ++m_onesite_expansions;
            }
            //resize all of the working buffers. 
            CALL_AND_HANDLE(buf.resize(A2().shape(0), A2().shape(1)), "Failed to resize working arrays.");
            return subspace_expanded;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing down the tree.");
        }
    }

    template <typename BufType>
    bool up(hnode& A1, hnode& A2, bond_matrix_type& r, const dmat_type& pops, env_node_type& h, const env_type& op, std::mt19937& rng, BufType& buf, real_type svd_scale)
    {
        try
        {
            bool unoccupied_spawning = m_unoccupied_threshold > 0;
            bool subspace_spawning = m_spawning_threshold > 0;

            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A1().hrank() >= A1().max_hrank()){return false;}

            //check if the max dim variable has been set and if the current bond dimension is greater or equal to this value don't attempt subspace expansion
            if(m_max_dim != 0 && A1().hrank() >= m_max_dim){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A1().max_hrank() - A1().hrank();
            size_type max_add3 = m_max_dim - A1().hrank();
            max_add = max_add < max_add3 ? max_add : max_add3;
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            size_type max_dimension = A1().shape(0);
            //size_type curr_dim = A1().hrank();

            auto A2tens = A2().as_rank_3(mode);
            //if the matricisation of the tensor we are currently trying to expand is square then we don't even attempt to expand
            if(A1().shape(1) >= max_dimension){return false;}

            real_type scale_factor = 1.0;
            size_type n_unocc = get_nunoccupied(pops, scale_factor);
            if(m_only_apply_when_no_unoccupied && n_unocc >= m_minimum_unoccupied){return false;}
            svd_scale /= scale_factor;

            //get the maximum number of elements that we can add
            size_type max_add2 = max_dimension - A1().shape(1);

            //we can add min(max_add, max_add2) elements
            max_add = max_add < max_add2 ? max_add : max_add2;

            //now determine the number of unoccupied functions, and using this determine the minimum number of terms we need to add
            size_type required_terms = 0;
            if(n_unocc < m_minimum_unoccupied)
            {
                required_terms = m_minimum_unoccupied - n_unocc;
            }
            //and limit the required number of terms to the maximum number of terms we can add
            if(required_terms > max_add){required_terms = max_add;}
    
            size_type nadd = 0;
            //if the other tensor we are expanding is not a square tensor then the two-site algorithm will provide a sensible search direction 
            if(A2tens.shape(1) < A2tens.shape(0)*A2tens.shape(2) && A2tens.shape(1) != 1 && subspace_spawning)
            {
                const auto& hinf = op.contraction_info()[h.id()];
                //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
                size_type nterms = twosite::get_nterms(hinf()); 

                ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
                CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
                CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");
                CALL_AND_HANDLE(m_coeffs.resize(nterms), "Failed to resize coefficients array.");

                twosite::set_indices(hinf(), m_inds, m_coeffs); 

                //compute the one site objects that are used to compute the two site projected energy
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, hinf, h, m_2s_2, buf.HA, buf.temp, buf.temp2, m_inds, true), 
                                "Failed to construct the component of the two site energy acting on the upper site.");
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, hinf, h, op, m_2s_1, buf.HA, buf.temp, m_inds, r, true), 
                                "Failed to construct the component of the two site energy acting on the lower site.");

                CALL_AND_HANDLE(m_twosite_energy.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");
                CALL_AND_HANDLE(m_twosite_temp.resize(A1().shape(0), A2tens.shape(0)*A2tens.shape(2)), "Failed to resize two site energy object.");

                CALL_AND_HANDLE(twosite::construct_two_site_energy(m_coeffs, m_2s_1, m_2s_2, m_twosite_temp, m_twosite_energy), "Failed to construct two site energy object.");
                CALL_AND_HANDLE(m_svd(m_twosite_energy, m_S, m_twosite_temp, m_V), "Failed to compute svd.");
                m_U = linalg::trans(m_twosite_temp);

                for(size_type i = 0; i < m_S.size(); ++i)
                {
                    real_type sv = 0;
                    //check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
                    
                    if(m_trunc_mode == orthogonality::truncation_mode::singular_values_truncation)
                    {
                        if(std::real(m_S(i, i)) > 0){sv = std::sqrt(std::real(m_S(i, i)))*svd_scale;}
                        if(!std::isnan(sv))
                        {
                            if(sv > m_spawning_threshold){++nadd;}
                        }
                    }
                    else
                    {
                        if(std::real(m_S(i, i)) > 0){sv = std::real(m_S(i, i))*svd_scale*svd_scale;}
                        if(!std::isnan(sv))
                        {
                            if(sv > m_spawning_threshold){++nadd;}
                        }
                    }
                }


                if(nadd > max_add){nadd = max_add;}
                //here we need to compute the eigenstates of the 
#ifdef ALLOW_EVAL_BUT_DONT_APPLY
                if(nadd != 0 && !m_eval_but_dont_apply)
#else
                if(nadd != 0)
#endif
                {
                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(A2().expand_bond(mode, nadd, buf.temp[0]), "Failed to expand A2 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], nadd), "Failed to expand R matrix.");

                    //expand the A1 tensor around the index pointing to the root zero padding
                    CALL_AND_HANDLE(A1().expand_bond(A1.nmodes(), nadd, buf.temp[0], m_U), "Failed to expand A1 tensor.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    subspace_expanded = true;
                    ++m_twosite_expansions;
                }
            }
            //otherwise we just use random one-site tensor expansion
#ifdef ALLOW_EVAL_BUT_DONT_APPLY
            if(nadd < required_terms && unoccupied_spawning && !m_eval_but_dont_apply)
#else
            if(nadd < required_terms && unoccupied_spawning)
#endif
            {
                size_type add = required_terms-nadd;

                //expand the A2 tensor padding with the right singular vectors
                CALL_AND_HANDLE(A2().expand_bond(mode, add, buf.temp[0]), "Failed to expand A2 tensor.");

                //expand the r-matrix
                CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], add), "Failed to expand R matrix.");

                //expand the A1 tensor around the index pointing to the root zero padding
                CALL_AND_HANDLE(A1().expand_bond(A1.nmodes(), add, buf.temp[0], rng), "Failed to expand A1 tensor.")

                //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                //stored in these matrices as they will be updated before they are used for anything.
                CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                nadd += add;
                subspace_expanded = true;
                ++m_onesite_expansions;
            }
            //resize all of the working buffers. 
            CALL_AND_HANDLE(buf.resize(A1().shape(0), A1().shape(1)), "Failed to resize working arrays.");
                          
            return subspace_expanded;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing up the tree.");
        }
    }

    //accessor functions for the subspace expansion object
    bool& only_apply_when_no_unoccupied(){return m_only_apply_when_no_unoccupied;}
    const bool& only_apply_when_no_unoccupied() const{return m_only_apply_when_no_unoccupied;}

    bool& eval_but_dont_apply(){return m_eval_but_dont_apply;}
    const bool& eval_but_dont_apply() const{return m_eval_but_dont_apply;}

    real_type& spawning_threshold(){return m_spawning_threshold;}
    const real_type& spawning_threshold() const{return m_spawning_threshold;}

    real_type& unoccupied_threshold(){return m_unoccupied_threshold;}
    const real_type& unoccupied_threshold() const{return m_unoccupied_threshold;}

    size_type& minimum_unoccupied(){return m_minimum_unoccupied;}    
    const size_type& minimum_unoccupied() const {return m_minimum_unoccupied;}    

    const size_type& neigenvalues() const {return m_neigenvalues;}    

    size_type& maximum_bond_dimension(){return m_max_dim;}
    const size_type& maximum_bond_dimension() const{return m_max_dim;}

    orthogonality::truncation_mode& truncation_mode() {return m_trunc_mode;}
    const orthogonality::truncation_mode& truncation_mode() const {return m_trunc_mode;}

    const size_type& Nonesite() const{return m_onesite_expansions;}
    const size_type& Ntwosite() const{return m_twosite_expansions;}
protected:
    void resize_two_site_energy_buffers(size_type n, const hdata& a1, const hdata& a2, size_type mode)
    {
        auto atens = a2.as_rank_3(mode);
        for(size_type _n = 0; _n < n; ++_n)
        {
            CALL_AND_HANDLE(m_2s_1[_n].resize(a1.shape(0), a1.shape(1)), "Failed to reshape two-site energy buffer object.");
            CALL_AND_HANDLE(m_2s_2[_n].resize(atens.shape(0), atens.shape(1), atens.shape(2)), "Failed to reshape two-site energy buffer object.");
        }
    }

    size_type get_nunoccupied(const dmat_type&  pops, real_type& scale_factor)
    {
        scale_factor = 0.0;
        size_type nunocc = 0;

        for(size_type i = 0; i  < pops.size(); ++i)
        {   
            scale_factor += pops(i,i)*pops(i, i);
        }
        scale_factor = std::sqrt(scale_factor);

        for(size_type i = 0; i  < pops.size(); ++i)
        {
            if(m_trunc_mode == orthogonality::truncation_mode::singular_values_truncation)
            {
                if(pops(i, i)/scale_factor < m_unoccupied_threshold){++nunocc;}
            }
            else
            {
                if(pops(i, i)*pops(i,i)/scale_factor < m_unoccupied_threshold){++nunocc;}
            }
        }
        return nunocc;
    }

protected:
    two_site_variations<T, backend> m_twosite;

    vec_type m_rvec;
    mat_type m_trvec;
    mat_type m_trvec2;

    std::vector<size_type> m_dim;

    //add in a second set of indices
    linalg::vector<size_type> m_inds;
    linalg::vector<T> m_coeffs;
    triad_type m_2s_1;
    std::vector<linalg::tensor<T, 3, backend>> m_2s_2;


    real_type m_spawning_threshold = -1.0;
    real_type m_unoccupied_threshold = -1.0;
    size_type m_minimum_unoccupied = 0;
    size_type m_neigenvalues = 2;
    size_type m_onesite_expansions = 0;
    size_type m_twosite_expansions = 0;
    size_type m_maxcapacity;
    size_type m_max_dim = 0;

    orthogonality::truncation_mode m_trunc_mode = orthogonality::truncation_mode::singular_values_truncation;

    linalg::singular_value_decomposition<mat_type, true> m_svd;
    mat_type m_twosite_energy;
    mat_type m_twosite_temp;

    mat_type m_U;
    linalg::diagonal_matrix<real_type, backend> m_S;
    mat_type m_V;
    bool m_only_apply_when_no_unoccupied = false;
    bool m_eval_but_dont_apply = false;

};  //class subspace_expansion_full_two_site
}   //namespace ttns

#endif

