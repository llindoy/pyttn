#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_HPP

#include "two_site_energy_variations.hpp"
#include "../environment/single_particle_operator.hpp"
#include "../environment/sum_of_product_operator_env.hpp"

namespace ttns
{
template <typename T, typename backend>
class subspace_expansion
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

    using engine_type = decomposition_engine<T, backend, false>;
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

    using dmat_type = typename engine_type::dmat_type;

    using hnode = ttn_node<T, backend>;
    using hdata = ttn_node_data<T, backend>;

    using matnode = typename tree<mat_type>::node_type;

public:
    subspace_expansion() : m_twosite(), m_spawning_threshold(-1), m_unoccupied_threshold(-1), m_minimum_unoccupied(0), m_neigenvalues(2), m_only_apply_when_no_unoccupied(false) {}
    subspace_expansion(const ttn<T, backend>& A, const env_container_type& ham, size_type neigs, size_type seed = 0)  : m_twosite(seed), m_spawning_threshold(-1), m_unoccupied_threshold(-1), m_minimum_unoccupied(0), m_only_apply_when_no_unoccupied(false)
    {
        CALL_AND_HANDLE(initialise(A, ham, neigs), "Failed to construct subspace_expansion.");
    }   
    subspace_expansion(const subspace_expansion& o) = default;
    subspace_expansion(subspace_expansion&& o) = default;

    subspace_expansion& operator=(const subspace_expansion& o) = default;
    subspace_expansion& operator=(subspace_expansion&& o) = default;

    void initialise(const ttn<T, backend>& A, const env_container_type& ham, size_type neigs)
    {
        try
        {
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");

            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");
            m_neigenvalues = neigs;

            size_type maxcapacity = 0;  size_type maxnmodes = 0;
            for(const auto& a : A)
            {
                size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}
                size_type nmodes = a().nmodes();    if(nmodes > maxnmodes){maxnmodes = nmodes;}
            }

            size_type max_two_site_energy_terms = 0;

            for(const auto& h : ham)
            {
                size_type two_site_energy_terms = twosite::get_nterms(h());
                if(two_site_energy_terms > max_two_site_energy_terms){max_two_site_energy_terms = two_site_energy_terms;}
            }

            CALL_AND_HANDLE(m_2s_1.resize(max_two_site_energy_terms), "Failed to resize two site spf buffer.");
            CALL_AND_HANDLE(m_2s_2.resize(max_two_site_energy_terms), "Failed to resize two site mf buffer.");

            for(size_type i = 0; i < max_two_site_energy_terms; ++i)
            {
                CALL_AND_HANDLE(m_2s_1[i].reallocate(maxcapacity), "Failed to reallocate two site spf buffer.");
                CALL_AND_HANDLE(m_2s_2[i].reallocate(maxcapacity), "Failed to reallocate two site mf buffer.");
            }

            m_maxcapacity = maxcapacity;
            m_inds.resize(maxnmodes);
            m_dim.resize(maxnmodes);

            m_S.resize(m_neigenvalues);
            m_U.reallocate(m_neigenvalues*maxcapacity);
            m_V.reallocate(m_neigenvalues*maxcapacity);
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
            CALL_AND_HANDLE(m_rvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_trvec2.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_1.clear(), "Failed to clear the rvec object.");
            CALL_AND_HANDLE(m_2s_2.clear(), "Failed to clear the rvec object.");
            m_onesite_expansions = 0;
            m_twosite_expansions = 0;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    template <typename IntegType, typename BufType>
    bool down(hnode& A1, hnode& A2, mat_type& r, const dmat_type& pops, env_node_type& h, const env_type& op, IntegType& eigensolver, BufType& buf, real_type svd_scale)
    {
        try
        {
            if(A2.is_root() || A1.is_root() ){return false;}
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A2().dim(mode) >= A2().max_dim(mode) || A1().hrank() >= A1().max_hrank()){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A2().max_dim(mode) - A2().dim(mode);
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            auto A2tens = A2().as_rank_3(mode);
            size_type max_dimension = A2tens.shape(0)*A2tens.shape(2);
            //size_type curr_dim = A2().dim(mode);

            //if the matricisation of the tensor we are currently trying to expand is square then we don't even attempt to expand
            if(A2tens.shape(1) >= max_dimension){return false;}

            real_type scale_factor = 0;
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
            if(A1().shape(1) < A1().shape(0) && A1().shape(1) != 1)
            {
                //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
                size_type nterms = twosite::get_nterms(h()); 
                ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
                CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
                CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");

                twosite::set_indices(h(), m_inds); 

                //compute the one site objects that are used to compute the two site projected energy
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, h, op, m_2s_1, buf.HA, buf.temp, m_inds, r, true), 
                                "Failed to construct the component of the two site energy acting on the lower site.");
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, h, m_2s_2, buf.HA, buf.temp, buf.temp2, m_inds, true), 
                                "Failed to construct the component of the two site energy acting on the upper site.");

                //now we compute the singular value using the sparse functions

                CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A2(), mode, m_rvec), "Failed to generate othrogonal tensor.");
                CALL_AND_HANDLE(m_V.resize(m_neigenvalues, m_rvec.size()), "Failed to resize m_U array.");

                //now we compute the singular value using the sparse functions
                for(size_type i = 0; i < m_neigenvalues; ++i)
                {
                    CALL_AND_HANDLE(m_V[i] = m_rvec, "Failed to copy rvec.");
                }
                
                //now compute the eigenvectors using this 
                bool mconjm = true;
                size_type maxkrylov_dim = eigensolver.krylov_dim();
                if(m_rvec.size() < maxkrylov_dim){maxkrylov_dim = m_rvec.size();}

                //computes the complex conjugate of the right singular vectors
                CALL_AND_HANDLE(eigensolver(m_V, m_S, m_twosite, m_2s_1, m_2s_2, nterms, m_trvec, m_trvec2, buf.temp[0], mconjm), "Failed to compute sparse svd.");

                CALL_AND_HANDLE(m_V = linalg::conj(m_V), "Failed to conjugate the right singular vectors.");

                for(size_type i = 0; i < m_S.size(); ++i)
                {
                    real_type sv = 0;
                    //check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
                    if(std::real(m_S(i, i)) > 0){sv = std::sqrt(std::real(m_S(i, i)))*svd_scale;}   
                    if(!std::isnan(sv))
                    {
                        if(sv > m_spawning_threshold){++nadd;}
                    }
                }

                //if(nadd == 0){++nadd;}
                if(nadd > max_add){nadd = max_add;}
                if(nadd != 0)
                {
                    //expand the A1 tensor zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), buf.temp[0], A1.nmodes(), nadd, m_dim), "Failed to expand A1 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], nadd), "Failed to expand R matrix.");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), buf.temp[0], mode, m_V, nadd, m_dim), "Failed to expand A2 tensor.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    subspace_expanded = true;
                }
                ++m_twosite_expansions;
            } 
            //otherwise we have a square other tensor and so the two-site algorithm will fail to provide a search direction so instead add on
            //a random search direction to this tensor
            else
            {
                while(nadd < required_terms)
                {
                    size_type add = 1;
                    //now we compute the singular value using the sparse functions
                    CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A2(), mode, m_rvec), "Failed to generate othrogonal tensor.");

                    CALL_AND_HANDLE(m_V.resize(1, m_rvec.size()), "Failed to resize V array.");
                    CALL_AND_HANDLE(m_V[0] = m_rvec, "Failed to copy random vector to V");

                    //expand the A1 tensor zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), buf.temp[0], A1.nmodes(), add, m_dim), "Failed to expand A1 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], add), "Failed to expand R matrix.");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), buf.temp[0], mode, m_V, add, m_dim), "Failed to expand A2 tensor.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");
                
                    ++nadd;
                    subspace_expanded = true;
                }
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

    template <typename IntegType, typename BufType>
    bool up(hnode& A1, hnode& A2, mat_type& r, const dmat_type& pops, env_node_type& h, const env_type& op, IntegType& eigensolver, BufType& buf, real_type svd_scale)
    {
        try
        {
            size_type mode = A1.child_id();

            //first check if the mode we are interested in has the capacity to be expanded
            if(A1().hrank() >= A1().max_hrank()){return false;}

            //get the maximum number of elements that we can add to the tensor
            size_type max_add = A1().max_hrank() - A1().hrank();
            bool subspace_expanded = false;

            //determine if we should attempt subspace expansion.
            
            size_type max_dimension = A1().shape(0);
            //size_type curr_dim = A1().hrank();

            auto A2tens = A2().as_rank_3(mode);
            //if the matricisation of the tensor we are currently trying to expand is square then we don't even attempt to expand
            if(A1().shape(1) >= max_dimension){return false;}

            real_type scale_factor = 0;
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
            if(A2tens.shape(1) < A2tens.shape(0)*A2tens.shape(2) && A2tens.shape(1) != 1)
            {
                //now resize the onesite buffer objects used for evaluation of the two site (projected energy) so that everything is the correct size
                size_type nterms = twosite::get_nterms(h()); 

                ASSERT(m_2s_1.size() >= nterms, "Unable to store temporaries needed for computing the two-site energy.  The buffers are not sufficient.");
                CALL_AND_HANDLE(resize_two_site_energy_buffers(nterms, A1(), A2(), mode), "Failed to resize the objects needed to compute the two-site energy.");
                CALL_AND_HANDLE(m_inds.resize(nterms), "Failed to resize indices array.");

                twosite::set_indices(h(), m_inds); 

                //compute the one site objects that are used to compute the two site projected energy
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_upper(A2, h, m_2s_2, buf.HA, buf.temp, buf.temp2, m_inds, true), 
                                "Failed to construct the component of the two site energy acting on the upper site.");
                CALL_AND_HANDLE(twosite::construct_two_site_energy_terms_lower(A1, h, op, m_2s_1, buf.HA, buf.temp, m_inds, r, true), 
                                "Failed to construct the component of the two site energy acting on the lower site.");
                        


                CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A1(), A1().nmodes(), m_rvec), "Failed to generate othrogonal tensor.");
                CALL_AND_HANDLE(m_U.resize(m_neigenvalues, m_rvec.size()), "Failed to resize m_U array.");
                //now we compute the singular value using the sparse functions
                for(size_type i = 0; i < m_neigenvalues; ++i)
                {
                    CALL_AND_HANDLE(m_U[i] = m_rvec, "Failed to copy rvec into U");
                }

                //firs go ahead and generate random r vector
                size_type maxkrylov_dim = eigensolver.krylov_dim();
                if(m_rvec.size() < maxkrylov_dim){maxkrylov_dim = m_rvec.size();}

                bool mconjm = false;

                //computes U but stored with its columns as rows - e.g. this is U^T.  E.g. the singular vectors are currently the rows of m_U
                CALL_AND_HANDLE(eigensolver(m_U, m_S, m_twosite, m_2s_1, m_2s_2, nterms, m_trvec, m_trvec2, buf.temp[0], mconjm), "Failed to compute sparse svd.");

                for(size_type i = 0; i < m_S.size(); ++i)
                {
                    real_type sv = 0;
                    //check if any of the dominant svds of the Hamiltonian acting on the twosite coefficient tensor are occupied through the half step
                    if(std::real(m_S(i, i)) > 0){sv = std::sqrt(std::real(m_S(i, i)))*svd_scale;}
                    if(!std::isnan(sv))
                    {
                        if(sv > m_spawning_threshold){++nadd;}
                    }
                }

                if(nadd > max_add){nadd = max_add;}
                //here we need to compute the eigenstates of the 
                if(nadd != 0)
                {
                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), buf.temp[0], mode, nadd, m_dim), "Failed to expand A2 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], nadd), "Failed to expand R matrix.");

                    //expand the A1 tensor around the index pointing to the root zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), buf.temp[0], A1.nmodes(), m_U, nadd, m_dim), "Failed to expand A1 tensor.");

                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    subspace_expanded = true;
                    ++m_twosite_expansions;
                }
            }
            //otherwise we just use random one-site tensor expansion
            else
            {
                while(nadd < required_terms)
                {
                    size_type add = 1;
                    CALL_AND_HANDLE(m_twosite.generate_orthogonal_trial_vector(A1(), A1().nmodes(), m_rvec), "Failed to generate othrogonal tensor.");
                    CALL_AND_HANDLE(m_U.resize(1, m_rvec.size()), "Failed to resize V array.");
                    CALL_AND_HANDLE(m_U[0] = m_rvec, "Failed to copy random vector to V");

                    //expand the A2 tensor padding with the right singular vectors
                    CALL_AND_HANDLE(twosite::expand_tensor(A2(), buf.temp[0], mode, add, m_dim), "Failed to expand A2 tensor.");

                    //expand the r-matrix
                    CALL_AND_HANDLE(twosite::expand_matrix(r, buf.temp[0], add), "Failed to expand R matrix.");

                    //expand the A1 tensor around the index pointing to the root zero padding
                    CALL_AND_HANDLE(twosite::expand_tensor(A1(), buf.temp[0], A1.nmodes(), m_U, add, m_dim), "Failed to expand A1 tensor.");


                    //resize the Hamiltonian object stored at the child node to the correct size.  Here we don't care about the values 
                    //stored in these matrices as they will be updated before they are used for anything.
                    CALL_AND_HANDLE(h().resize_matrices(r.shape(0), r.shape(1)), "Failed to resize Hamiltonian matrices.");

                    ++nadd;
                    subspace_expanded = true;
                }
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

    real_type& spawning_threshold(){return m_spawning_threshold;}
    const real_type& spawning_threshold() const{return m_spawning_threshold;}

    real_type& unoccupied_threshold(){return m_unoccupied_threshold;}
    const real_type& unoccupied_threshold() const{return m_unoccupied_threshold;}

    size_type& minimum_unoccupied(){return m_minimum_unoccupied;}    
    const size_type& minimum_unoccupied() const {return m_minimum_unoccupied;}    

    const size_type& neigenvalues() const {return m_neigenvalues;}    

    template <typename Arg>
    void set_rng(const Arg& rng){m_twosite.set_rng(rng);}

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
            //std::cerr << pops(i, i)/scale_factor << std::endl;
            if(pops(i, i)/scale_factor < m_unoccupied_threshold){++nunocc;}
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
    triad_type m_2s_1;
    std::vector<linalg::tensor<T, 3, backend>> m_2s_2;

    real_type m_spawning_threshold;
    real_type m_unoccupied_threshold;
    size_type m_minimum_unoccupied;
    size_type m_neigenvalues;
    size_type m_onesite_expansions;
    size_type m_twosite_expansions;
    size_type m_maxcapacity;

    mat_type m_U;
    linalg::diagonal_matrix<T, backend> m_S;
    mat_type m_V;
    bool m_only_apply_when_no_unoccupied;

};  //class subspace_expansion
}   //namespace ttns

#endif

