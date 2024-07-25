#ifndef TTNS_LIB_SWEEPING_ALGORITHM_VARIANCE_SUBSPACE_EXPANSION_ENGINE_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_VARIANCE_SUBSPACE_EXPANSION_ENGINE_HPP

#include <random>
#include <iterative_linear_algebra/arnoldi.hpp>
#include "two_site_energy_variations.hpp"
#include "subspace_expansion.hpp"
#include "../sweeping_forward_decl.hpp"
#include "../environment/sum_of_product_operator_env.hpp"

namespace ttns
{


template <typename T, typename backend>
class variance_subspace_expansion<T, backend, ttn, sop_environment>
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using environment_type = sop_environment<T, backend, ttn>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_type = typename environment_type::environment_type;

    using ttn_type = ttn<T, backend>;
    using hnode = typename ttn_type::node_type;

    using mat_type = linalg::matrix<T, backend>;

    using engine_type = decomposition_engine<T, backend, false>;
    using dmat_type = typename engine_type::dmat_type;

    using buffer_type = typename environment_type::buffer_type;

    using twosite = two_site_variations<T, backend>;
    using eigensolver_type = utils::arnoldi<T, backend>;

    struct parameter_list
    {
        parameter_list() : krylov_dim(4), neigenvalues(2){}
        parameter_list(size_type kdim) : krylov_dim(kdim), neigenvalues(2){}
        parameter_list(size_type kdim, size_type neig) : krylov_dim(kdim), neigenvalues(neig){}
        parameter_list(const parameter_list& o) = default;
        parameter_list(parameter_list&& o) = default;
        parameter_list& operator=(const parameter_list& o) = default;
        parameter_list& operator=(parameter_list&& o) = default;

        size_type krylov_dim;
        size_type neigenvalues;
    };

public:
    variance_subspace_expansion() : m_ss_expand(), m_subspace_weighting_factor(1.0) {}
    variance_subspace_expansion(const ttn_type& A, const env_container_type& ham, size_type eigensolver_krylov_dim = 4, size_type neigenvalues = 2)  :  m_ss_expand(),m_subspace_weighting_factor(1.0)
    {
        CALL_AND_HANDLE(initialise(A, ham, eigensolver_krylov_dim, neigenvalues), "Failed to construct variance_subspace_expansion.");
    }   
    variance_subspace_expansion(const variance_subspace_expansion& o) = default;
    variance_subspace_expansion(variance_subspace_expansion&& o) = default;

    variance_subspace_expansion& operator=(const variance_subspace_expansion& o) = default;
    variance_subspace_expansion& operator=(variance_subspace_expansion&& o) = default;
    
    void initialise(const ttn_type& A, const env_container_type& ham, size_type eigensolver_krylov_dim = 4, size_type neigenvalues = 2)
    {
        try
        {
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must have been orthogonalised.");

            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");
            
            size_type maxcapacity = 0;
            for(const auto& a : A){size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}}
            CALL_AND_HANDLE(m_eigensolver.resize(eigensolver_krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");
            m_ss_expand.initialise(A, ham, neigenvalues);
            m_eigensolver.mode() = utils::eigenvalue_target::largest_magnitude;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void initialise(const ttn_type& A, const env_container_type& ham, const parameter_list& o){CALL_AND_RETHROW(initialise(A, ham, o.krylov_dim, o.neigenvalues));}
    void initialise(const ttn_type& A, const env_container_type& ham, parameter_list&& o){CALL_AND_RETHROW(initialise(A, ham, o.krylov_dim, o.neigenvalues));}

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_ss_expand.clear(), "Failed to clear subspace expansion object.");
            CALL_AND_HANDLE(m_eigensolver.clear(), "Failed to clear the krylov subspace engine.");
            m_eigensolver.mode() = utils::eigenvalue_target::largest_magnitude;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    bool& only_apply_when_no_unoccupied(){return m_ss_expand.only_apply_when_no_unoccupied();}
    const bool& only_apply_when_no_unoccupied() const{return m_ss_expand.only_apply_when_no_unoccupied();}

    real_type& spawning_threshold(){return m_ss_expand.spawning_threshold();}
    const real_type& spawning_threshold() const{return m_ss_expand.spawning_threshold();}

    real_type& unoccupied_threshold(){return m_ss_expand.unoccupied_threshold();}
    const real_type& unoccupied_threshold() const{return m_ss_expand.unoccupied_threshold();}

    size_type& minimum_unoccupied(){return m_ss_expand.minimum_unoccupied();}    
    const size_type& minimum_unoccupied() const {return m_ss_expand.minimum_unoccupied();}    

    const size_type& neigenvalues() const {return m_ss_expand.neigenvalues();}    

    const eigensolver_type& subspace_eigensolver() const{return m_eigensolver;}
    eigensolver_type& subspace_eigensolver(){return m_eigensolver;}

    real_type& subspace_eigensolver_tol(){return m_eigensolver.error_tolerance();}
    const real_type& subspace_eigensolver_tol() const{return m_eigensolver.error_tolerance();}

    real_type& subspace_eigensolver_reltol(){return m_eigensolver.rel_tol();}
    const real_type& subspace_eigensolver_reltol() const {return m_eigensolver.rel_tol();}

    template <typename Arg>
    void set_rng(const Arg& rng){m_ss_expand.set_rng(rng);}

    const subspace_expansion<T, backend>& subspace_expander()  const{return m_ss_expand;}

    const real_type& subspace_weighting_factor() const{return m_subspace_weighting_factor;}
    real_type& subspace_weighting_factor(){return m_subspace_weighting_factor;}

public:
    //perform the subspace expansion as we are moving down a tree.  This requires us to evaluate the optimal functions to add 
    //into A2.  For A1 they will be overwriten by the r matrix in the next step so we just 
    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_down(hnode& A1, hnode& A2, env_node_type& h, const env_type& op, environment_type& env)
    {
        try
        {
            return m_ss_expand.down(A1, A2, m_r, m_pops, h, op, m_eigensolver, env.buffer(), m_subspace_weighting_factor);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing down the tree.");
        }
    }

    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_up(hnode& A1, hnode& A2, env_node_type& h, const env_type& op, environment_type& env)
    {
        try
        {
            return m_ss_expand.up(A1, A2, m_r, m_pops, h, op, m_eigensolver, env.buffer(), m_subspace_weighting_factor);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing up the tree.");
        }
    }

public:
    const size_type& Nonesite() const{return m_ss_expand.Nonesite();}
    const size_type& Ntwosite() const{return m_ss_expand.Ntwosite();}

protected:
    subspace_expansion<T, backend> m_ss_expand;
    mat_type m_r;
    dmat_type m_pops;

    //the krylov subspace engine
    eigensolver_type m_eigensolver;

    real_type m_subspace_weighting_factor;
};  //class variance_subspace_expansion
}   //namespace ttns

#endif  //TTNS_SUBSPACE_TDVP_ENGINE_HPP//

