#ifndef TTNS_LIB_SWEEPING_ALGORITHM_TDVP_ENGINE_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_TDVP_ENGINE_HPP

#include <common/omp.hpp>
#include <utils/iterative_linear_algebra/expmv.hpp>

#include "simple_update_parameter_list.hpp"

namespace ttns
{

template <typename T, typename backend, template <typename , typename> class ttn_class>
class tdvp_engine<T, backend, ttn_class, sop_environment>
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using environment_type = sop_environment<T, backend, ttn_class>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_data_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;

    using ttn_type = ttn_class<T, backend>;
    using hnode = typename ttn_type::node_type;
    using mat_type = linalg::matrix<T, backend>;
    using bond_matrix_type = typename ttn_type::bond_matrix_type;

    using buffer_type = typename environment_type::buffer_type;
    using expmv_type = utils::expmv<T, backend, false>;

    using parameter_list = simple_update_parameter_list;
public:
    tdvp_engine() : m_dt(0), m_t(0), m_coeff(1) {}
    tdvp_engine(const ttn_type& A, size_type krylov_dim = 12, size_type ndt = 1) : m_dt(0), m_t(0), m_coeff(1)
    {
        CALL_AND_HANDLE(initialise(A, krylov_dim, ndt), "Failed to construct tdvp_engine.");
    }   
    tdvp_engine(const tdvp_engine& o) = default;
    tdvp_engine(tdvp_engine&& o) = default;

    tdvp_engine& operator=(const tdvp_engine& o) = default;
    tdvp_engine& operator=(tdvp_engine&& o) = default;
    
    void initialise(const ttn_type& A, size_type krylov_dim = 16, size_type ndt=1)
    {
        try
        {
            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");
            
            size_type maxcapacity = 0;
            for(const auto& a : A){size_type capacity = a().capacity();    if(capacity > maxcapacity){maxcapacity = capacity;}}
            CALL_AND_HANDLE(m_expmv.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");
            m_expmv.nsteps() = ndt;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void initialise(const ttn_type& A, const parameter_list& o){CALL_AND_RETHROW(initialise(A, o.krylov_dim));}
    void initialise(const ttn_type& A, parameter_list&& o){CALL_AND_RETHROW(initialise(A, o.krylov_dim));}

    void clear()
    {
        try
        {
            CALL_AND_HANDLE(m_expmv.clear(), "Failed to clear the krylov subspace engine.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

    void advance_half_step(){m_t += m_dt/2.0;}

    T& coefficient(){return m_coeff;}
    const T& coefficient() const{return m_coeff;}

    real_type& t(){return m_t;}
    const real_type& t() const{return m_t;}

    real_type& dt(){return m_dt;}
    const real_type& dt() const{return m_dt;}

    real_type& expmv_tol(){return m_expmv.error_tolerance();}
    const real_type& expmv_tol() const {return m_expmv.error_tolerance();}

    size_t& krylov_steps(){return m_expmv.nsteps();}
    const size_t& krylov_steps() const {return m_expmv.nsteps();}

    const expmv_type& expmv() const{return m_expmv;}
    expmv_type& expmv(){return m_expmv;}

    size_type update_site_tensor(hnode& A, const environment_type& env, env_node_type& h, env_type& op)
    {                    
        try
        {
          if(!A.is_leaf())
          {
              env.ceb.set_pointer(&(A()));
#ifdef PARALLELISE_FOR_LOOPS
              CALL_AND_HANDLE              
              (
                  return m_expmv(A().as_matrix(), m_dt/2.0, m_coeff, env.ceb, h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2),
                  "Failed to evolve the branch coefficient matrix."
              );
#else
              CALL_AND_HANDLE
              (
                  return m_expmv(A().as_matrix(), m_dt/2.0, m_coeff, env.ceb, h, op, env.buffer().HA[0], env.buffer().temp[0], env.buffer().temp2[0]),
                  "Failed to evolve the branch coefficient matrix."
              );
#endif
              env.ceb.unset_pointer();
          }
          else
          {
              //if we are at a leaf node we update its child operators.  This is simply done by calling the update yunction
              //v
              CALL_AND_HANDLE(op.update(A.leaf_index(), m_t, m_dt/2.0), "Failed to update primitive Hamiltonian object.");

#ifdef PARALLELISE_FOR_LOOPS
              CALL_AND_HANDLE
              (
                  return  m_expmv(A().as_matrix(), m_dt/2.0, m_coeff, env.cel, h, op, env.buffer().HA, env.buffer().temp),
                  "Failed to evolve the leaf coefficient matrix."
              );
#else
              CALL_AND_HANDLE
              (
                  return m_expmv(A().as_matrix(), m_dt/2.0, m_coeff, env.cel, h, op, env.buffer().HA[0], env.buffer().temp[0]),
                  "Failed to evolve the leaf coefficient matrix."
              );
#endif
            }
        }        
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("Failed to update node tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to update node tensor.");
        }

    }

    void update_bond_tensor(bond_matrix_type& r, const environment_type& env, env_node_type& h, env_type& op)
    {
        try
        {
#ifdef PARALLELISE_FOR_LOOPS
            CALL_AND_HANDLE(m_expmv(r, -m_dt/2.0, m_coeff, env.fha, h, op, env.buffer().HA, env.buffer().temp), "Failed to time evolve the r matrix backwards in time.");
#else   
            CALL_AND_HANDLE(m_expmv(r, -m_dt/2.0, m_coeff, env.fha, h, op, env.buffer().HA[0]), "Failed to time evolve the r matrix backwards in time.");
#endif        
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("Failed to update R tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to update R tensor.");
        }
    }

protected:
    //the krylov subspace engine
    expmv_type m_expmv;

    real_type m_dt;
    real_type m_t;
    T m_coeff;
    
};  //class tdvp_engine

}


#endif

