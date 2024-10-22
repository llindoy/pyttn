#ifndef TTNS_LIB_SWEEPING_ALGORITHM_TDVP_ENGINE_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_TDVP_ENGINE_HPP

#include <common/omp.hpp>
#include <utils/iterative_linear_algebra/expmv.hpp>

#include "simple_update_parameter_list.hpp"

namespace ttns
{

//implementation of the tdvp update for a standard ttn type
template <typename T, typename backend>
class tdvp_engine<T, backend, ttn, sop_environment>
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using environment_type = sop_environment<T, backend, ttn>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_data_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;

    using ttn_type = ttn<T, backend>;
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
            for(const auto& a : A){size_type capacity = a.buffer_maxcapacity();    if(capacity > maxcapacity){maxcapacity = capacity;}}
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

    const bool& use_time_dependent_hamiltonian() const{return m_use_time_dependent_hamiltonian;}
    bool& use_time_dependent_hamiltonian(){return m_use_time_dependent_hamiltonian;}

    size_type update_site_tensor(hnode& A, const environment_type& env, env_node_type& h, env_type& op)
    {                    
        try
        {
          if(!A.is_leaf())
          {
              env.ceb.set_pointer(&(A()));
              CALL_AND_HANDLE              
              (
                  return m_expmv(A().as_matrix(), m_dt/2.0, m_coeff, env.ceb, h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2),
                  "Failed to evolve the branch coefficient matrix."
              );
              env.ceb.unset_pointer();
          }
          else
          {
              //if we are at a leaf node we update its child operators.  This is simply done by calling the update yunction
              CALL_AND_HANDLE(op.update(A.leaf_index(), m_t, m_dt/2.0), "Failed to update primitive Hamiltonian object.");
              CALL_AND_HANDLE
              (
                  return  m_expmv(A().as_matrix(), m_dt/2.0, m_coeff, env.cel, h, op, env.buffer().HA, env.buffer().temp),
                  "Failed to evolve the leaf coefficient matrix."
              );
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
            CALL_AND_HANDLE(m_expmv(r, -m_dt/2.0, m_coeff, env.fha, h, op, env.buffer().temp, env.buffer().HA), "Failed to time evolve the r matrix backwards in time.");
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

    void advance_hamiltonian(ttn_type& A, environment_type& env, env_container_type& h, env_type& op)
    {
        if(m_use_time_dependent_hamiltonian)
        {
            op.update_coefficients(m_t+(m_dt/4.0));
            if(op.has_time_dependent_operators())
            {
                for(auto z : common::rzip(A, h))
                {
                    const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                    CALL_AND_HANDLE(env.update_env_up(op, a, hspf, true), "Failed to update the environment tensor.");
                }
            }
        }
    }
protected:
    //the krylov subspace engine
    expmv_type m_expmv;

    real_type m_dt;
    real_type m_t;
    T m_coeff;

    bool m_use_time_dependent_hamiltonian = false;
    
};  //class tdvp_engine



//implementation of the tdvp update for a multiset ttn type
template <typename T, typename backend>
class tdvp_engine<T, backend, ms_ttn, sop_environment>
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using environment_type = sop_environment<T, backend, ms_ttn>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_data_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;

    using ttn_type = ms_ttn<T, backend>;
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
            for(const auto& a : A){size_type capacity = a.buffer_maxcapacity();    if(capacity > maxcapacity){maxcapacity = capacity;}}
            CALL_AND_HANDLE(m_expmv.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");
            m_expmv.nsteps() = ndt;
            CALL_AND_RETHROW(mbuf.initialise(A));
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
            CALL_AND_RETHROW(mbuf.clear());
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

    const bool& use_time_dependent_hamiltonian() const{return m_use_time_dependent_hamiltonian;}
    bool& use_time_dependent_hamiltonian(){return m_use_time_dependent_hamiltonian;}

    size_type update_site_tensor(hnode& A, const environment_type& env, env_node_type& h, env_type& op)
    {                    
        try
        {
            size_type nevals=0;
            mbuf.setup(A());
            if(!A.is_leaf())
            {
                env.ceb.set_pointer(&(A()));
                CALL_AND_HANDLE              
                (
                    nevals += m_expmv(mbuf.A(), m_dt/2.0, m_coeff, env.ceb, h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2, mbuf.res()),
                    "Failed to evolve the branch coefficient matrix."
                );
                env.ceb.unset_pointer();
            }
            else
            {
                //if we are at a leaf node we update its child operators.  This is simply done by calling the update yunction
                CALL_AND_HANDLE(op.update(A.leaf_index(), m_t, m_dt/2.0), "Failed to update primitive Hamiltonian object.");

                env.cel.set_pointer(&(A()));
                CALL_AND_HANDLE
                (
                    nevals += m_expmv(mbuf.A(), m_dt/2.0, m_coeff, env.cel, h, op, env.buffer().HA, env.buffer().temp, mbuf.res()),
                    "Failed to evolve the leaf coefficient matrix."
                );
                env.cel.unset_pointer();
            }
            ttn_type::unpack(mbuf.A(), A());
            return nevals;
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
            mbuf.setup(r);
            env.fha.set_pointer(&(r));
            CALL_AND_HANDLE(m_expmv(mbuf.A(), -m_dt/2.0, m_coeff, env.fha, h, op, env.buffer().temp, env.buffer().HA, mbuf.res()), "Failed to time evolve the r matrix backwards in time.");
            env.fha.unset_pointer();
            ttn_type::unpack(mbuf.A(), r);
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

    void advance_hamiltonian(ttn_type& A, environment_type& env, env_container_type& h, env_type& op)
    {
        if(m_use_time_dependent_hamiltonian)
        {
            op.update_coefficients(m_t+(m_dt/4.0));

            if(op.has_time_dependent_operators())
            {
                for(auto z : common::rzip(A, h))
                {
                    const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
                    CALL_AND_HANDLE(env.update_env_up(op, a, hspf, true), "Failed to update the environment tensor.");
                }
            }
        }
    }
protected:
    //the krylov subspace engine
    expmv_type m_expmv;

    real_type m_dt;
    real_type m_t;
    T m_coeff;

    bool m_use_time_dependent_hamiltonian = false;
    
    multiset_update_buffer<T, backend> mbuf;
};  //class tdvp_engine

}


#endif

