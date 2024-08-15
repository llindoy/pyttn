#ifndef TTNS_LIB_SWEEPING_ALGORITHM_TDVP_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_TDVP_HPP

#include "sweeping_algorithm.hpp"

#include "environment/sum_of_product_operator_env.hpp"
#include "update/tdvp_engine.hpp"
#include "subspace_expansion/variance_subspace_expansion_engine.hpp"

namespace ttns
{

template <typename T, typename backend, template <typename, typename> class ttn_class>
class _one_site_tdvp : public sweeping_algorithm<T, backend, ttn_class, tdvp_engine, sop_environment, single_site>
{
public:
    using base_type =  sweeping_algorithm<T, backend, ttn_class, tdvp_engine, sop_environment, single_site>;

    using update_type = typename base_type::update_type;
    using subspace_type = typename base_type::subspace_type;
    using environment_type = typename base_type::environment_type;

    using update_params = typename update_type::parameter_list;
    using subspace_params = typename subspace_type::parameter_list;
    using environment_params = typename environment_type::parameter_list;

    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;

    using ttn_type = ttn_class<T, backend>;
    using env_container_type = typename base_type::env_container_type;
    using env_node_type = typename base_type::env_node_type;
    using env_data_type = typename base_type::env_data_type;
    using env_type = typename base_type::env_type;

public:
    _one_site_tdvp() : base_type() {}
    _one_site_tdvp(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type nstep = 1, size_type num_threads = 1)   : base_type(A, ham, {krylov_dim, nstep}, {}, {}, num_threads){}

    _one_site_tdvp(const _one_site_tdvp& o) = default;
    _one_site_tdvp(_one_site_tdvp&& o) = default;

    _one_site_tdvp& operator=(const _one_site_tdvp& o) = default;
    _one_site_tdvp& operator=(_one_site_tdvp&& o) = default;

    void initialise(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type nstep = 1, size_type num_threads = 1)
    {
        CALL_AND_RETHROW(base_type::initialise(A, ham, {krylov_dim, nstep}, {}, {}, num_threads));
    }
};

template <typename T, typename backend>
using one_site_tdvp = _one_site_tdvp<T, backend, ttn>;

template <typename T, typename backend>
using multiset_one_site_tdvp = _one_site_tdvp<T, backend, ms_ttn>;



template <typename T, typename backend, template <typename, typename> class ttn_class>
class _adaptive_one_site_tdvp : public sweeping_algorithm<T, backend, ttn_class, tdvp_engine, sop_environment, variance_subspace_expansion>
{
public:
    using base_type =  sweeping_algorithm<T, backend, ttn_class, tdvp_engine, sop_environment, variance_subspace_expansion>;

    using update_type = typename base_type::update_type;
    using subspace_type = typename base_type::subspace_type;
    using environment_type = typename base_type::environment_type;

    using update_params = typename update_type::parameter_list;
    using subspace_params = typename subspace_type::parameter_list;
    using environment_params = typename environment_type::parameter_list;

    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;

    using ttn_type = ttn_class<T, backend>;
    using env_container_type = typename base_type::env_container_type;
    using env_node_type = typename base_type::env_node_type;
    using env_data_type = typename base_type::env_data_type;
    using env_type = typename base_type::env_type;

public:
    _adaptive_one_site_tdvp() : base_type() {}
    _adaptive_one_site_tdvp(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type nstep = 1, size_type eigensolver_krylov_dim = 4, size_type neigenvalues = 2, size_type num_threads = 1)   : base_type(A, ham, {krylov_dim, nstep}, {}, {eigensolver_krylov_dim, neigenvalues}, num_threads){}

    _adaptive_one_site_tdvp(const _adaptive_one_site_tdvp& o) = default;
    _adaptive_one_site_tdvp(_adaptive_one_site_tdvp&& o) = default;

    _adaptive_one_site_tdvp& operator=(const _adaptive_one_site_tdvp& o) = default;
    _adaptive_one_site_tdvp& operator=(_adaptive_one_site_tdvp&& o) = default;

    void initialise(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type nstep = 1, size_type eigensolver_krylov_dim = 4, size_type neigenvalues = 2, size_type num_threads = 1)
    {
        CALL_AND_RETHROW(base_type::initialise(A, ham, {krylov_dim, nstep}, {}, {eigensolver_krylov_dim, neigenvalues}, num_threads));
    }

    template <typename ... Args>
    bool operator()(Args&& ... args)
    {
        //for the time evolution scheme we set the subspace weighting factor to be equal to the timestep through the partial step dt/2
        this->subspace_weighting_factor() = this->dt()/2.0;
        CALL_AND_RETHROW(return static_cast<base_type*>(this)->operator()(std::forward<Args>(args)...));
    }
};

template <typename T, typename backend>
using adaptive_one_site_tdvp = _adaptive_one_site_tdvp<T, backend, ttn>;

//template <typename T, typename backend>
//using multiset_adaptive_one_site_tdvp = _adaptive_one_site_tdvp<T, backend, ms_ttn>;

}

#endif

