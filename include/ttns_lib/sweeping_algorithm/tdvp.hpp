#ifndef TTNS_LIB_SWEEPING_ALGORITHM_TDVP_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_TDVP_HPP

#include "sweeping_algorithm.hpp"

#include "environment/sum_of_product_operator_env.hpp"
#include "update/tdvp_engine.hpp"

namespace ttns
{

template <typename T, typename backend>
class one_site_tdvp : public sweeping_algorithm<T, backend, ttn, tdvp_engine, sop_environment, single_site>
{
public:
    using base_type =  sweeping_algorithm<T, backend, ttn, tdvp_engine, sop_environment, single_site>;

    using update_type = typename base_type::update_type;
    using subspace_type = typename base_type::subspace_type;
    using environment_type = typename base_type::environment_type;

    using update_params = typename update_type::parameter_list;
    using subspace_params = typename subspace_type::parameter_list;
    using environment_params = typename environment_type::parameter_list;

    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;

    using ttn_type = ttn<T, backend>;
    using env_container_type = typename base_type::env_container_type;
    using env_node_type = typename base_type::env_node_type;
    using env_data_type = typename base_type::env_data_type;
    using env_type = typename base_type::env_type;

public:
    one_site_tdvp() : base_type() {}
    one_site_tdvp(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type nstep = 1, size_type num_threads = 1)   : base_type(A, ham, {krylov_dim, nstep}, {}, {}, num_threads){}

    one_site_tdvp(const one_site_tdvp& o) = default;
    one_site_tdvp(one_site_tdvp&& o) = default;

    one_site_tdvp& operator=(const one_site_tdvp& o) = default;
    one_site_tdvp& operator=(one_site_tdvp&& o) = default;

    void initialise(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type nstep = 1, size_type num_threads = 1)
    {
        CALL_AND_RETHROW(base_type::initialise(A, ham, {krylov_dim, nstep}, {}, {}, num_threads));
    }
};

}

#endif

