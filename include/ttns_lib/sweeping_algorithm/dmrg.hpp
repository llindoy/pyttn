#ifndef TTNS_LIB_SWEEPING_ALGORITHM_DMRG_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_DMRG_HPP


#include "sweeping_algorithm.hpp"

#include "environment/sum_of_product_operator_env.hpp"
#include "update/gso_engine.hpp"


namespace ttns
{

template <typename T, typename backend>
class one_site_dmrg : public sweeping_algorithm<T, backend, ttn, gso_engine, sop_environment, single_site>
{
public:
    using base_type =  sweeping_algorithm<T, backend, ttn, gso_engine, sop_environment, single_site>;

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
    one_site_dmrg() : base_type() {}
    one_site_dmrg(const ttn_type& A, const env_type& ham) : base_type(A, ham, 1){}
    one_site_dmrg(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type num_threads = 1)   : base_type(A, ham, {krylov_dim, 1}, {}, {}, num_threads){}

    one_site_dmrg(const one_site_dmrg& o) = default;
    one_site_dmrg(one_site_dmrg&& o) = default;

    one_site_dmrg& operator=(const one_site_dmrg& o) = default;
    one_site_dmrg& operator=(one_site_dmrg&& o) = default;

    void initialise(const ttn_type& A, const env_type& ham, size_type krylov_dim = 16, size_type num_threads = 1)
    {
        CALL_AND_RETHROW(base_type::initialise(A, ham, {krylov_dim, 1}, {}, {}, num_threads));
    }
};

}


#endif

