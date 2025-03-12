#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SIMPLE_UPDATE_PARAMETER_LIST_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SIMPLE_UPDATE_PARAMETER_LIST_HPP

namespace ttns
{
struct simple_update_parameter_list
{
    simple_update_parameter_list() : krylov_dim(4), nstep(1){}
    simple_update_parameter_list(size_t kdim) : krylov_dim(kdim), nstep(1){}
    simple_update_parameter_list(size_t kdim, size_t ns) : krylov_dim(kdim), nstep(ns){}
    simple_update_parameter_list(const simple_update_parameter_list& o) = default;
    simple_update_parameter_list(simple_update_parameter_list&& o) = default;
    simple_update_parameter_list& operator=(const simple_update_parameter_list& o) = default;
    simple_update_parameter_list& operator=(simple_update_parameter_list&& o) = default;

    size_t krylov_dim;
    size_t nstep;
};
}

#endif

