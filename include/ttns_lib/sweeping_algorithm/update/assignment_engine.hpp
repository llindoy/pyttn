#ifndef TTNS_LIB_SWEEPING_ALGORITHM_ASSIGNMENT_ENGINE_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_ASSIGNMENT_ENGINE_HPP

#include <omp.hpp>

#include <iterative_linear_algebra/arnoldi.hpp>
#include "simple_update_parameter_list.hpp"

namespace ttns
{


template <typename T, typename backend, template <typename, typename> class ttn_class, template <typename, typename, template <typename, typename> class> class Environment>
class assigment_engine<T, backend, Environment>
{
public:
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;
    using environment_type = Environment<T, backend, ttn_class>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_data_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;

    using ttn_type = ttn_class<T, backend>;
    using hnode = typename ttn_type::node_type;;

    using mat_type = linalg::matrix<T, backend>;
    using bond_matrix_type = typename ttn_type::bond_matrix_type;

    using matnode = typename tree<mat_type>::node_type;

    using buffer_type = typename environment_type::buffer_type;
    using eigensolver_type = utils::arnoldi<T, backend>;

    struct parameter_list{};
public:
    assignment_engine() :  m_curr_E(0) {}
    assignment_engine(const ttn_type& A) : m_curr_E(0)
    {
        CALL_AND_HANDLE(initialise(A), "Failed to construct assignment_engine.");
    }   
    assignment_engine(const assignment_engine& o) = default;
    assignment_engine(assignment_engine&& o) = default;

    assignment_engine& operator=(const assignment_engine& o) = default;
    assignment_engine& operator=(assignment_engine&& o) = default;
    
    void initialise(const ttn_type& A)
    {
        try
        {
            CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
        }
    }

    void initialise(const ttn_type& A, const parameter_list& o){CALL_AND_RETHROW(initialise(A));}
    void initialise(const ttn_type& A, parameter_list&& o){CALL_AND_RETHROW(initialise(A));}

    void clear(){}

    void advance_half_step(){}
    T E() const{return 0;}

    size_type update_site_tensor(hnode& A, const environment_type& env, env_node_type& h, env_type& op)
    {                    
      
    }

    void update_bond_tensor(bond_matrix_type& /* r */, const environment_type& /* env */, env_node_type& /* h */, env_type& /* op */){}

protected:
};  //class assignment_engine

}

#endif

