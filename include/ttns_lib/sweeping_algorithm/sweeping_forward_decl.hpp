#ifndef TTNS_LIB_SWEEPING_ALGORITHM_SWEEPING_FORWARD_DECL_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_SWEEPING_FORWARD_DECL_HPP

namespace ttns
{

/* 
 *  Forward declaration of the different environment types that are implemented
 */
template <typename T, typename backend, template <typename, typename> class ttn_class> 
class empty_environment
{
public:
    using size_type = typename backend::size_type;
    using ttn_type = ttn_class<T, backend>;

    struct environment_type{};

    using hnode = typename ttn_type::node_type;
    using mat_type = linalg::matrix<T, backend>;

    using container_type = tree<bool>;
    using node_type = typename container_type::node_type;
    using data_type = typename container_type::value_type;

    struct buffer_type
    {
        using size_type = typename backend::size_type;
        void reallocate(size_t, size_t){}
        void resize(size_type, size_type){}
        void clear(){}
    };

    empty_environment(){}
    empty_environment(const empty_environment& o) = default;
    empty_environment(empty_environment&& o) = default;

    empty_environment& operator=(const empty_environment& o) = default;
    empty_environment& operator=(empty_environment&& o) = default;

    struct parameter_list{};
public:
    void initialise(const ttn_type& A, const environment_type&, container_type& ham)
    {
        ham.construct_topology(A);
    }
    static inline void initialise(const ttn_type& A, const environment_type& h, container_type& ham, const parameter_list&)
    {
        CALL_AND_RETHROW(initialise(A, h, ham));
    }
    static inline void initialise(const ttn_type& A, const environment_type& h, container_type& ham, parameter_list&&)
    {
        CALL_AND_RETHROW(initialise(A, h, ham));
    }
    static inline size_type maximum_dimension(const ttn_type&){return 0;}

    inline void update_env_down(const environment_type&, const hnode&, node_type&){}
    inline void update_env_down(const environment_type&, const hnode&, node_type&, node_type&){}
    inline void update_env_up(const environment_type&, const hnode&, node_type&, bool = false){}
    inline void update_env_up(const environment_type&, const hnode&, node_type&, node_type&, bool = false){}

    const size_type& num_buffers() const{return m_num_buffers;}
    size_type& num_buffers(){return m_num_buffers;}

protected:
    size_type m_num_buffers = 1;
    container_type m_container;
};

template <typename T, typename backend, template <typename, typename> class ttn_class> class sop_environment;


/* 
 *  Forward declaration of the different subspace expansion types that have been implemented
 */
template <typename T, typename backend, template <typename, typename> class ttn_class, template <typename, typename, template <typename, typename> class > class Environment>
class single_site 
{
public:
    using size_type = typename backend::size_type;
    using environment_type = Environment<T, backend, ttn_class>;
    using ttn_type = ttn_class<T, backend>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_type = typename environment_type::environment_type;

    using hnode = typename ttn_type::node_type;
    using mat_type = linalg::matrix<T, backend>;
    using bond_matrix_type = typename ttn_type::bond_matrix_type;
    using population_matrix_type = typename ttn_type::population_matrix_type;

    using buffer_type = typename environment_type::buffer_type;

    struct parameter_list{};
public:
    single_site(){}
    single_site(const single_site& o) = default;
    single_site(single_site&& o) = default;

    single_site& operator=(const single_site& o) = default;
    single_site& operator=(single_site&& o) = default;
    
    void initialise(const ttn_type&, const env_type&){}
    static inline void initialise(const ttn_type&, const env_type&, const parameter_list&){}
    static inline void initialise(const ttn_type&, const env_type&, parameter_list&&){}
    void clear(){}


    //perform the subspace expansion as we are moving down a tree.  This requires us to evaluate the optimal functions to add 
    //into A2.  For A1 they will be overwriten by the r matrix in the next step so we just 
    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_down(hnode& /* A1 */, hnode& /* A2 */, const bond_matrix_type& /* r */, const population_matrix_type& /* s */, env_node_type& /* h */, const env_type& /* op */, environment_type& /* env */, std::mt19937& /* rng */)
    {
        return false;
    }

    //here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
    bool subspace_expansion_up(hnode& /* A1 */, hnode& /* A2 */, const bond_matrix_type& /* r */, const population_matrix_type& /* s */, env_node_type& /* h */, const env_type& /* op */, environment_type& /* env */, std::mt19937& /* rng */)
    {
        return false;
    }

public:
    const size_type& Nonesite() const{return 0;}
    const size_type& Ntwosite() const{return 0;}

};  //class single_site

template <typename T, typename backend, template <typename, typename> class ttn_class, template <typename, typename, template <typename, typename> class > class Environment>
class variance_subspace_expansion;

template <typename T, typename backend, template <typename, typename> class ttn_class, template <typename, typename, template <typename, typename> class > class Environment>
class variance_subspace_expansion_full_two_site;

/* 
 *  Forward declaration of the different update types that have been implemented
 */
template <typename T, typename backend, template <typename, typename> class ttn_class, template <typename, typename, template <typename, typename> class> class Environment> 
class trivial_update
{
public:
    using size_type = typename backend::size_type;
    using environment_type = Environment<T, backend, ttn_class>;
    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_data_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;
    using ttn_type = ttn_class<T, backend>;
    using bond_matrix_type = typename ttn_type::bond_matrix_type;

    using hnode = typename ttn_type::node_type;
    using mat_type = linalg::matrix<T, backend>;

    using buffer_type = typename environment_type::buffer_type;
    struct parameter_list{};
public:
    trivial_update() {}
    trivial_update(const trivial_update& o) = default;
    trivial_update(trivial_update&& o) = default;

    trivial_update& operator=(const trivial_update& o) = default;
    trivial_update& operator=(trivial_update&& o) = default;

    void initialise(const ttn_type&){}
    static inline void initialise(const ttn_type&,  const parameter_list&){}
    static inline void initialise(const ttn_type&, parameter_list&&){}
    void clear(){}
    T E() const{return T(0);}
    size_type update_site_tensor(hnode& /* A */, const environment_type& /* env */, env_node_type& /* h */, const env_type& /* op */){return 0;}
    void update_bond_tensor(bond_matrix_type& /* r */, const environment_type& /* env */, env_node_type& /* h */, const env_type& /* op */){}
    void advance_hamiltonian(ttn_type&, environment_type&, env_container_type& , env_type& ){}
    void advance_half_step(){}

};

template <typename T, typename backend, template <typename, typename > class ttn_class, template <typename, typename, template <typename, typename> class > class Environment> class gso_engine;
template <typename T, typename backend, template <typename, typename > class ttn_class, template <typename, typename, template <typename, typename> class > class Environment> class tdvp_engine;
template <typename T, typename backend, template <typename, typename > class ttn_class, template <typename, typename, template <typename, typename> class > class Environment> class energy_debug_engine;
template <typename T, typename backend, template <typename, typename > class ttn_class, template <typename, typename, template <typename, typename> class > class Environment> class assignment_engine;

}

#endif

