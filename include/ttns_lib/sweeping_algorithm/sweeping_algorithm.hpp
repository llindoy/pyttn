#ifndef TTNS_SWEEPING_ALGORITHM_ENGINE_HPP
#define TTNS_SWEEPING_ALGORITHM_ENGINE_HPP

#define TIMING

#include <common/timing_macro.hpp>
#include <common/zip.hpp>

#include "../ttn/ttn.hpp"
#include "../ttn/ms_ttn.hpp"

/* Information about sweeping algorithms */
#include "sweeping_forward_decl.hpp"

/* The key updating algorithms */
#include "update/energy_debug_update.hpp"

namespace ttns
{

//the generic single-site sweeping algorithm object.  This provides the core sweeping and updating schemes required for many different strategies for updating ttns.  To actually implement a specific algorithm
//it is necessary to define and provide (as templates) 
//  Update: describing how site tensors and bond tensors are updated given the environment objects
//  Environment: which computes and applies environment tensors to the tensor nodes and 
//  SubspaceExpansion: for potentially expanding the bond dimension when traversing the tree.
//  classes.  By default no subspace expansion is performed.
template <typename T, typename backend, template <typename, typename> class ttn_class = ttn,
            template <typename, typename, template <typename, typename> class, template <typename, typename, template <typename, typename> class> class > class Update = trivial_update, 
            template <typename, typename, template <typename, typename> class> class Environment = empty_environment, 
            template <typename, typename, template <typename, typename> class, template <typename, typename, template <typename, typename> class> class > class SubspaceExpansion = single_site>
class sweeping_algorithm : public Update<T, backend, ttn_class, Environment>, public SubspaceExpansion<T, backend, ttn_class, Environment>
{
public:
    using update_type = Update<T, backend, ttn_class, Environment>;
    using subspace_type = SubspaceExpansion<T, backend, ttn_class, Environment>;
    using environment_type = Environment<T, backend, ttn_class>;

    using update_params = typename update_type::parameter_list;
    using subspace_params = typename subspace_type::parameter_list;
    using environment_params = typename environment_type::parameter_list;
  
    using vec_type = linalg::vector<T, backend>;
    using mat_type = linalg::matrix<T, backend>;

    using env_container_type = typename environment_type::container_type;
    using env_node_type = typename env_container_type::node_type;
    using env_data_type = typename env_container_type::value_type;
    using env_type = typename environment_type::environment_type;
    
    using size_type = typename backend::size_type;
    using real_type = typename tmp::get_real_type<T>::type;

    using ttn_type = ttn_class<T, backend>;
    using hnode = typename ttn_type::node_type;
    using hdata = typename hnode::value_type;
    using bond_matrix_type = typename ttn_type::bond_matrix_type;
    using population_matrix_type = typename ttn_type::population_matrix_type;

    using buffer_type = typename environment_type::buffer_type;

public:
    sweeping_algorithm() {}
    sweeping_algorithm(const ttn_type& A, const env_type& ham, size_type num_threads = 1) 
    {
        m_validate_inputs = true;
        CALL_AND_HANDLE(initialise_default(A, ham, num_threads), "Failed to construct sweeping_algorithm using minimum parameters.");
    }   

    sweeping_algorithm(const ttn_type& A, const env_type& ham, const update_params& upd, const environment_params& env, const subspace_params& sub, size_type num_threads = 1)  
    {
        m_validate_inputs = true;
        CALL_AND_HANDLE(initialise(A, ham, upd, env, sub, num_threads), "Failed to construct sweeping_algorithm.");
    }   

    sweeping_algorithm(const ttn_type& A, const env_type& ham, update_params&& upd, environment_params&& env, subspace_params&& sub, size_type num_threads = 1)  
    {
        m_validate_inputs = true;
        CALL_AND_HANDLE(initialise(A, ham, std::forward<update_params>(upd), std::forward<environment_params>(env), std::forward<subspace_params>(sub), num_threads), "Failed to construct sweeping_algorithm.");
    }   

    sweeping_algorithm(const sweeping_algorithm& o) = default;
    sweeping_algorithm(sweeping_algorithm&& o) = default;

    sweeping_algorithm& operator=(const sweeping_algorithm& o) = default;
    sweeping_algorithm& operator=(sweeping_algorithm&& o) = default;

    void initialise_default(const ttn_type& A, const env_type& ham, size_type num_threads = 1)
    {
        try
        {
            m_env.num_buffers() = num_threads;
            CALL_AND_HANDLE(m_env.initialise(A, ham, m_ham), "Failed to initialise environment object.");
            CALL_AND_HANDLE(subspace_type::initialise(A, ham), "Failed to initialise subspace expansion object.");
            CALL_AND_HANDLE(update_type::initialise(A), "Failed to initialise the update object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise sweeping algorithm object.");
        }
    }

    void initialise(const ttn_type& A, const env_type& ham, const update_params& upd, const environment_params& env, const subspace_params& sub, size_type num_threads = 1)
    {
        try
        {
            m_env.num_buffers() = num_threads;
            CALL_AND_HANDLE(m_env.initialise(A, ham, m_ham, env), "Failed to initialise environment object.");
            CALL_AND_HANDLE(subspace_type::initialise(A, ham, sub), "Failed to initialise subspace expansion object.");
            CALL_AND_HANDLE(update_type::initialise(A, upd), "Failed to initialise the update object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise sweeping algorithm object.");
        }
    }

    void initialise(const ttn_type& A, const env_type& ham, update_params&& upd, environment_params&& env, subspace_params&& sub, size_type num_threads = 1)
    {
        try
        {
            m_env.num_buffers() = num_threads;
            CALL_AND_HANDLE(m_env.initialise(A, ham, m_ham, std::forward<environment_params>(env)), "Failed to initialise environment object.");
            CALL_AND_HANDLE(subspace_type::initialise(A, ham, sub), "Failed to initialise subspace expansion object.");
            CALL_AND_HANDLE(update_type::initialise(A, upd), "Failed to initialise the update object.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to initialise sweeping algorithm object.");
        }
    }

    void clear()
    {
        try
        {
            //clear all of the structures needed for updating
            CALL_AND_HANDLE(update_type::clear(), "Failed to clear update type object.");
            CALL_AND_HANDLE(subspace_type::clear(), "Failed to clear update type object.");
            CALL_AND_HANDLE(m_env.clear(), "Failed to clear update type object.");
            m_nh_evals = 0;
            m_env_set = false;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
        }
    }

public:
    environment_type& environment_engine(){return m_env;}
    const environment_type& environment_engine() const {return m_env;}

public:
    /*
     * Functions for applying and preparing the sweeping algorithm
     */
    bool operator()(ttn_type& A, env_type& op, bool update_environment = false)
    {
        try
        {
            A.setup_orthogonality();

            //if we have a purely time-independent Hamiltonian then we will want to make sure all the buffers are correctly set for the sweeping algorithm
            if( (!A.is_orthogonalised() || update_environment || !m_env_set))
            {
                CALL_AND_HANDLE(prepare_environment(A, op), "Failed to setup environment for evolution");
            }

            return update(A, op,  
                [this](hnode& _A, const environment_type& _env, env_node_type& _h, env_type& _op)
                {
                    CALL_AND_RETHROW(static_cast<update_type*>(this)->update_site_tensor(_A, _env, _h, _op));
                }, 
                [this](bond_matrix_type& _r, const environment_type& _env, env_node_type& _h, env_type& _op)
                {   
                    CALL_AND_RETHROW(static_cast<update_type*>(this)->update_bond_tensor(_r, _env, _h, _op));
                }, 
                [this](hnode& _a1, hnode& _a2, bond_matrix_type& _r, population_matrix_type& _s, env_node_type& _h, env_type& _op, environment_type& _env, std::mt19937& _rng)
                {
                    CALL_AND_RETHROW(return static_cast<subspace_type*>(this)->subspace_expansion_down(_a1, _a2, _r, _s, _h, _op, _env, _rng));
                },
                [this](hnode& _a1, hnode& _a2, bond_matrix_type& _r, population_matrix_type& _s, env_node_type& _h, env_type& _op, environment_type& _env, std::mt19937& _rng)
                {
                    CALL_AND_RETHROW(return static_cast<subspace_type*>(this)->subspace_expansion_up(_a1, _a2, _r, _s, _h, _op, _env, _rng));
                },
                [this](env_type& _op, hnode& _a1, env_node_type& _h)
                {
                    CALL_AND_RETHROW(this->m_env.update_env_down(_op, _a1, _h));
                },
                [this](env_type& _op, hnode& _a1, env_node_type& _h)
                {
                    CALL_AND_RETHROW(this->m_env.update_env_up(_op, _a1, _h));
                }
            );
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("Failed to apply sweeping algorithm.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply sweeping algorithm.");
        }
    }

    /* The function for preparing the environment arrays for an updating step*/
    bool prepare_environment(ttn_type& A, env_type& op, bool attempt_expansion = false)
    { 
        //ASSERT(has_same_structure(A, op), "Incompatible tensor and environment object.");
        using common::zip;   using common::rzip;
        if(!A.is_orthogonalised()){A.orthogonalise();}

        //first iterate through the tree computing the single particle Hamiltonians.  Here we do not attempt to do any bond dimension adaptation
        //as as we are not time-evolving all information about the optimal unoccupied SHFs will be destroyed before it could be used.
        for(auto z : rzip(A, m_ham))
        {
            const auto& a = std::get<0>(z); auto& hspf = std::get<1>(z);
            CALL_AND_HANDLE(m_env.update_env_up(op, a, hspf), "Failed to update the environment tensor.");
        }

        bool subspace_expanded = false;
        
        //we keep attempting the subspace expansion initially until it stops expanding.
        while(attempt_expansion)
        {
            bool subspace_expanded_f = false;
            CALL_AND_HANDLE
            (
                subspace_expanded_f = forward_loop_step(A, op,
                    [](hnode& , const environment_type&, env_node_type&, env_type&){},
                    [](bond_matrix_type& , const environment_type& , env_node_type&, env_type&){},
                    [this](hnode& _a1, hnode& _a2, bond_matrix_type& _r, population_matrix_type& _s, env_node_type& _h, env_type& _op, environment_type& _env, std::mt19937& _rng)
                    {
                        CALL_AND_RETHROW(return static_cast<subspace_type*>(this)->subspace_expansion_down(_a1, _a2, _r, _s, _h, _op, _env, _rng));
                    },
                    [this](env_type& _op, hnode& _a1, env_node_type& _h)
                    {
                        CALL_AND_RETHROW(this->m_env.update_env_down(_op, _a1, _h));
                    },
                    [this](env_type& _op, hnode& _a1, env_node_type& _h)
                    {
                        CALL_AND_RETHROW(this->m_env.update_env_up(_op, _a1, _h));
                    }
                ),
                "Failed to perform backward branch of expansion step."
            );

            //now perform the backwards loop step but without any evolution of nodes.  Here we apply the subspace expansion in order to
            //construct optimal unoccupied SPFs used for the first step of the evolution
            bool subspace_expanded_b = false;
            CALL_AND_HANDLE
            (
                subspace_expanded_b = backward_loop_step(A, op, 
                    [](hnode& , const environment_type&, env_node_type&, env_type&){},
                    [](bond_matrix_type& , const environment_type& , env_node_type&, env_type&){},
                    [this](hnode& _a1, hnode& _a2, bond_matrix_type& _r, population_matrix_type& _s, env_node_type& _h, env_type& _op, environment_type& _env, std::mt19937& _rng)
                    {
                        CALL_AND_RETHROW(return static_cast<subspace_type*>(this)->subspace_expansion_up(_a1, _a2, _r, _s, _h, _op, _env, _rng));
                    },
                    [this](env_type& _op, hnode& _a1, env_node_type& _h)
                    {
                        CALL_AND_RETHROW(this->m_env.update_env_down(_op, _a1, _h));
                    },
                    [this](env_type& _op, hnode& _a1, env_node_type& _h)
                    {
                        CALL_AND_RETHROW(this->m_env.update_env_up(_op, _a1, _h));
                    }
                ),
                "Failed to perform subspace expansion step."
            );

            if(subspace_expanded_f || subspace_expanded_b){subspace_expanded = true;}
            if(!subspace_expanded_f && !subspace_expanded_b){attempt_expansion = false;}
        }
        m_env_set = true;
        A.force_set_orthogonality_centre(0);
        return subspace_expanded;
    }

protected:  
    /* The generic updating function, here we pass in the functions that are called at the various different points of the updating algorithm.  This calls the forward and backward steps*/
    template <typename NodeFunc, typename RFunc, typename SubspaceFuncDown, typename SubspaceFuncUp, typename EnvFuncDown, typename EnvFuncUp>
    bool update(ttn_type& A, env_type& op, NodeFunc&& nf, RFunc&& rf, SubspaceFuncDown&& sfd, SubspaceFuncUp&& sfu, EnvFuncDown&& evd, EnvFuncUp&& evu)
    {
        if(m_validate_inputs)
        {   
            //ASSERT(has_same_structure(A, op), "Incompatible tensor and environment object.");
            ASSERT(A.is_orthogonalised(), "The input hierarchical tucker tensor must be in the orthogonalised form.");
        }

        bool subspace_expanded_f = false;

        //if the operator is time dependent then we need to advance it to the current time point and update the single particle function operators.  
        //For updating schemes that do not have an explicit time dependence this 
        if(op.is_time_dependent())
        {
            CALL_AND_HANDLE(update_type::advance_hamiltonian(A, m_env, m_ham, op), "Failed to update time dependent Hamiltonian.")
        }

        CALL_AND_HANDLE
        (   
            subspace_expanded_f = forward_loop_step(A, op, std::forward<NodeFunc>(nf), std::forward<RFunc>(rf), std::forward<SubspaceFuncDown>(sfd), std::forward<EnvFuncDown>(evd), std::forward<EnvFuncUp>(evu)), 
            "Failed to perform a step of the tdvp_engine object.  Exception raised when performing the forward loop half step."
        );
        CALL_AND_HANDLE(static_cast<update_type*>(this)->advance_half_step(), "Failed to advance the implementation specific objects.");


        //if the operator is time dependent then we need to advance it to the current time point and update the single particle function operators
        if(op.is_time_dependent())
        {
            CALL_AND_HANDLE(update_type::advance_hamiltonian(A, m_env, m_ham, op), "Failed to update time dependent Hamiltonian.");
        }

        //if the forward step failed due to a numerical issue we return that it failed.
        bool subspace_expanded_b = false;
        CALL_AND_HANDLE
        (
            subspace_expanded_b = backward_loop_step(A, op, std::forward<NodeFunc>(nf), std::forward<RFunc>(rf), std::forward<SubspaceFuncUp>(sfu), std::forward<EnvFuncDown>(evd), std::forward<EnvFuncUp>(evu)), 
            "Failed to perform a step of the tdvp_engine object.  Exception raised when performing the backward loop half step."
        );
        CALL_AND_HANDLE(static_cast<update_type*>(this)->advance_half_step(), "Failed to advance the implementation specific objects.");
        A.force_set_orthogonality_centre(0);
        return subspace_expanded_f || subspace_expanded_b;
    }


    /* The forward euler tour update step */
    template <typename NodeFunc, typename RFunc, typename SubspaceFunc, typename EnvFuncDown, typename EnvFuncUp>
    bool forward_loop_step(ttn_type& psi, env_type& op, NodeFunc&& nf, RFunc&& rf, SubspaceFunc&& sf, EnvFuncDown&& evd, EnvFuncUp&& evu)
    {
        bool subspace_expanded = false;
        psi.euler_tour().reset_visits();

        auto& orthog = psi.orthogonality_engine();
        
        //traverse the tree in the order specified by m_traversal
        for(size_type id : psi.euler_tour())
        {
            //define aliases for all of the arguments at the current node
            auto& A = psi[id];  auto& h = m_ham[id];       
            
            size_type times_visited = psi.euler_tour().times_visited(id);
            psi.euler_tour().visit(id);

            //If this isn't our last time visiting we will need to apply a root to leaf node decomposition to 
            //it so that we can propagate factors down the tree structure to its children.
            if(!psi.euler_tour().last_visit(id))
            {
                //get the index of the child we will be performing the decomposition for
                size_type mode = times_visited;

                //if it is our first time visiting the node and we are not at the root node we need to apply the parent nodes root to leaf decomposition
                if(psi.euler_tour().first_visit(id) && !A.is_root())
                {
                    //now we apply the parents r matrix to this node
                    CALL_AND_HANDLE(A.apply_bond_matrix_from_parent(orthog), "Failed to apply bond matrix down tree.");
                }

                //evaluate the root to leaf decomposition provided we aren't at the leaf node and update the mean field hamiltonian
                if(!A.is_leaf())
                {
                    CALL_AND_HANDLE(A.decompose_down(orthog, mode), "Failed to shift orthogonality down.");
                    CALL_AND_HANDLE(A.apply_to_node(orthog), "Failed to enforce orthogonality condition at node.");

                    bool seloc = false;
                    //as we descend the tree apply the subspace expansion
                    CALL_AND_HANDLE(seloc = sf(A[mode], A, psi.active_bond_matrix(), psi.active_population_matrix(), h[mode], op, m_env, psi.rng()), "Subspace expansion Failed.");
                    if(seloc){subspace_expanded = true;}

                    //now we can update the mean field Hamiltonian at the node.  
                    CALL_AND_HANDLE(evd(op, A, h[mode]), "Failed to update the environment tensor.");
                }
            }
            //if it is our final time accessing the node we need to perform the evolution steps
            else
            {
                CALL_AND_HANDLE(nf(A, m_env, h, op), "Failed to update node tensor.");

                //now provided this node is not the root we evaluate the leaf to root decomposition of this tensor
                //time evolve the coefficient tensor and apply it to its parent
                if(!A.is_root())
                {
                    //modify the l2r_core::evaluate routine to include subspace expansion of the child node
                    CALL_AND_HANDLE(A.decompose_up(orthog), "Failed to shift orthogonality down.");
                    CALL_AND_HANDLE(A.apply_to_node(orthog), "Failed to enforce orthogonality condition at node.");
                }

                //swap this out for a general update environment function call

                //now we can update the single particle Hamiltonian at the node
                CALL_AND_HANDLE(evu(op, A, h), "Failed to update the environment tensor.");

                if(!A.is_root())
                {
                    CALL_AND_HANDLE(rf(psi.active_bond_matrix(), m_env, h, op), "Failed to update R coefficient tensor.");
                    CALL_AND_HANDLE(A.apply_bond_matrix_to_parent(orthog), "Failed to apply bond matrix down up.");
                }
            }
        } 
        return subspace_expanded;
    }

    /* The backward euler tour update step */
    template <typename NodeFunc, typename RFunc, typename SubspaceFunc, typename EnvFuncDown, typename EnvFuncUp>
    bool backward_loop_step(ttn_type& psi, env_type& op, NodeFunc&& nf, RFunc&& rf, SubspaceFunc&& sf, EnvFuncDown&& evd, EnvFuncUp&& evu)
    {
        bool subspace_expanded = false;
        psi.euler_tour().reset_visits();

        auto& orthog = psi.orthogonality_engine();
        
        //traverse the tree in the reverse of the order specified by m_traversal
        for(size_type id : reverse(psi.euler_tour()))
        {
            //define aliases for all of the arguments at the current node
            auto& A = psi[id];  auto& h = m_ham[id];       

            size_type times_visited = psi.euler_tour().times_visited(id);
            psi.euler_tour().visit(id);

            //if it is not our last time visiting the node we only need to update the single hole decomposition and mean field Hamiltonian
            if(!psi.euler_tour().last_visit(id))
            {
                //get the index of the child we will be performing the decomposition for
                size_type mode = A.nmodes() - (times_visited+1);

                //now if this is the first time we have accessed this node we firt need to apply its parent's decomposition, 
                //which is first backwards time evolved.  Following which we can time evolve this nodes coefficient matrix.
                if(psi.euler_tour().first_visit(id))
                {
                    //we only have a parent node if we aren't at the root node.
                    if(!A.is_root())
                    {
                        //time evolve the parents r matrix backwards in time through half a time step using this nodes representation of the full Hamiltonian
                        CALL_AND_HANDLE(rf(psi.active_bond_matrix(), m_env, h, op), "Failed to update R coefficient tensor.");

                        //now we apply the parents r matrix to this node
                        CALL_AND_HANDLE(A.apply_bond_matrix_from_parent(orthog), "Failed to apply bond matrix down tree.");
                    }

                    CALL_AND_HANDLE(nf(A, m_env, h, op), "Failed to update node tensor.");
                }

                //now provided this node isn't a leaf node we need to evaluate its root to leaf decomposition so that we can apply this result
                //to its children.  Upon doing so we can now update the mean field operators at this node
                if(!A.is_leaf())
                {
                    CALL_AND_HANDLE(A.decompose_down(orthog, mode), "Failed to shift orthogonality down.");
                    CALL_AND_HANDLE(A.apply_to_node(orthog), "Failed to enforce orthogonality condition at node.");

                    CALL_AND_HANDLE(evd(op, A, h[mode]), "Failed to update the environment tensor.");
                }
            }
            //on the final time accessing we need to apply the leaf to root decomposition to construct the new single particle functions at this node.
            else 
            {
                //in the backwards loop we attempt to expand the bond dimension when moving up the tree
                if(!A.is_root())
                {
                    CALL_AND_HANDLE(A.decompose_up(orthog), "Failed to shift orthogonality down.");
                    CALL_AND_HANDLE(A.apply_to_node(orthog), "Failed to enforce orthogonality condition at node.");

                    bool seloc = false;
                    CALL_AND_HANDLE(seloc = sf(A, A.parent(), psi.active_bond_matrix(), psi.active_population_matrix(), h, op, m_env, psi.rng()), "Subspace expansion failed.");
                    if(seloc){subspace_expanded = true;}

                    CALL_AND_HANDLE(A.apply_bond_matrix_to_parent(orthog), "Failed to apply bond matrix down up.");
                }
                //now we can update the single particle Hamiltonian at the node
                CALL_AND_HANDLE(evu(op, A, h), "Failed to update the environment tensor.");
            }
        }
        return subspace_expanded;
    }

    bool& validate_inputs(){return m_validate_inputs;}
    const bool& validate_inputs() const {return m_validate_inputs;}

public:
    size_type nh_applications() const{return m_nh_evals;}

protected:
    environment_type m_env;
    size_type m_nh_evals = 0;
    bool m_env_set = false;

    //an object storing the traversal order required for evaluating the root to leaf decomposition
    size_type m_num_threads = 1;
    bool m_validate_inputs = true;

  
    env_container_type m_ham;
};  //class sweeping_algorithm

template <typename T, typename backend>
using one_site_debug = sweeping_algorithm<T, backend, ttn, energy_debug_engine, sop_environment, single_site>;

}   //namespace ttns

#endif  //TTNS_TDVP_ALGORITHM_ENGINE_HPP//

