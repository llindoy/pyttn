#ifndef TTNS_LIB_SINGLE_PARTICLE_OPERATOR_HPP
#define TTNS_LIB_SINGLE_PARTICLE_OPERATOR_HPP

#include <linalg/linalg.hpp>
#include "../ttn/ttn.hpp"
#include "../ttn/ms_ttn.hpp"
#include "../operators/sop_operator.hpp"
#include "../operators/multiset_sop_operator.hpp"
#include "observable_node.hpp"
#include "multiset_sop_env_node.hpp"
#include "kronecker_product_operator_helper.hpp"

namespace ttns
{

//TODO: Clean this up it is currently nowhere near as nicely 
//    : Change how the matrices are being resized to make it a bit cleaner
template <typename T, typename backend>
class single_particle_operator_engine
{
    using kpo = kronecker_product_operator_mel<T, backend>;
    using hnode = ttn_node<T, backend>;
    using hdata = ttn_node_data<T, backend>;

    using soptype = sop_operator<T, backend>;
    using cinfnode = typename soptype::node_type;

    using ms_hnode = ms_ttn_node<T, backend>;
    using ms_hdata = multiset_node_data<T, backend>;

    using ms_soptype = multiset_sop_operator<T, backend>;
    using ms_cinfnode = typename ms_soptype::node_type;

    using op_container = typename soptype::container_type;

    using mat = linalg::matrix<T, backend>;
    using triad = std::vector<mat>;
    using cinftype = sttn_node_data<T>;

    using size_type = typename backend::size_type;

public:
    template <typename spfnode>
    static inline void evaluate(const soptype& h, const cinfnode& cinf, const hnode& A, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        //resize the matrices in the event that the tensor objects have changed size
        if(A.is_leaf()){CALL_AND_RETHROW(evaluate_leaf(h, cinf, A, A, HA, hspf, compute_identity, update_all));}
        else{CALL_AND_RETHROW(evaluate_branch(cinf, A, A, HA, temp, hspf, compute_identity, update_all));}
    }

    template <typename spfnode>
    static inline void evaluate(const soptype& h, const cinfnode& cinf, const hnode& B, const hnode& A, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        //resize the matrices in the event that the tensor objects have changed size
        if(A.is_leaf()){CALL_AND_RETHROW(evaluate_leaf(h, cinf, B, A, HA, hspf, compute_identity, update_all));}
        else{CALL_AND_RETHROW(evaluate_branch(cinf, B, A, HA, temp, hspf, compute_identity, update_all));}
    }

    template <typename spfnode>
    static inline void evaluate(const soptype& h, const cinfnode& cinf, const hnode& A, size_t /* i */, size_t /* c */, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        CALL_AND_RETHROW(evaluate(h, cinf, A, A, hspf, HA, temp, compute_identity, update_all));
    }

    template <typename spfnode>
    static inline void evaluate(const soptype& h, const cinfnode& cinf, const hnode& B, const hnode& A, size_t /* i */, size_t /* c */, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        CALL_AND_RETHROW(evaluate(h, cinf, B, A, hspf, HA, temp, compute_identity, update_all));
    }


public:
    template <typename spfnode>
    static inline void evaluate(const ms_soptype& h, const ms_cinfnode& cinf, const ms_hnode& A, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        //resize the matrices in the event that the tensor objects have changed size
        if(A.is_leaf()){CALL_AND_RETHROW(evaluate_leaf(h, cinf, A, A, HA, hspf, compute_identity, update_all));}
        else{CALL_AND_RETHROW(evaluate_branch(cinf, A, A, HA, temp, hspf, compute_identity, update_all));}
    }

    template <typename spfnode>
    static inline void evaluate(const ms_soptype& h, const ms_cinfnode& cinf, const ms_hnode& B, const ms_hnode& A, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        //resize the matrices in the event that the tensor objects have changed size
        if(A.is_leaf()){CALL_AND_RETHROW(evaluate_leaf(h, cinf, B, A, HA, hspf, compute_identity, update_all));}
        else{CALL_AND_RETHROW(evaluate_branch(cinf, B, A, HA, temp, hspf, compute_identity, update_all));}
    }

    template <typename spfnode>
    static inline void evaluate(const ms_soptype& h, const ms_cinfnode& cinf, const ms_hnode& A, size_t i, size_t c, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        //resize the matrices in the event that the tensor objects have changed size
        if(A.is_leaf()){CALL_AND_RETHROW(evaluate_leaf(h, cinf, A, A, i, c, HA, hspf, compute_identity, update_all));}
        else{CALL_AND_RETHROW(evaluate_branch(cinf, A, A, i, c, HA, temp, hspf, compute_identity, update_all));}
    }

    template <typename spfnode>
    static inline void evaluate(const ms_soptype& h, const ms_cinfnode& cinf, const ms_hnode& B, const ms_hnode& A, size_t i, size_t c, spfnode& hspf, triad& HA, triad& temp, bool compute_identity = false, bool update_all = true)
    {
        //resize the matrices in the event that the tensor objects have changed size
        if(A.is_leaf()){CALL_AND_RETHROW(evaluate_leaf(h, cinf, B, A, i, c, HA, hspf, compute_identity, update_all));}
        else{CALL_AND_RETHROW(evaluate_branch(cinf, B, A, i, c, HA, temp, hspf, compute_identity, update_all));}
    }

protected:
    template <typename spftype>
    static inline void evaluate_leaf(const op_container& h, const cinftype& cinf, const hdata& B, const hdata& A, triad& HA, spftype& hspf, bool compute_identity = false, bool update_all = true)
    {
        try
        {
            CALL_AND_HANDLE(hspf.resize_matrices(B.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");
            const auto& a = A.as_matrix();  
            const auto& b = B.as_matrix();  

            //if the A matrix and B matrix are not the same then we simply go through and compute the identity matrix
            if( (&A != &B || compute_identity) && update_all)
            {
                CALL_AND_HANDLE(hspf.spf_id() = adjoint(b)*a, "Failed to compute the id matrix term.");
            }

#ifdef USE_OPENMP
#ifdef PARALLELISE_HAMILTONIAN_SUM
            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
#endif
#endif
            for(size_type ind = 0; ind < cinf.nterms(); ++ind)
            {
                size_type ti = omp_get_thread_num();
                CALL_AND_HANDLE(HA[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                //update all terms if we aren't worrying about time dependence otherwise only deal with time dependence
                if( update_all ||  cinf[ind].is_time_dependent())
                {
                    if(!cinf[ind].is_identity_spf())
                    {
                        hspf.spf(ind).fill_zeros();
                        for(size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                        {
                            auto& indices = cinf[ind].spf_indexing()[i][0];
                            CALL_AND_HANDLE(h[indices[0]][indices[1]].apply(a, HA[ti]), "Failed to apply leaf operator.");
                            CALL_AND_HANDLE(hspf.spf(ind) += cinf[ind].spf_coeff(i)*adjoint(b)*HA[ti], "Failed to apply matrix product to obtain result.");
                        }
                    }
                }
            }
        }        
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating single particle operator at a leaf node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate single particle operator at a leaf node.");
        }
    }

    //static inline void resize_buffer(const hnode& A, triad& HA, 

    template <typename spfnode>
    static inline void evaluate_leaf(const soptype& h, const cinfnode& cinf, const hnode& B, const hnode& A, triad& HA, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
        CALL_AND_RETHROW(evaluate_leaf(h.mode_operators(), cinf(), B(), A(), HA, hspf(), compute_identity, update_all));
    }

    template <typename spfnode>
    static inline void evaluate_leaf(const ms_soptype& h, const ms_cinfnode& cinf, const ms_hnode& B, const ms_hnode& A, size_t i, size_t c, triad& HA, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
        //now we need to iterate over each of the set indices and perform the correct contractions storing them in the correct locations
        size_t j = cinf()[i][c].col();
        CALL_AND_RETHROW(evaluate_leaf(h.mode_operators()[i][c], cinf()[i][c], B(i), A(j), HA, hspf(), compute_identity, update_all));
    }

    template <typename spfnode>
    static inline void evaluate_leaf(const ms_soptype& h, const ms_cinfnode& cinf, const ms_hnode& B, const ms_hnode& A, triad& HA, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
#ifdef USE_OPENMP 
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for default(shared) if(HA.size() > 1 && cinf().size() > 1) num_threads(HA.size())
#endif
#endif
        for(size_t row = 0; row < cinf().size(); ++row)
        {
            for(size_t ci = 0; ci < cinf()[row].size(); ++ci)
            {
                size_t col = cinf()[row][ci].col();
                CALL_AND_RETHROW(evaluate_leaf(h.mode_operators()[row][ci], cinf()[row][ci], B(row), A(col), HA, hspf()[row][ci], compute_identity, update_all));
            }
        }
    }

protected:
    template <typename spfnode>
    static inline void _evaluate_branch(const cinftype& cinf, const hdata& A, triad& HA, triad& temp, spfnode& hspf, bool update_all = true)
    {
        try
        {
            CALL_AND_HANDLE(hspf().resize_matrices(A.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");
            const auto& a = A.as_matrix();  
#ifdef USE_OPENMP
#ifdef PARALLELISE_HAMILTONIAN_SUM
            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
#endif
#endif
            for(size_type ind = 0; ind < cinf.nterms(); ++ind)
            {
                size_type ti = omp_get_thread_num();
                CALL_AND_HANDLE(HA[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                CALL_AND_HANDLE(temp[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                if( update_all || cinf[ind].is_time_dependent())
                {
                    if(!cinf[ind].is_identity_spf())
                    {
                        hspf().spf(ind).fill_zeros();
                        for(size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                        {
                            CALL_AND_HANDLE(kron_prod(hspf, cinf, ind, i, A, temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                            CALL_AND_HANDLE(hspf().spf(ind) += cinf[ind].spf_coeff(i)*adjoint(a)*HA[ti], "Failed to apply matrix product to obtain result.");
                        }
                    }
                }
            }
        }        
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating single particle operator at a branch node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate single particle operator at a branch node.");
        }
    }

    template <typename spfnode>
    static inline void _evaluate_branch(const cinftype& cinf, const hdata& B, const hdata& A, triad& HA, triad& temp, spfnode& hspf, bool update_all = true)
    {
        try
        {
            size_type ti = omp_get_thread_num();
            CALL_AND_HANDLE(HA[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
            CALL_AND_HANDLE(temp[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
            CALL_AND_HANDLE(hspf().resize_matrices(B.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");

            const auto& b = B.as_matrix();  
            if(update_all)
            {
                CALL_AND_HANDLE(kpo::kpo_id(hspf, A, temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
                CALL_AND_HANDLE(hspf().spf_id() = adjoint(b)*HA[ti], "Failed to compute the id matrix term.");
            }

#ifdef USE_OPENMP
#ifdef PARALLELISE_HAMILTONIAN_SUM
            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
#endif
#endif
            for(size_type ind = 0; ind < cinf.nterms(); ++ind)
            {
                if( update_all || cinf[ind].is_time_dependent())
                {
                    size_type ti2 = omp_get_thread_num();
                    CALL_AND_HANDLE(HA[ti2].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                    CALL_AND_HANDLE(temp[ti2].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                    if(!cinf[ind].is_identity_spf())
                    {
                        hspf().spf(ind).fill_zeros();
                        for(size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                        {
                            CALL_AND_HANDLE(kron_prod(hspf, cinf, ind, i, B, A, temp[ti2], HA[ti2]), "Failed to apply kronecker product operator.");
                            CALL_AND_HANDLE(hspf().spf(ind) += cinf[ind].spf_coeff(i)*adjoint(b)*HA[ti2], "Failed to apply matrix product to obtain result.");
                        }
                    }
                }
            }
        }        
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("evaluating single particle operator at a branch node.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate single particle operator at a branch node.");
        }
    }

    template <typename spfnode>
    static inline void evaluate_branch(const cinftype& cinf, const hdata& B, const hdata& A, triad& HA, triad& temp, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
        if(&A == &B && !compute_identity)
        {
            CALL_AND_RETHROW(_evaluate_branch(cinf, A, HA, temp, hspf, update_all));
        }
        else
        {
            CALL_AND_RETHROW(_evaluate_branch(cinf, B, A, HA, temp, hspf, update_all));
        }
    }
protected:
    template <typename spfnode>
    static inline void evaluate_branch(const cinfnode& cinf, const hnode& B, const hnode& A, triad& HA, triad& temp, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
        CALL_AND_RETHROW(evaluate_branch(cinf(), B(), A(), HA, temp, hspf, compute_identity, update_all));
    }

    template <typename spfnode>
    static inline void evaluate_branch(const ms_cinfnode& cinf, const ms_hnode& B, const ms_hnode& A, size_t i, size_t c, triad& HA, triad& temp, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
        size_t j = cinf()[i][c].col();
        CALL_AND_RETHROW(evaluate_branch(cinf()[i][c], B(i), A(j), HA, temp, hspf, compute_identity, update_all));
    }

    template <typename spfnode>
    static inline void evaluate_branch(const ms_cinfnode& cinf, const ms_hnode& B, const ms_hnode& A, triad& HA, triad& temp, spfnode& hspf, bool compute_identity = false, bool update_all = true)
    {
#ifdef USE_OPENMP 
#ifdef PARALLELISE_SET_VARIABLES
        #pragma omp parallel for default(shared) if(temp.size() > 1 && cinf().size() > 1) num_threads(temp.size())
#endif
#endif
        for(size_t row = 0; row < cinf().size(); ++row)
        {
            for(size_t ci = 0; ci < cinf()[row].size(); ++ci)
            {
                size_t col = cinf()[row][ci].col();
                ms_sop_env_slice<T, backend> hslice(hspf, row, ci);
                CALL_AND_RETHROW(evaluate_branch(cinf()[row][ci], B(row), A(col), HA, temp, hslice, compute_identity, update_all));
            }
        }
    }

public:
    //kronecker product operators for the operator ype
    template <typename spfnode>
    static void kron_prod(const spfnode& op, const cinftype& cinf, size_type ind, size_type ri, const hdata& A, mat& temp, mat& res)
    {
        CALL_AND_RETHROW(kpo::kron_prod([&op](size_t nu, size_t cri){return op[nu]().spf(cri);}, cinf[ind].spf_indexing()[ri], A, temp, res));
    }

    //kronecker product operators for the operator type
    template <typename spfnode>
    static void kron_prod(const spfnode& op, const cinftype& cinf, size_type ind, size_type ri, const hdata& B, const hdata& A, mat& temp, mat& res)
    {
        ASSERT(op().has_identity(), "Cannot apply rectangular hamiltonian without having identity matrices bound");
        CALL_AND_RETHROW(kpo::kron_prod([&op](size_t nu, size_t cri){return op[nu]().spf(cri);}, [&op](size_t nu){return op[nu]().spf_id();}, cinf[ind].spf_indexing()[ri], B, A, temp, res));
    }

};  //class single_particle_operator engine
}   //namespace ttns

#endif  //TTNS_LIB_SINGLE_PARTICLE_OPERAT[ind]OR_HPP//

