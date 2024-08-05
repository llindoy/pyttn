#ifndef TTNS_LIB_SWEEPING_ALGORITHM_MEAN_FIELD_OPERATOR_CORE_REFACTOR_HPP
#define TTNS_LIB_SWEEPING_ALGORITHM_MEAN_FIELD_OPERATOR_CORE_REFACTOR_HPP

#include "sop_env_node.hpp"
#include "kronecker_product_operator_helper.hpp"

namespace ttns
{

template <typename T, typename backend>
class mean_field_operator_engine
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

protected:
    static inline size_type contraction_buffer_size(const hdata& A, bool use_capacity = false)
    {
        size_type maxdim = 0;
        for(size_type mode = 0; mode < A().nmodes(); ++mode)
        {
            auto _A = A().as_rank_3(mode, use_capacity);
            size_type dim = _A.shape(0)*_A.shape(1)*_A.shape(1);
            if(dim > maxdim){maxdim = dim;}
        }
        return maxdim;
    }

public:
    static inline size_type contraction_buffer_size(const hnode& A, bool use_capacity = false)
    {
        return contraciton_buffer_size(A(), use_capacity);
    }

    static inline size_type contraction_buffer_size(const ms_hnode& A, bool use_capacity = false)
    {
        size_type maxdim = 0;
        for(size_t i = 0; i < A.nset(); ++i)
        {
            size_type dim = contraction_buffer_size(A(i), use_capacity);
            if(dim > maxdim){maxdim = dim;}
        }
        return maxdim;
    }


public:
    template <typename opnode>
    static inline void evaluate(const cinfnode& hinf, const hnode& A, triad& HA, triad& temp, opnode& h)
    {
        if(!h.is_root())
        {
            size_type mode = h.child_id();
            const auto& hinf_p = hinf.parent();
            CALL_AND_RETHROW(_evaluate_term(hinf(), hinf_p(), mode, 0, A(), HA, temp, h));
        }
    }

    template <typename opnode>
    static inline void evaluate(const ms_cinfnode& hinf, const ms_hnode& A, triad& HA, triad& temp, opnode& h)
    {
        if(!h.is_root())
        {
            //std::cerr << "node index: " << h.id() << " " << (h.is_leaf() ? "leaf" : "not leaf") << " " << (h.parent().is_root() ? "parent root" : "not root") << std::endl;
            size_type mode = h.child_id();
            const auto& hinf_p = hinf.parent();

            for(size_t row = 0; row < hinf().size(); ++row)
            {
                for(size_t ci = 0; ci < hinf()[row].size(); ++ci)
                {
                    size_t col = hinf()[row][ci].col();
                    //std::cerr << "row col" << row << " " << col << std::endl;
                    ms_sop_env_slice<T, backend> hslice(h, row, ci);
                    size_type ti = omp_get_thread_num();
                    if(row == col)
                    {
                        _evaluate_term(hinf()[row][ci], hinf_p()[row][ci], mode, ti, A(row), HA, temp, hslice);
                    }
                    else
                    {
                        _evaluate_term(hinf()[row][ci], hinf_p()[row][ci], mode, ti, A(row), A(col), HA, temp, hslice);
                    }
                }
            }
        }
    }


protected:
    template <typename opnode>
    static inline void _evaluate_term(const cinftype& hinf, const cinftype& hinf_p, size_type mode, size_type ti, const hdata& A, triad& HA, triad& temp, opnode& h)
    {
        try
        {
            CALL_AND_HANDLE(HA[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");
            CALL_AND_HANDLE(temp[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");

            const auto& h_p = h.parent();

            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
            for(size_type ind = 0; ind < hinf.nterms(); ++ind)
            {
                //if the mean field operator is the identity then we don't need to do anything.
                if(!hinf[ind].is_identity_mf())
                {
                    h().mf(ind).fill_zeros();
                    for(size_type it = 0; it < hinf[ind].nmf_terms(); ++it)
                    {
                        size_type pi = hinf[ind].mf_indexing()[it].parent_index();

                        if(!hinf_p[pi].is_identity_mf())
                        {
                            CALL_AND_HANDLE(kron_prod(h, hinf, ind, it, A, HA[ti], temp[ti]), "Failed to evaluate action of kronecker product operator.");
                            CALL_AND_HANDLE(HA[ti] = temp[ti]*trans(h_p().mf(pi)), "Failed to apply action of parent mean field operator.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(kron_prod(h, hinf, ind, it, A, temp[ti], HA[ti]), "Failed to evaluate action of kronecker product operator.");
                        }

                        CALL_AND_HANDLE(temp[ti] = conj(A.as_matrix()), "Failed to compute conjugate of the A matrix.");

                        try
                        {
                            auto _A = A.as_rank_3(mode);
                            auto _HA = HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                            auto _temp = temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                            CALL_AND_HANDLE(h().mf(ind) += hinf[ind].mf_coeff(it)*(contract(_temp, 0, 2, _HA, 0, 2)), "Failed when evaluating the final contraction.");
                        }                                         
                        catch(const std::exception& ex)
                        {
                            std::cerr << ex.what() << std::endl;
                            RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                        }                       
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate mean field operator at a node.");
        }
    }

    //THE OFF DIAGONAL TERMS CURRENTLY SEEM TO BE INCORRECT. It seems that the values are correct but the sign is wrong.
    template <typename opnode>
    static inline void _evaluate_term(const cinftype& hinf, const cinftype& hinf_p, size_type mode, size_type ti, const hdata& B, const hdata& A, triad& HA, triad& temp, opnode& h)
    {
        if(&A == &B){CALL_AND_RETHROW(return _evaluate_term(hinf, hinf_p, mode, ti, A, HA, temp, h));}
        try
        {
            CALL_AND_HANDLE(HA[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");
            CALL_AND_HANDLE(temp[ti].resize(B.size(0), A.size(1)), "failed to resize working buffers.");

            const auto& h_p = h.parent();
            {
                //std::cerr << "evaluating id pre" << std::endl;
                CALL_AND_HANDLE(kpo::kpo_id(h_p, A, mode, HA[ti], temp[ti]), "Failed to apply kronecker product operator.");
                //std::cerr << "evaluating id post" << std::endl;
                CALL_AND_HANDLE(HA[ti] = temp[ti]*trans(h_p().mf_id()), "Failed to apply action of parent mean field operator.");
                CALL_AND_HANDLE(temp[ti] = conj(B.as_matrix()), "Failed to compute conjugate of the A matrix.");

                try
                {
                    auto _A = A.as_rank_3(mode);
                    auto _B = B.as_rank_3(mode);
                    auto _HA = HA[ti].reinterpret_shape(_B.shape(0), _A.shape(1), _B.shape(2));
                    auto _temp = temp[ti].reinterpret_shape(_B.shape(0), _B.shape(1), _B.shape(2));
  
                    CALL_AND_HANDLE(h().mf_id() = (contract(_temp, 0, 2, _HA, 0, 2)), "Failed when evaluating the final contraction.");
                }                                         
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                }   
            }

            //#pragma omp parallel for default(shared) schedule(dynamic, 1)
            for(size_type ind = 0; ind < hinf.nterms(); ++ind)
            {
                //if the mean field operator is the identity then we don't need to do anything.
                if(!hinf[ind].is_identity_mf())
                {
                    h().mf(ind).fill_zeros();
                    for(size_type it = 0; it < hinf[ind].nmf_terms(); ++it)
                    {
                        size_type pi = hinf[ind].mf_indexing()[it].parent_index();
                        
                        if(hinf[ind].mf_indexing()[it].sibling_indices().size() == 0)
                        {
                            //std::cerr << "idop before" << std::endl;
                            CALL_AND_HANDLE(kpo::kpo_id(h_p, A, mode, HA[ti], temp[ti]), "Failed to apply kronecker product operator.");
                            //std::cerr << "idop" << std::endl;
                        }
                        else
                        {
                            //std::cerr << "standard before" << mode << std::endl;
                            CALL_AND_HANDLE(kron_prod(h, hinf, ind, it, B, A, mode, HA[ti], temp[ti]), "Failed to evaluate action of kronecker product operator.");
                            //std::cerr << "standard" << mode << std::endl;
                        }
                        if(!hinf_p[pi].is_identity_mf())
                        {
                            CALL_AND_HANDLE(HA[ti] = temp[ti]*trans(h_p().mf(pi)), "Failed to apply action of parent mean field operator.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(HA[ti] = temp[ti]*trans(h_p().mf_id()), "Failed to apply action of parent mean field operator.");
                        }

                        CALL_AND_HANDLE(temp[ti] = conj(B.as_matrix()), "Failed to compute conjugate of the A matrix.");

                        try
                        {
                            auto _A = A.as_rank_3(mode);
                            auto _B = B.as_rank_3(mode);

                            //std::cerr << "A" << _A.size(0) << " " << _A.size(1) <<  " " << _A.size(2) << std::endl;
                            //std::cerr << "B" << _B.size(0) << " " << _B.size(1) <<  " " << _B.size(2) << std::endl;
                            //std::cerr << "HA" << HA[ti].size(0) << " " << HA[ti].size(1) << std::endl;
                            //std::cerr << "temp" << temp[ti].size(0) << " " << temp[ti].size(1) << std::endl;
                            auto _HA = HA[ti].reinterpret_shape(_B.shape(0), _A.shape(1), _B.shape(2));
                            auto _temp = temp[ti].reinterpret_shape(_B.shape(0), _B.shape(1), _B.shape(2));

                            CALL_AND_HANDLE(h().mf(ind) += hinf[ind].mf_coeff(it)*(contract(_temp, 0, 2, _HA, 0, 2)), "Failed when evaluating the final contraction.");
                        }                                         
                        catch(const std::exception& ex)
                        {
                            std::cerr << ex.what() << std::endl;
                            RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                        }                       
                    }
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate mean field operator at a node.");
        }
    }

    template <typename opnode>
    static inline void evaluate_term(const cinftype& hinf, const cinftype& hinf_p, size_type mode, size_type ti, const hdata& B, const hdata& A, triad& HA, triad& temp, opnode& h)
    {
        if(&A == &B)
        {
            CALL_AND_RETHROW(_evaluate_term(hinf, hinf_p, mode, ti, A, HA, temp, h));
        }
        else
        {
            CALL_AND_RETHROW(_evaluate_term(hinf, hinf_p, mode, ti, B, A, HA, temp, h));
        }
    }
public:     
    template <typename opnode>
    static void kron_prod( const opnode& op, const cinftype& cinf, size_type ind, size_type ri, const hdata& A, mat& temp, mat& res)
    {
        CALL_AND_RETHROW(kpo::kron_prod([&op](size_t nu, size_t cri){return op.parent()[nu]().spf(cri);}, cinf[ind].mf_indexing()[ri].sibling_indices(), A, temp, res));
    }

    //kronecker product operators for the operator type
    template <typename spfnode>
    static void kron_prod(const spfnode& op, const cinftype& cinf, size_type ind, size_type ri, const hdata& B, const hdata& A, size_type mode, mat& temp, mat& res)
    {
        ASSERT(op().has_identity(), "Cannot apply rectangular hamiltonian without having identity matrices bound");
        ASSERT(cinf[ind].mf_indexing()[ri].sibling_indices().size() != 0, "Cannot apply kron prod if all spf matrices are identity.");
        CALL_AND_RETHROW
        (
            kpo::kron_prod(
              [&op](size_t nu, size_t cri){return op.parent()[nu]().spf(cri);}, 
              [&op](size_t nu){return op.parent()[nu]().spf_id();}, 
              cinf[ind].mf_indexing()[ri].sibling_indices(), B, A, mode, temp, res)
            );
    }

};  //class mean field operator engine

}   //namespace ttns

#endif  //TTNS_MEAN_FIELD_OPERATOR_CORE_HPP//

