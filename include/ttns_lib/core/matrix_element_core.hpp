#ifndef TTNS_MATRIX_ELEMENT_CORE_HPP
#define TTNS_MATRIX_ELEMENT_CORE_HPP

#include <linalg/linalg.hpp>
#include "kronecker_product_operator_helper.hpp"
#include "observable_node.hpp"

namespace ttns
{


//A class containing static functions that evaluate the contractions required for computing matrix elements.
template <typename T, typename backend>
class matrix_element_engine
{
public:
    using hdata = ttn_node_data<T, backend>;
    using mat = linalg::matrix<T, backend>;

    using triad_type = std::vector<mat>;

    using ob_type = observable_node_data<T, backend>;
    using observable_node = typename tree<ob_type>::node_type;
    using boolnode = typename tree<bool>::node_type;

    using size_type = typename hdata::size_type;
    using op_base = ops::primitive<T, backend>;

public:
    static inline void compute_leaf(const hdata& p, observable_node& res, size_type r, boolnode& is_identity)
    {
        try
        {
            CALL_AND_HANDLE(res()[r].resize(p.hrank(), p.hrank()), "Failed to resize matel object.");

            const auto& psi = p.as_matrix();
            CALL_AND_HANDLE(res()[r] = adjoint(psi)*psi, "Failed to apply the leaf node contraction.");
            is_identity() = false;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing leaf node norm squared of hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute leaf node norm squared of hierarchical tucker tensor.");
        }
    }

    template <typename op_type>
    static inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, void>::type compute_leaf(op_type& op, const hdata& p, triad_type& temp, observable_node& res, size_type r, boolnode& is_identity)
    {
        try
        {
            if(op.is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(p, res, r, is_identity), "Failed to treate the case where the operator is the identity operator.");
            }
            else
            {
                size_type ti = omp_get_thread_num();
                ASSERT(op.nmodes() == p.dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the ttn node.");

                CALL_AND_HANDLE(res()[r].resize(p.hrank(), p.hrank()), "Failed to resize matel object.");

                const auto& psi = p.as_matrix();      auto& HA = temp[ti];
                CALL_AND_HANDLE(op.apply(psi, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res()[r] = adjoint(psi)*HA, "Failed to apply the leaf node contraction.");
                is_identity() = false;
            }
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
    }


    static inline void compute_leaf(std::shared_ptr<op_base> op, const hdata& p, triad_type& temp, observable_node& res, size_type r, boolnode& is_identity)
    {
        try
        {
            if(op->is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(p, res, r, is_identity), "Failed to treate the case where the operator is the identity operator.");
            }
            else
            {
                size_type ti = omp_get_thread_num();
                ASSERT(op->size() == p.dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the ttn node.");

                CALL_AND_HANDLE(res()[r].resize(p.hrank(), p.hrank()), "Failed to resize matel object.");

                const auto& psi = p.as_matrix();      auto& HA = temp[ti];
                CALL_AND_HANDLE(op->apply(psi, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res()[r] = adjoint(psi)*HA, "Failed to apply the leaf node contraction.");
                is_identity() = false;
            }
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
    }

    static inline void compute_leaf(site_operator<T>& op, const hdata& p, triad_type& temp, observable_node& res, size_type r, boolnode& is_identity)
    {
        try
        {
            if(op.is_identity())
            {
                CALL_AND_HANDLE(compute_leaf(p, res, r, is_identity), "Failed to treate the case where the operator is the identity operator.");
            }
            else
            {
                size_type ti = omp_get_thread_num();
                ASSERT(op.size() == p.dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the ttn node.");

                CALL_AND_HANDLE(res()[r].resize(p.hrank(), p.hrank()), "Failed to resize matel object.");

                const auto& psi = p.as_matrix();      auto& HA = temp[ti];
                CALL_AND_HANDLE(op.apply(psi, HA), "Failed to evaluate the action of the operator on the ket vector.");
                CALL_AND_HANDLE(res()[r] = adjoint(psi)*HA, "Failed to apply the leaf node contraction.");
                is_identity() = false;
            }
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute operator expectation value for a leaf node of the hierarchical tucker tensor.");
        }
    }

    static inline void compute_branch(const hdata& p, triad_type& HA, triad_type& temp, observable_node& res, size_type r, boolnode& is_identity)
    {
        try
        {
            ASSERT(!res.is_leaf(), "Cannot apply branch contraction to a leaf node.");
            CALL_AND_HANDLE(res()[r].resize(p.hrank(), p.hrank()), "Failed to resize matel object.");

            //check the size of the child nodes and check if all child nodes are identity operators
            for(size_type i=0; i<is_identity.size(); ++i)
            {
                if(!is_identity[i]())
                {
                    ASSERT(res[i]()[r].size(0) == res[i]()[r].size(1) && res[i]()[r].size(1) == p.dim(i), "The child operator nodes are not the correct shape.");
                }
            }

            //if some of the children are not the identity operators then we need to evaluate the kronecker product operator.
            size_type ti = omp_get_thread_num();
            const auto& psi = p.as_matrix();  auto& ha = HA[ti];    auto& t = temp[ti];
    
            using kpo = kronecker_product_operator_mel<T, backend>;
            CALL_AND_HANDLE(kpo::apply(res, r, is_identity, p, t, ha), "Failed to apply kronecker product operator.");
            CALL_AND_HANDLE(res()[r] = adjoint(psi)*ha, "Failed to apply matrix product to obtain result.");
            is_identity() = false;
        }        
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
    }

public:
    static inline void compute_leaf(const hdata& b, const hdata& k, observable_node& res, size_type r)
    {
        try
        {            
            CALL_AND_HANDLE(res()[r].resize(b.hrank(), k.hrank()), "Failed to resize matel object.");
            const auto& bra = b.as_matrix();      const auto& ket = k.as_matrix();
            CALL_AND_HANDLE(res()[r] = adjoint(bra)*ket, "Failed to apply the leaf node contraction.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing leaf node inner product between two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute leaf node inner product between two hierarchical tucker tensors.");
        }
    }

    template <typename op_type>
    static inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, void>::type compute_leaf(op_type& op, const hdata& b, const hdata& k, triad_type& temp, observable_node& res, size_type r)
    {
        try
        {
            ASSERT(op.nmodes() == b.dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the ttn node.");

            CALL_AND_HANDLE(res()[r].resize(b.hrank(), k.hrank()), "Failed to resize matel object.");

            size_type ti = omp_get_thread_num();
            const auto& bra = b.as_matrix();      const auto& ket = k.as_matrix();      auto& HA = temp[ti];
            CALL_AND_HANDLE(op.apply(ket, HA), "Failed to evaluate the action of the operator on the ket vector.");
            CALL_AND_HANDLE(res()[r] = adjoint(bra)*HA, "Failed to apply the leaf node contraction.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute a matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
    }

    static inline void compute_leaf(std::shared_ptr<op_base> op, const hdata& b, const hdata& k, triad_type& temp, observable_node& res, size_type r)
    {
        try
        {
            ASSERT(op->size() == b.dimen(), "The dimension of the operator must be the same as the dimension of the primitive dimensions of the ttn node.");

            CALL_AND_HANDLE(res()[r].resize(b.hrank(), k.hrank()), "Failed to resize matel object.");

            size_type ti = omp_get_thread_num();
            const auto& bra = b.as_matrix();      const auto& ket = k.as_matrix();      auto& HA = temp[ti];
            CALL_AND_HANDLE(op->apply(ket, HA), "Failed to evaluate the action of the operator on the ket vector.");
            CALL_AND_HANDLE(res()[r] = adjoint(bra)*HA, "Failed to apply the leaf node contraction.");
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute a matrix element of an operator between the leaf nodes of two hierarchical tucker tensors.");
        }
    }

    static inline void compute_leaf(site_operator<T>& op, const hdata& b, const hdata& k, triad_type& temp, observable_node& res, size_type r)
    {
        CALL_AND_RETHROW(compute_leaf(op.op(), b, k, temp, res, r));
    }

    static inline void compute_branch(const hdata& b, const hdata& k, triad_type& HA, triad_type& temp, observable_node& res, size_type r)
    {
        try
        {
            ASSERT(!res.is_leaf(), "Cannot apply branch contraction to a leaf node.");
            CALL_AND_HANDLE(res()[r].resize(b.hrank(), k.hrank()), "Failed to resize matel object.");

            const auto& bra = b.as_matrix();
            using kpo = kronecker_product_operator_mel<T, backend>;

            size_type ti = omp_get_thread_num();
            //handle the case where the dimensions of the bra and ket are the same
            if(b.dims() == k.dims())
            {
                CALL_AND_HANDLE(kpo::apply(res, r, k, temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
            }
            else
            {
                CALL_AND_HANDLE(kpo::apply_rectangular(res, r, k, temp[ti], HA[ti]), "Failed to apply kronecker product operator.");
            }
            
            auto _HA = HA[ti].reinterpret_shape(b.shape(0), k.shape(1));
            CALL_AND_HANDLE(res()[r] = adjoint(bra)*_HA, "Failed to apply matrix product to obtain result.");
        }        
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute branch contraction required for evaluating matrix elements using a hierarchical tucker tensor representation of states.");
        }
    }

};

}   //namespace ttns


#endif  //TTNS_MATRIX_ELEMENT_CORE_HPP//

