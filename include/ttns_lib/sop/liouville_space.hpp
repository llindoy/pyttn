#ifndef TTNS_OPERATOR_GEN_LIB_LIOUVILLE_SPACE_HPP
#define TTNS_OPERATOR_GEN_LIB_LIOUVILLE_SPACE_HPP

#include <linalg/linalg.hpp>

#include "sSOP.hpp"
#include "SOP.hpp"
#include "system_information.hpp"
#include "../operators/site_operators/site_operator.hpp"
#include "operator_dictionaries/default_operator_dictionaries.hpp"
#include "operator_dictionaries/operator_dictionary.hpp"


namespace ttns
{
class liouville_space
{
    //implementation of the function for generating generic superoperators without handling user defined operators
public:
    //for the left superoperator object.  We just iterate over each term in op and construct a new SOP object acting
    //on a space with twice the dimension where the original modes correspond to the even degrees of freedom
    template <typename T>
    static inline void left_superoperator(const SOP<T>& op, const system_modes& sysinf, SOP<T>& res, T gcoeff = T(1.0))
    {
        ASSERT(sysinf.nprimitive_modes() == op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
        //make sure the res operator can fit the result
        if(res.nmodes() != 2*op.nmodes())
        {
            res.clear();
            res.resize(2*op.nmodes());
        }

        //iterater over each term in the operator
        for(const auto& t : op)
        {
            //extracting its coefficient
            auto coeff = t.second;

            //and a product operator representation of it
            sPOP term = t.first.as_prod_op(op.operator_dictionary());

            sPOP lt;
            //now iterate over each term in the product
            for(const auto& site_op : term)
            {
                lt *= sOP(site_op.op(), 2*site_op.mode(), site_op.fermionic());
            }
            res += gcoeff*coeff*lt;
        }
    }
    
    //for the right superoperator object.  We just iterate over each term in op and construct a new SOP object acting
    //on a space with twice the dimension where the original modes correspond to the odd degrees of freedom.  And for
    //each operator we need to find out the transpose degree of freedom.
    template <typename T>
    static inline void right_superoperator(const SOP<T>& op, const system_modes& sysinf, SOP<T>& res, T gcoeff = T(1.0))
    {
        ASSERT(sysinf.nprimitive_modes() == op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
        //make sure the res operator can fit the result
        if(res.nmodes() != 2*op.nmodes())
        {
            res.clear();
            res.resize(2*op.nmodes());
        }

        //iterater over each term in the operator
        for(const auto& t : op)
        {
            //extracting its coefficient
            auto coeff = t.second;

            //and a product operator representation of it
            sPOP term = t.first.as_prod_op(op.operator_dictionary());

            sPOP lt;
            //now iterate over each term in the product
            for(const auto& site_op : term)
            {
                //now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                //with the system mode std::pair<T, std::string>
                auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                auto tinfo = opinfo->transpose();
                coeff *= std::get<0>(tinfo);
                lt *= sOP(std::get<1>(tinfo), 2*site_op.mode()+1, site_op.fermionic());
            }
            res += gcoeff*coeff*lt;
        }
    }

    //construct the commutator superoperator object
    template <typename T>
    static inline void commutator_superoperator(const SOP<T>& op, const system_modes& sysinf, SOP<T>& res, T gcoeff = T(1.0))
    {
        CALL_AND_RETHROW(left_superoperator(op, sysinf, res, gcoeff));
        CALL_AND_RETHROW(right_superoperator(op, sysinf, res, T(-1.0)*gcoeff));
    }

    //construct the anticommutator superoperator object
    template <typename T>
    static inline void anticommutator_superoperator(const SOP<T>& op, const system_modes& sysinf, SOP<T>& res, T gcoeff = T(1.0))
    {
        CALL_AND_RETHROW(left_superoperator(op, sysinf, res, gcoeff));
        CALL_AND_RETHROW(right_superoperator(op, sysinf, res, gcoeff));
    }

    //implementation of the function for generating generic superoperators that handle and update user defined operators
public:
    //for the left superoperator object.  We just iterate over each term in op and construct a new SOP object acting
    //on a space with twice the dimension where the original modes correspond to the even degrees of freedom
    template <typename T, typename backend = linalg::blas_backend>
    static inline void left_superoperator(const SOP<T>& op, const system_modes& sysinf, const operator_dictionary<T, backend>& opdict, SOP<T>& res, operator_dictionary<T, backend>& opdictf, T gcoeff = T(1.0))
    {
        ASSERT(sysinf.nprimitive_modes() == op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
        //make sure the res operator can fit the result
        if(res.nmodes() != 2*op.nmodes())
        {
            res.clear();
            res.resize(2*op.nmodes());
        }

        ASSERT(op.nmodes() == opdict.nmodes(), "Failed to construct left_superoperator the operator dictionary and operator are not compatible.");

        if(opdictf.nmodes() != 2*opdict.nmodes())
        {
            opdictf.clear();
            opdictf.resize(2*opdict.nmodes());
        }
        using op_type = ops::primitive<T, backend>;

        //iterater over each term in the operator
        for(const auto& t : op)
        {
            //extracting its coefficient
            auto coeff = t.second;

            //and a product operator representation of it
            sPOP term = t.first.as_prod_op(op.operator_dictionary());

            sPOP lt;
            //now iterate over each term in the product
            for(const auto& site_op : term)
            {
                //first try to query the operator from the opdict 
                std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());

                //if the operator is in the user defined dictionary
                if(_op != nullptr)
                {
                    opdictf.insert(2*site_op.mode(), site_op.op(), site_operator<T, backend>(_op, 2*site_op.mode()));
                }
                lt *= sOP(site_op.op(), 2*site_op.mode(), site_op.fermionic());
            }
            res += gcoeff*coeff*lt;
        }
    }

    //for the right superoperator object.  We just iterate over each term in op and construct a new SOP object acting
    //on a space with twice the dimension where the original modes correspond to the odd degrees of freedom.  And for
    //each operator we need to find out the transpose degree of freedom.
    template <typename T, typename backend = linalg::blas_backend>
    static inline void right_superoperator(const SOP<T>& op, const system_modes& sysinf, const operator_dictionary<T, backend>& opdict, SOP<T>& res, operator_dictionary<T, backend>& opdictf, T gcoeff = T(1.0))
    {
        ASSERT(sysinf.nprimitive_modes() == op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
        //make sure the res operator can fit the result
        if(res.nmodes() != 2*op.nmodes())
        {
            res.clear();
            res.resize(2*op.nmodes());
        }

        ASSERT(op.nmodes() == opdict.nmodes(), "Failed to construct left_superoperator the operator dictionary and operator are not compatible.");

        if(opdictf.nmodes() != 2*opdict.nmodes())
        {
            opdictf.clear();
            opdictf.resize(2*opdict.nmodes());
        }

        using op_type = ops::primitive<T, backend>;

        //iterater over each term in the operator
        for(const auto& t : op)
        {
            //extracting its coefficient
            auto coeff = t.second;

            //and a product operator representation of it
            sPOP term = t.first.as_prod_op(op.operator_dictionary());

            sPOP lt;
            //now iterate over each term in the product
            for(const auto& site_op : term)
            {
                //first try to query the operator from the opdict 
                std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());

                //if the operator is in the user defined dictionary
                if(_op != nullptr)
                {
                    std::string label = site_op.op()+std::string("_tilde");
                    opdictf.insert(2*site_op.mode()+1, label, site_operator<T, backend>(_op->transpose(), 2*site_op.mode()+1));
                    lt *= sOP(label, 2*site_op.mode()+1, site_op.fermionic());
                }
                else
                {
                    //now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                    //with the system mode std::pair<T, std::string>
                    auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                    auto tinfo = opinfo->transpose();
                    coeff *= std::get<0>(tinfo);
                    lt *= sOP(std::get<1>(tinfo), 2*site_op.mode()+1, site_op.fermionic());
                }
            }
            res += gcoeff*coeff*lt;
        }

    }

    //construct the commutator superoperator object
    template <typename T, typename backend = linalg::blas_backend>
    static inline void commutator_superoperator(const SOP<T>& op, const system_modes& sysinf, const operator_dictionary<T, backend>& opdict, SOP<T>& res, operator_dictionary<T, backend>& opdictf, T gcoeff = T(1.0))
    {
        CALL_AND_RETHROW(left_superoperator(op, sysinf, opdict, res, opdictf, gcoeff));
        CALL_AND_RETHROW(right_superoperator(op, sysinf, opdict, res, opdictf, T(-1.0)*gcoeff));
    }

    //construct the anticommutator superoperator object
    template <typename T, typename backend = linalg::blas_backend>
    static inline void anticommutator_superoperator(const SOP<T>& op, const system_modes& sysinf, const operator_dictionary<T, backend>& opdict, SOP<T>& res, operator_dictionary<T, backend>& opdictf, T gcoeff = T(1.0))
    {
        CALL_AND_RETHROW(left_superoperator(op, sysinf, opdict, res, opdictf, gcoeff));
        CALL_AND_RETHROW(right_superoperator(op, sysinf, opdict, res, opdictf, gcoeff));
    }
};

}

#endif
