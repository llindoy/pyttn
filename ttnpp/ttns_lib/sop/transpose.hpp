/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTTN_TTNS_LIB_SOP_SYMBOLIC_TRANSPOSE_HPP_
#define PYTTN_TTNS_LIB_SOP_SYMBOLIC_TRANSPOSE_HPP_

#include <linalg/linalg.hpp>

#include "sSOP.hpp"
#include "SOP.hpp"
#include "system_information.hpp"
#include "../operators/site_operators/site_operator.hpp"
#include "operator_dictionaries/default_operator_dictionaries.hpp"
#include "operator_dictionaries/operator_dictionary.hpp"

namespace ttns
{
    class symbolic_transpose
    {
    public:
        template <typename T>
        static inline void apply(const sOP& op, const system_modes &sysinf, sNBO<T> &res)
        {
            ASSERT(op.mode() < sysinf.nmodes(), "Failed to transpose operator.  Operator index out of bounds.");
            auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(op.mode()).type(), op.op());
            auto tinfo = opinfo->transpose();
            res = std::get<0>(tinfo)*sOP(std::get<1>(tinfo), op.mode(), op.fermionic());
        }

        template <typename T>
        static inline void apply(const sPOP& op, const system_modes &sysinf, sNBO<T> &res)
        {

            T coeff(1.0);
            sPOP lt;
            for(const auto& site_op : op)
            {
                ASSERT(site_op.mode() < sysinf.nmodes(), "Failed to transpose operator.  Operator index out of bounds.");

                auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                auto tinfo = opinfo->transpose();

                coeff *= std::get<0>(tinfo);
                lt *= sOP(std::get<1>(tinfo), site_op.mode(), site_op.fermionic());
            }
            res = coeff*lt;
        }

        template <typename T>
        static inline void apply(const sNBO<T>& op, const system_modes &sysinf, sNBO<T> &res)
        {
            auto coeff = op.coeff();
            sPOP lt;
            for(const auto& site_op : op.pop())
            {
                ASSERT(site_op.mode() < sysinf.nmodes(), "Failed to transpose operator.  Operator index out of bounds.");

                auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                auto tinfo = opinfo->transpose();

                coeff *= std::get<0>(tinfo);
                lt *= sOP(std::get<1>(tinfo), site_op.mode(), site_op.fermionic());
            }
            res = coeff*lt;
        }

        template <typename T>
        static inline void apply(const sSOP<T>& op, const system_modes &sysinf, sSOP<T> &res)
        {
            ASSERT(sysinf.nprimitive_modes() >= op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
            // iterater over each term in the operator
            for (const auto &t : op)
            {
                // extracting its coefficient
                auto coeff = t.coeff();

                // and a product operator representation of it
                sPOP term = t.pop();

                sPOP lt;
                // now iterate over each term in the product
                for (const auto &site_op : term)
                {
                    // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                    // with the system mode std::pair<T, std::string>
                    auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                    auto tinfo = opinfo->transpose();
                    coeff *= std::get<0>(tinfo);
                    lt *= sOP(std::get<1>(tinfo), site_op.mode(), site_op.fermionic());
                }
                res += coeff * lt;
            }
        }

        template <typename T>
        static inline void apply(const SOP<T>& op, const system_modes &sysinf, SOP<T> &res)
        {
            ASSERT(sysinf.nprimitive_modes() >= op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
            // iterater over each term in the operator
            res.resize(op.nmodes());
            res.Eshift() = op.Eshift();
            // iterater over each term in the operator
            for (const auto &t : op)
            {
                // extracting its coefficient
                ttns::literal::coeff<T> coeff = std::get<1>(t);

                // and a product operator representation of it
                sPOP term = std::get<0>(t).as_prod_op(op.operator_dictionary());
                sPOP lt;

                // now iterate over each term in the product
                for (const auto &site_op : term)
                {
                    // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                    // with the system mode std::pair<T, std::string>
                    auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                    auto tinfo = opinfo->transpose();
                    coeff *= std::get<0>(tinfo);
                    lt *= sOP(std::get<1>(tinfo), site_op.mode(), site_op.fermionic());
                }
                res.insert(coeff, lt);
            }
        }

 public:
        template <typename T, typename U, typename backend = linalg::blas_backend>
        static inline void apply(const sOP& site_op, const operator_dictionary<T, backend> &opdict, const system_modes &sysinf, sNBO<U> &res, operator_dictionary<T, backend> &opdictf, const std::string& suffix = std::string("~"))
        {
            ASSERT(site_op.mode() < sysinf.nmodes(), "Failed to transpose operator.  Operator index out of bounds.");
            using op_type = ops::primitive<T, backend>;

            std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());
            // if the operator is in the user defined dictionary
            if (_op != nullptr)
            {
                std::string label = site_op.op() +  suffix;
                opdictf.insert(site_op.mode(), label, site_operator<T, backend>(_op->transpose(), site_op.mode()));
                res = sOP(label, site_op.mode() , site_op.fermionic());
            }
            else
            {
                // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                // with the system mode std::pair<T, std::string>
                auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                auto tinfo = opinfo->transpose();
                res = static_cast<U>(std::get<0>(tinfo))*sOP(std::get<1>(tinfo), site_op.mode(), site_op.fermionic());
            }
        }

        template <typename T, typename U, typename backend = linalg::blas_backend>
        static inline void apply(const sPOP& op, const operator_dictionary<T, backend> &opdict, const system_modes &sysinf, sNBO<U> &res, operator_dictionary<T, backend> &opdictf, const std::string& suffix = std::string("~"))
        {
            using op_type = ops::primitive<T, backend>;

            U coeff(1.0);
            sPOP lt;
            // now iterate over each term in the product
            for (const auto &site_op : op)
            {
                ASSERT(site_op.mode() < sysinf.nmodes(), "Failed to transpose operator.  Operator index out of bounds.");

                // first try to query the operator from the opdict
                std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());
                // if the operator is in the user defined dictionary
                if (_op != nullptr)
                {
                    std::string label = site_op.op() + suffix;
                    opdictf.insert(site_op.mode(), label, site_operator<T, backend>(_op->transpose(), site_op.mode()));
                    lt *= sOP(label, site_op.mode() , site_op.fermionic());
                }
                else
                {
                    // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                    // with the system mode std::pair<T, std::string>
                    auto opinfo = query_default_operator_dictionary<T>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                    auto tinfo = opinfo->transpose();
                    coeff *= std::get<0>(tinfo);
                    lt *= sOP(std::get<1>(tinfo), site_op.mode() , site_op.fermionic());
                }
            }
            res = coeff * lt;
        }

        template <typename T, typename U, typename V, typename backend = linalg::blas_backend>
        static inline void apply(const sNBO<T>& op, const operator_dictionary<U, backend> &opdict, const system_modes &sysinf, sNBO<V> &res, operator_dictionary<U, backend> &opdictf, const std::string& suffix = std::string("~"))
        {
            using op_type = ops::primitive<U, backend>;

            // extracting its coefficient
            ttns::literal::coeff<V> coeff = op.coeff();
            // and a product operator representation of it
            sPOP lt;
            // now iterate over each term in the product
            for (const auto &site_op : op.pop())
            {
                ASSERT(site_op.mode() < sysinf.nmodes(), "Failed to transpose operator.  Operator index out of bounds.");

                // first try to query the operator from the opdict
                std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());
                // if the operator is in the user defined dictionary
                if (_op != nullptr)
                {
                    std::string label = site_op.op() + suffix;
                    opdictf.insert(site_op.mode(), label, site_operator<U, backend>(_op->transpose(), site_op.mode()));
                    lt *= sOP(label, site_op.mode() , site_op.fermionic());
                }
                else
                {
                    // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                    // with the system mode std::pair<T, std::string>
                    auto opinfo = query_default_operator_dictionary<U>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                    auto tinfo = opinfo->transpose();
                    coeff *= std::get<0>(tinfo);
                    lt *= sOP(std::get<1>(tinfo), site_op.mode() , site_op.fermionic());
                }
            }
            res = coeff * lt;
        }

        template <typename T, typename U, typename V,  typename backend = linalg::blas_backend>
        static inline void apply(const sSOP<T>& op, const operator_dictionary<U, backend> &opdict, const system_modes &sysinf, sSOP<V> &res, operator_dictionary<U, backend> &opdictf, const std::string& suffix = std::string("~"))
        {
            ASSERT(sysinf.nprimitive_modes() >= op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
            ASSERT(op.nmodes() <= opdict.nmodes(), "Failed to construct left_superoperator the operator dictionary and operator are not compatible.");

            using op_type = ops::primitive<U, backend>;
            
            // iterater over each term in the operator
            for (const auto &t : op)
            {
                // extracting its coefficient
                ttns::literal::coeff<V> coeff = t.coeff();

                // and a product operator representation of it
                sPOP term = t.pop();

                sPOP lt;
                // now iterate over each term in the product
                for (const auto &site_op : term)
                {
                    // first try to query the operator from the opdict
                    std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());

                    // if the operator is in the user defined dictionary
                    if (_op != nullptr)
                    {
                        std::string label = site_op.op() + suffix;
                        opdictf.insert(site_op.mode(), label, site_operator<U, backend>(_op->transpose(), site_op.mode()));
                        lt *= sOP(label, site_op.mode() , site_op.fermionic());
                    }
                    else
                    {
                        // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                        // with the system mode std::pair<T, std::string>
                        auto opinfo = query_default_operator_dictionary<U>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                        auto tinfo = opinfo->transpose();
                        coeff *= std::get<0>(tinfo);
                        lt *= sOP(std::get<1>(tinfo), site_op.mode() , site_op.fermionic());
                    }
                }
                res += coeff * lt;
            }
        }

        template <typename T, typename U, typename V,  typename backend = linalg::blas_backend>
        static inline void apply(const SOP<T>& op, const operator_dictionary<U, backend> &opdict, const system_modes &sysinf, SOP<V> &res, operator_dictionary<U, backend> &opdictf, const std::string& suffix = std::string("~"))
        {
            ASSERT(sysinf.nprimitive_modes() >= op.nmodes(), "Failed to construct left superoperator input operator and system information are incompatible.");
            ASSERT(op.nmodes() <= opdict.nmodes(), "Failed to construct left_superoperator the operator dictionary and operator are not compatible.");

            using op_type = ops::primitive<U, backend>;
            res.resize(op.nmodes());
            res.Eshift() = op.Eshift();
            // iterater over each term in the operator
            for (const auto &t : op)
            {
                // extracting its coefficient
                ttns::literal::coeff<V> coeff = std::get<1>(t);

                // and a product operator representation of it
                sPOP term = std::get<0>(t).as_prod_op(op.operator_dictionary());
                sPOP lt;
                // now iterate over each term in the product
                for (const auto &site_op : term)
                {
                    // first try to query the operator from the opdict
                    std::shared_ptr<op_type> _op = opdict.query(site_op.mode(), site_op.op());

                    // if the operator is in the user defined dictionary
                    if (_op != nullptr)
                    {
                        std::string label = site_op.op() + suffix;
                        opdictf.insert(site_op.mode(), label, site_operator<U, backend>(_op->transpose(), site_op.mode()));
                        lt *= sOP(label, site_op.mode() , site_op.fermionic());
                    }
                    else
                    {
                        // now for each term we get the transpose operator.  This requires a query of the default operator dictionary associated
                        // with the system mode std::pair<T, std::string>
                        auto opinfo = query_default_operator_dictionary<U>(sysinf.primitive_mode(site_op.mode()).type(), site_op.op());
                        auto tinfo = opinfo->transpose();
                        coeff *= std::get<0>(tinfo);
                        lt *= sOP(std::get<1>(tinfo), site_op.mode() , site_op.fermionic());
                    }
                }
                res.insert(coeff, lt);
            }
        }
    };
}

#endif // PYTTN_TTNS_LIB_SOP_SYMBOLIC_TRANSPOSE_HPP_
