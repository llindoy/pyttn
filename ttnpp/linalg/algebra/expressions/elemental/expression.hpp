/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_EXPRESSION_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_EXPRESSION_HPP_

#include "../../../linalg_forward_decl.hpp"
#include "../expression_base.hpp"


#include "result_type.hpp"
#include "storage_traits.hpp"
#include "../../../backends/blas/blas_backend.hpp"

namespace linalg
{

    namespace expression_templates
    {

        template <typename T, typename backend>
        class literal_type
        {
            static_assert(is_number<T>::value, "Failed to initialise literal type object.  The literal must be a number type.");

        public:
            using value_type = typename device_type<T, backend>::type;

            using backend_type = backend;
            literal_type(T val) ;
        };

        namespace internal
        {
            // generic wrapper
            template <typename type, typename expr, typename backend>
            struct expression_applicative
            {
                using size_type = typename traits<backend>::size_type;
                template <typename res>
                static inline void apply(res &_res, const expr &_expr);
                template <typename res>
                static inline void addition_assign(res &_res, const expr &_expr);
                template <typename res>
                static inline void subtraction_assign(res &_res, const expr &_expr);
            };

            // wrapper for diagonal matrix return type
            template <typename expr, typename backend>
            struct expression_applicative<diagonal_matrix_type, expr, backend>
            {
                using size_type = typename traits<backend>::size_type;
                template <typename res>
                static inline void apply(res &_res, const expr &_expr);
                template <typename res>
                static inline void addition_assign(res &_res, const expr &_expr);
                template <typename res>
                static inline void subtraction_assign(res &_res, const expr &_expr);
            };
        } // namespace internal

        /////////////////////////////////////////////////////////////////////////////////
        //               Unary expression type wrapper for generic backend                //
        /////////////////////////////////////////////////////////////////////////////////
        template <typename vtype, template <typename> class operation, typename backend>
        class unary_expression
        {
        private:
            using vtraits = storage_traits<vtype>;
            typename vtraits::type m_val;
            typename vtraits::eval_type m_eval;

        public:
            using self_type = unary_expression<vtype, operation, backend>;
            using backend_type = backend;
            using size_type = typename traits<backend_type>::size_type;
            using op_type = operation<backend_type>;
            using value_type = typename result_type<self_type>::value_type;

            using result_type = typename result_type<self_type>::type;
            using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

            unary_expression(typename vtraits::type v);
            typename vtraits::type obj() const;

            template <typename array_type>
            void operator()(array_type &res) const;            
            template <typename array_type>
            void addition_assign(array_type &res) const;            
            template <typename array_type>
            void subtraction_assign(array_type &res) const;        
        }; // class unary_expression

        /////////////////////////////////////////////////////////////////////////////////
        //              Binary expression type wrapper for generic backend                //
        /////////////////////////////////////////////////////////////////////////////////
        template <typename ltype, typename rtype, template <typename> class operation, typename backend>
        class binary_expression
        {
        private:
            using ltraits = storage_traits<ltype>;
            using rtraits = storage_traits<rtype>;

        public:
            using self_type = binary_expression<ltype, rtype, operation, backend>;
            using backend_type = backend;
            using size_type = typename traits<backend_type>::size_type;
            using op_type = operation<backend_type>;
            using value_type = typename result_type<self_type>::value_type;
            using result_type = typename result_type<self_type>::type;
            using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

            binary_expression(typename ltraits::type l, typename rtraits::type r);            
            
            typename ltraits::type left() const;
            typename rtraits::type right() const;

            template <typename array_type>
            void operator()(array_type &res) const;
            template <typename array_type>
            void addition_assign(array_type &res) const;
            template <typename array_type>
            void subtraction_assign(array_type &res) const;      
        }; // class binary_expression

        /////////////////////////////////////////////////////////////////////////////////
        // Top level expression tree object for wrapping an exposed binary expression. //
        /////////////////////////////////////////////////////////////////////////////////
        template <typename ltype, typename rtype, template <typename> class operation, typename backend, size_t _rank>
        class expression_tree<binary_expression<ltype, rtype, operation, backend>, _rank, backend>
            : public expression_base<expression_tree<binary_expression<ltype, rtype, operation, backend>, _rank, backend>, false>, public result_type<binary_expression<ltype, rtype, operation, backend>>::type
        {
        public:
            using expr = binary_expression<ltype, rtype, operation, backend>;
            using rtraits = result_type<expr>;
            using type = typename rtraits::type;
            using shape_type = typename rtraits::shape_type;
            using const_shape_reference = typename rtraits::const_shape_reference;

            using value_type = typename rtraits::value_type;
            static constexpr size_t rank = rtraits::rank;
            static_assert(_rank == rank, "Failed to construct expression_tree object.  The specified rank is not compatible with the result_type rank.");
            using self_type = expression_tree<expr, rank, backend>;
            using base_type = expression_base<self_type, false>;

        private:
            expr m_expr;

            template <typename arr>
            using valid_result_array = typename std::conditional<std::is_base_of<type, arr>::value && std::is_same<backend, typename traits<arr>::backend_type>::value && std::is_same<value_type, typename traits<arr>::value_type>::value, std::true_type, std::false_type>::type;

        public:
            expression_tree(const expr &_expr, shape_type _shape) : base_type(_shape), m_expr(_expr) {}
            expression_tree() = delete;
            ~expression_tree() {}

            const expr &expression() const { return m_expr; }
            auto left() const -> decltype(m_expr.left()) { return m_expr.left(); }
            auto right() const -> decltype(m_expr.right()) { return m_expr.right(); }
            template <typename array_type>
            typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type applicative(array_type &res) const { CALL_AND_HANDLE(m_expr(res), "Failed to evaluate expression object into result array."); }

            template <typename array_type>
            typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type addition_applicative(array_type &res) const { CALL_AND_HANDLE(m_expr.addition_assign(res), "Failed to evaluate expression object into result array."); }

            template <typename array_type>
            typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type subtraction_applicative(array_type &res) const { CALL_AND_HANDLE(m_expr.subtraction_assign(res), "Failed to evaluate expression object into result array."); }
        }; // class expression_tree

        /////////////////////////////////////////////////////////////////////////////////
        //  Top level expression tree object for wrapping an exposed unary expression  //
        /////////////////////////////////////////////////////////////////////////////////
        template <typename vtype, template <typename> class operation, typename backend, size_t _rank>
        class expression_tree<unary_expression<vtype, operation, backend>, _rank, backend>
            : public expression_base<expression_tree<unary_expression<vtype, operation, backend>, _rank, backend>, false>, public result_type<unary_expression<vtype, operation, backend>>::type
        {
        public:
            using expr = unary_expression<vtype, operation, backend>;

            using rtraits = result_type<expr>;
            using type = typename rtraits::type;
            using shape_type = typename rtraits::shape_type;
            using const_shape_reference = typename rtraits::const_shape_reference;
            using value_type = typename rtraits::value_type;

            static constexpr size_t rank = rtraits::rank;
            static_assert(_rank == rank, "Failed to construct expression_tree object.  The specified rank is not compatible with the result_type rank.");

            using self_type = expression_tree<expr, rank, backend>;
            using base_type = expression_base<self_type, false>;

        private:
            expr m_expr;

            template <typename arr>
            using valid_result_array = typename std::conditional<std::is_base_of<type, arr>::value && std::is_same<backend, typename traits<arr>::backend_type>::value && std::is_same<value_type, typename traits<arr>::value_type>::value, std::true_type, std::false_type>::type;

        public:
            expression_tree(const expr &_expr, shape_type _shape) : base_type(_shape), m_expr(_expr) {}
            expression_tree() = delete;
            ~expression_tree() {}

            const expr &expression() const { return m_expr; }
            auto obj() const -> decltype(m_expr.obj()) { return m_expr.obj(); }

            template <typename array_type>
            typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type applicative(array_type &res) const { CALL_AND_HANDLE(m_expr(res), "Failed to evaluate expression object into result array."); }
            template <typename array_type>
            typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type addition_applicative(array_type &res) const { CALL_AND_HANDLE(m_expr.addition_assign(res), "Failed to evaluate expression object into result array."); }
            template <typename array_type>
            typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type subtraction_applicative(array_type &res) const { CALL_AND_HANDLE(m_expr.subtraction_assign(res), "Failed to evaluate expression object into result array."); }

        }; // class expression_tree

    } // namespace expression_templates

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //                          traits objects for the expression types                            //
    /////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename expr, size_t D, typename backend>
    struct traits<expression_templates::expression_tree<expr, D, backend>>
    {
        static constexpr size_t rank = D;
        using rtraits = expression_templates::result_type<expr>;
        using shape_type = typename rtraits::shape_type;
        using const_shape_reference = typename rtraits::const_shape_reference;
        using value_type = typename rtraits::value_type;
        using backend_type = backend;
    };

} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_EXPRESSION_HPP_//
