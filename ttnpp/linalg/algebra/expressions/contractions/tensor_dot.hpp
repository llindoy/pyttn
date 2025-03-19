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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_TENSOR_DOT_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_TENSOR_DOT_HPP_

#include <type_traits>
#include <utility>
#include <algorithm>
#include <set>

#include "../permutations/transpose_expression.hpp"
#include "../permutations/tensor_transpose.hpp"

namespace linalg
{

    namespace expression_templates
    {

        // diagonal matrix dense matrix product
        template <typename a_tensor_type, typename b_tensor_type, size_t D>
        class tensordot_expr<a_tensor_type, b_tensor_type, D, void> : public tensor_contraction_expression, public expression_base<tensordot_expr<a_tensor_type, b_tensor_type, D>, true>, public dense_type
        {
            static_assert(is_same_value<a_tensor_type, b_tensor_type>::value, "Failed to construct tensordot_expr object.  The two input tensors must have the same value_type.");
            static_assert(is_same_backend<a_tensor_type, b_tensor_type>::value, "Failed to construct tensordot_expr object.  The two input tensors must have the same backend type.");

            static_assert(traits<a_tensor_type>::rank >= D && traits<b_tensor_type>::rank >= D, "Failed to construct tensordot_expr object.  The two input tensor have rank larger than the contraction index arrays.");
            static_assert(is_dense<a_tensor_type>::value && is_dense<b_tensor_type>::value, "Failed to construct tensordot_expr object.  The two input tensor must both be dense tensors.");

            using a_value_type = typename traits<a_tensor_type>::value_type;
            using b_value_type = typename traits<b_tensor_type>::value_type;
            using a_value_ptr = typename std::add_pointer<typename std::add_const<a_value_type>::type>::type;
            using b_value_ptr = typename std::add_pointer<typename std::add_const<b_value_type>::type>::type;

        public:
            using base_type = expression_base<tensordot_expr<a_tensor_type, b_tensor_type, D>, true>;
            static constexpr size_t rank = traits<a_tensor_type>::rank + traits<b_tensor_type>::rank - (2 * D);
            using backend_type = typename traits<a_tensor_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using shape_type = std::array<size_type, rank>;
            using ttype = typename backend_type::transform_type;

            using result_type = linalg::tensor<decltype(a_value_type() * b_value_type()), rank, backend_type>;

        protected:
            a_tensor_type _trans_a;
            b_tensor_type _trans_b;

            const a_tensor_type &_A;
            const b_tensor_type &_B;

            std::array<size_type, D> _Ainds;
            std::array<size_type, D> _Binds;

        public:
            template <typename i1, typename i2>
            tensordot_expr(const a_tensor_type &A, const b_tensor_type &B, const std::array<i1, D> &cindsa, const std::array<i2, D> &cindsb) : base_type(shape_type{{}}), _A(A), _B(B)
            {
                // check that the index arrays are all valid
                for (size_type i = 0; i < D; ++i)
                {
                    ASSERT(cindsa[i] >= 0, "Failed to compute tensordot, A contraction index out of bounds.");
                    ASSERT(cindsb[i] >= 0, "Failed to compute tensordot, B contraction index out of bounds.");

                    _Ainds[i] = cindsa[i];
                    _Binds[i] = cindsb[i];

                    ASSERT(_Ainds[i] < traits<a_tensor_type>::rank, "Failed to compute tensordot, A contraction index out of bounds.");
                    ASSERT(_Binds[i] < traits<b_tensor_type>::rank, "Failed to compute tensordot, B contraction index out of bounds.");
                }
                {
                    std::set<size_type> sA(_Ainds.begin(), _Ainds.end());
                    std::set<size_type> sB(_Binds.begin(), _Binds.end());
                    ASSERT(sA.size() == _Ainds.size(), "Failed to compute tensordot, repeated A contraction index.");
                    ASSERT(sB.size() == _Binds.size(), "Failed to compute tensordot, repeated B contraction index.");
                }

                // work out the size of the final contracted array. This makes use of the numpy result order so the result indices will be ordered
                // uncontracted indices of A in order followed by uncontracted indices of B in order
                size_type counter = 0;
                for (size_type i = 0; i < traits<a_tensor_type>::rank; ++i)
                {
                    if (std::find(_Ainds.begin(), _Ainds.end(), i) == _Ainds.end())
                    {
                        this->m_shape[counter] = A.shape(i);
                        ++counter;
                    }
                }
                for (size_type i = 0; i < traits<b_tensor_type>::rank; ++i)
                {
                    if (std::find(_Binds.begin(), _Binds.end(), i) == _Binds.end())
                    {
                        this->m_shape[counter] = B.shape(i);
                        ++counter;
                    }
                }
            }

            const a_tensor_type &Atensor() const { return _A; }
            const b_tensor_type &Btensor() const { return _B; }
            const std::array<size_type, D> &contraction_inds_A() const { return _Ainds; }
            const std::array<size_type, D> &contraction_inds_B() const { return _Binds; }

        public:
            // function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
            template <typename array_type>
            typename std::enable_if<is_dense<array_type>::value && array_type::rank, void>::type applicative(array_type &res)
            {
                ASSERT(res.buffer() != _A.buffer() && res.buffer() != _B.buffer(), "tensordot does not support inplace products.");

                // now check that the contraction indices of the two arrays are compatible
                size_type k = 1;
                for (size_type i = 0; i < D; ++i)
                {
                    ASSERT(_A.shape(_Ainds[i]) == _B.shape(_Binds[i]), "Failed to compute tensordot.  Tensor dimensions are incompatible.");
                    k *= _A.shape(_Ainds[i]);
                }

                std::array<size_type, D> Binds = _Binds;

                size_type m = _A.size() / k;
                size_type n = _B.size() / k;

                // check if the Ainds array is sorted
                bool A_requires_transpose = false;
                bool B_requires_transpose = false;
                // bool B_contiguous = true;

                std::vector<size_type> Ainds;
                // check to see if the A tensor is going to require being transposed.
                for (size_type i = 0; i < D; ++i)
                {
                    if (_Ainds[i] >= D)
                    {
                        A_requires_transpose = true;
                    }
                }
                for (size_type i = 0; i < D; ++i)
                {
                    if (Binds[i] >= D)
                    {
                        B_requires_transpose = true;
                    }
                }

                // TODO: ensure that we only compute the transposes if it is required.

                /*
                //now if A requires being transposed then see if B requires being transposed. This will be the same check as if A requires transpose
                if(A_requires_transpose)
                {
                    for(size_type i = 0; i < D; ++i)
                    {
                        if(Binds[i] >= D){B_requires_transpose = true;}
                    }
                }
                //but if A doesn't require transpose then there are two possible cases.
                //Either B would independently require a transpose to get the elements in the correct order, or it only requires the transpose because
                //its elements aren't ordered the same as A.  If that is the case we will set the binds array to be slightly different
                else
                {
                    bool all_at_front = true;
                    if(size_type i = 0; i < D; ++i)
                    {
                        if(Binds[i] != _Ainds[i]){B_requires_transpose = true;}
                        if(Binds[i] >= D){all_at_front = false;}
                    }


                    //if all  of the b elements that are being contracted are all at the front, but we require a transpose of B but not
                    //A, then we won't be automatically putting the A elements into the correct spot.  So here we will permute the B indices
                    //So that the transpose gets them all matching up with the A elements.  To do this we just permute the indices as we require
                    if(all_at_front && B_requires_transpose)
                    {
                        for(size_type i = 0; i < D; ++i)
                        {

                        }
                    }

                }
                */

                if (A_requires_transpose || true)
                {
                    std::vector<size_type> inds;
                    // make sure the contracted A indices are at the start of the tensor
                    for (size_type i = 0; i < D; ++i)
                    {
                        inds.push_back(_Ainds[i]);
                    }
                    // and set the rest to the back in the order they come
                    for (size_type i = 0; i < traits<a_tensor_type>::rank; ++i)
                    {
                        if (std::find(inds.begin(), inds.end(), i) == inds.end())
                        {
                            inds.push_back(i);
                        }
                    }

                    // and now tensor transpose storing into _trans_a.
                    auto transa = tensor_transpose_expression<a_tensor_type>(_A, inds);
                    _trans_a = transa;
                }
                if (B_requires_transpose || true)
                {
                    std::vector<size_type> inds;
                    // make sure the contracted A indices are at the start of the tensor
                    for (size_type i = 0; i < D; ++i)
                    {
                        inds.push_back(_Binds[i]);
                    }
                    // and set the rest to the back in the order they come
                    for (size_type i = 0; i < traits<b_tensor_type>::rank; ++i)
                    {
                        if (std::find(inds.begin(), inds.end(), i) == inds.end())
                        {
                            inds.push_back(i);
                        }
                    }

                    // and now tensor transpose storing into _trans_a.
                    auto transb = tensor_transpose_expression<b_tensor_type>(_B, inds);
                    _trans_b = transb;
                }

                res.reinterpret_shape(m, n) = transpose_expression<decltype(_trans_a.reinterpret_shape(k, m)), false>(_trans_a.reinterpret_shape(k, m)) * _trans_b.reinterpret_shape(k, n);
                /*
                if(A_requires_transpose && B_requires_transpose)
                {
                    CALL_AND_RETHROW(apply_contraction(m, n, k, _trans_a.buffer(), _trans_b.buffer(), res.buffer()));
                }
                else if (A_requires_transpose)
                {
                    CALL_AND_RETHROW(apply_contraction(m, n, k, _trans_a.buffer(), _B.buffer(), res.buffer()));
                }
                else if(B_requires_transpose)
                {
                    CALL_AND_RETHROW(apply_contraction(m, n, k, _A.buffer(), _trans_b.buffer(), res.buffer()));
                }
                else
                {
                    CALL_AND_RETHROW(apply_contraction(m, n, k, _A.buffer(), _B.buffer(), res.buffer()));
                }*/
            }

        }; // tensordot

    } // namespace expression_templates
    template <typename a_tensor_type, typename b_tensor_type, size_t D>
    struct traits<expression_templates::tensordot_expr<a_tensor_type, b_tensor_type, D>>
    {
        using value_type = typename traits<a_tensor_type>::value_type;
        using backend_type = typename traits<a_tensor_type>::backend_type;
        static constexpr size_t rank = traits<a_tensor_type>::rank + traits<b_tensor_type>::rank - (2 * D);
        using shape_type = std::array<typename backend_type::size_type, rank>;
        using const_shape_reference = const shape_type &;
    };
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_TENSOR_DOT_HPP_
