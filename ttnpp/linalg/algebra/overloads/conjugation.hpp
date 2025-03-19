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

#ifndef PYTTN_LINALG_ALGEBRA_OVERLOADS_CONJUGATION_HPP_
#define PYTTN_LINALG_ALGEBRA_OVERLOADS_CONJUGATION_HPP_

///
//  A File containing the various overloaded functions for constructing complex conjugate expressions
///

namespace linalg
{

    template <typename T>
    conj_return_type<T> conj(const T &a) { return conj_type<T>(conj_unary_type<T>(a), a.shape()); }

    // conjugation of scalar * tensor
    template <typename T1, typename T2>
    scalconj_return_type<T1, T2> conj(const scal_type<T1, T2> &a)
    {
        using std::conj;
        using lit_type = expression_templates::literal_type<T1, typename traits<T2>::backend_type>;
        return scalconj_type<T1, T2>(scalconj_binary_type<T1, T2>(lit_type(conj(static_cast<T1>(a.left()))), conj_type<T2>(conj_unary_type<T2>(a.right()), a.shape())), a.shape());
    }

    // conjugation of scalar * conj(tensor)
    template <typename T1, typename T2>
    scal_type<T1, T2> conj(const scalconj_type<T1, T2> &a)
    {
        using std::conj;
        using lit_type = expression_templates::literal_type<T1, typename traits<T2>::backend_type>;
        return scal_type<T1, T2>(scal_binary_type<T1, T2>(lit_type(conj(static_cast<T1>(a.left()))), a.right().obj()), a.shape());
    }

    // conjugation of conjugate
    template <template <typename, size_t, typename> class c1, typename T1, size_t D, typename backend>
    typename conj_unary_type<c1<T1, D, backend>>::object_type conj(const conj_type<c1<T1, D, backend>> &a) { return a.obj(); }

    template <template <typename, size_t, typename> class c1, typename T, typename T1, size_t D, size_t D1, typename backend>
    typename conj_unary_type<tensor_slice<c1<T, D1, backend>, T1, D>>::object_type conj(const conj_type<tensor_slice<c1<T, D1, backend>, T1, D>> &a) { return a.obj(); }

    // conjugate of transpose operator
    template <typename array_type, bool conjugate>
    trans_type<array_type, !conjugate> conj(const trans_type<array_type, conjugate> &a)
    {
        using std::conj;
        return trans_type<array_type, !conjugate>(a.matrix(), conj(a.coeff()));
    }

    // conjugate of tensor permutation operator
    template <typename array_type, bool conjugate>
    perm3_type<array_type, !conjugate> conj(const perm3_type<array_type, conjugate> &a)
    {
        using std::conj;
        return perm3_type<array_type, !conjugate>(a.tensor(), a.permutation_index(), conj(a.coeff()));
    }

    // conjugate of matrix matrix product
    template <typename T1, typename T2>
    gemm_return_type<T1, T2> conj(const gemm_type<T1, T2> &r) { return gemm_type<T1, T2>(r, false, true); }

    template <typename T1, typename T2>
    gemv_return_type<T1, T2> conj(const gemv_type<T1, T2> &r) { return gemv_type<T1, T2>(r, true); }

}

#endif // PYTTN_LINALG_ALGEBRA_OVERLOADS_CONJUGATION_HPP_
