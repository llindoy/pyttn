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

#ifndef PYTTN_LINALG_ALGEBRA_OVERLOADS_REAL_IMAG_HPP_
#define PYTTN_LINALG_ALGEBRA_OVERLOADS_REAL_IMAG_HPP_

namespace linalg
{

    template <typename T>
    real_return_type<T> real(const T &a) { return real_type<T>(real_unary_type<T>(a), a.shape()); }
    template <typename T>
    imag_return_type<T> imag(const T &a) { return imag_type<T>(imag_unary_type<T>(a), a.shape()); }
    template <typename T>
    norm_return_type<T> elementwise_norm(const T &a) { return norm_type<T>(norm_unary_type<T>(a), a.shape()); }
    template <typename T>
    arg_return_type<T> elementwise_arg(const T &a) { return arg_type<T>(arg_unary_type<T>(a), a.shape()); }
    template <typename T>
    unit_polar_return_type<T> unit_polar(const T &a) { return unit_polar_type<T>(unit_polar_unary_type<T>(a), a.shape()); }

} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_OVERLOADS_REAL_IMAG_HPP_//
