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

#ifndef PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_
#define PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_

#include "../../linalg_forward_decl.hpp"
#include "../../utils/linalg_utils.hpp"
#include "../../../common/exception_handling.hpp"
#include "../backend.hpp"
#include "cuda_environment.hpp"

#include <array>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <type_traits>
#include <string>

namespace linalg
{
    enum class cuda_transform_type : uint8_t {n, t, h, c};
    enum class eig_mode : uint8_t {no_vectors, vectors};
    enum class fill_mode : uint8_t {upper, lower};


    class cuda_backend : public backend_base
    {
    public:
        using size_type = typename cuda_environment::size_type;
        using index_type = typename cuda_environment::index_type;
        using int_type = index_type;
        using transform_type = cuda_transform_type;
        static constexpr transform_type op_n = cuda_transform_type::n;
        static constexpr transform_type op_t = cuda_transform_type::t;
        static constexpr transform_type op_h = cuda_transform_type::h;
        static constexpr transform_type op_c = cuda_transform_type::c;

    protected:
        static void clean_up_ones();
        static void initialise_empty_ones_buffers();
        
    public:
        /*
         * Functions for handling environment properties of the backend
         */
        static cuda_environment &environment();
        static bool is_initialised();

        // the initialisation routines for the cuda_backend are not thread safe.
        static void initialise(cuda_environment &&env);
        static void initialise(size_type device_id = 0, size_type nstreams = 1);
        static void destroy();
        static void synchronise();
        static void set_cublas_stream();

        static std::ostream &device_properties(std::ostream &out);
    }; // cuda_backend

    template <>
    struct traits<cuda_backend>
    {
        using size_type = typename cuda_environment::size_type;
        using index_type = typename cuda_environment::index_type;
        using int_type = index_type;
        using transform_type = cuda_transform_type;
        static inline std::string label() { return std::string("cuda"); }
    };

} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_//
