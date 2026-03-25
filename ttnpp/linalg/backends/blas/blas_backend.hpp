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

#ifndef PYTTN_LINALG_BACKENDS_BLAS_BACKEND_HPP_
#define PYTTN_LINALG_BACKENDS_BLAS_BACKEND_HPP_

#include "../../linalg_forward_decl.hpp"
#include "../../utils/linalg_utils.hpp"
#include "../backend.hpp"

#ifdef USE_MKL
#define blas_set_num_threads(x) mkl_set_num_threads(x)
#else
#define blas_set_num_threads(x)
#endif

#ifdef USE_LIBXSMM
#include <libxsmm.h>
#endif

#include "extended_blas_functions.hpp"
#include "blas_wrapper.hpp"
#include "mkl_wrapper.hpp"
#include "lapack_wrapper.hpp"
#include <random>
#include <vector>
#include <tuple>
#include <algorithm>

// fix up the const correctness here
namespace linalg
{
    class blas_backend : public backend_base
    {
    public:
        using size_type = std::size_t;
        using blas_int_type = blas::blas_int_type;
        using int_type = blas_int_type;
        using index_type = blas_int_type;
#ifndef USE_MKL
        using select_type = bool;
#else
        using select_type = blas_int_type;
#endif

        static inline std::string label() { return std::string("blas"); }

    protected:
        static constexpr size_type default_nthreads = 1;
        static constexpr bool default_batchpar = false;

        static void synchronise(){}
        static size_type &nthreads()
        {
            static size_type _nthreads;
            return _nthreads;
        }

        static bool &batchpar()
        {
            static bool _batchpar;
            return _batchpar;
        }

    public:
        static void initialise()
        {
            initialise(default_nthreads, default_batchpar);
            // nthreads() = default_nthreads;
            // batchpar() = default_batchpar;
        }
        static void initialise(size_type _nthreads, bool _batchpar)
        {
#ifdef USE_LIBXSMM
            libxsmm_init(void);
#endif
            set_num_threads(_nthreads);
            batchpar() = _batchpar;
        }

        static void set_num_threads(size_type _nthreads)
        {
            nthreads() = _nthreads;
            blas_set_num_threads(_nthreads);
        }
        static size_type get_num_threads() { return nthreads(); }

        static bool is_initialised() { return true; }

        static void destroy()
        {
#ifdef USE_LIBXSMM
            libxsmm_finalize(void);
#endif
        }

    public:
        using transform_type = char;
        static constexpr transform_type op_n = 'N';
        static constexpr transform_type op_t = 'T';
        static constexpr transform_type op_h = 'C';
        static constexpr transform_type op_c = 'I';

    };

    template <>
    struct traits<blas_backend>
    {
        using size_type = std::size_t;
        using int_type = blas::blas_int_type;
        using index_type = int_type;
        using transform_type = char;
        static inline std::string label() { return std::string("blas"); }
    };

} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_BLAS_BACKEND_HPP_
