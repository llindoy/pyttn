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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_EIGENSOLVERS_EIGENSOLVER_HERMITIAN_CUH_
#define PYTTN_LINALG_DECOMPOSITIONS_EIGENSOLVERS_EIGENSOLVER_HERMITIAN_CUH_

#include "eigensolver_base.hpp"
#include "eigensolver_hermitian.hpp"
#include "../../backends/cuda/cuda_backend.hpp"
#include "../../backends/cuda/cusolver_wrapper.cuh"

namespace linalg
{
    namespace internal
    {
        template <typename T>
        struct hermitian_eigensolver_helper<T, cuda_backend>
        {
            static_assert(is_number<T>::value, "Failed to initialise hermitian eigensolver working space object.");
            using size_type = typename traits<cuda_backend>::size_type;
            using int_type = typename traits<cuda_backend>::int_type;

            struct additional_working
            {
                tensor<int, 1, cuda_backend> m_gpu_info;
                tensor<int, 1> m_cpu_info;
                void resize(size_type n)
                {
                    if (m_gpu_info.size() == 0)
                    {
                        m_gpu_info.resize(1);
                        m_cpu_info.resize(1);
                    }
                }
                void clear() {}
            };

            static inline void call(const char JOBZ, const char UPLO, const int_type N, T *A, const int_type LDA, typename get_real_type<T>::type *W, T *WORK, const int_type LWORK, additional_working &working)
            {
                int_type n = N;
                int_type lda = LDA;
                int_type lwork = LWORK;
                eig_mode jobz;
                CALL_AND_RETHROW(jobz = get_jobz(JOBZ));
                fill_mode uplo;
                CALL_AND_RETHROW(uplo = get_uplo(UPLO));

                CALL_AND_RETHROW(backend_algebra<cuda_backend>::heev(jobz, uplo, n, A, lda, W, WORK, lwork, working.m_gpu_info.buffer());)
                working.m_cpu_info = working.m_gpu_info;
                CALL_AND_RETHROW(cusolver::heev_error_handling(working.m_cpu_info(0), 'a'));
            }

            static inline int_type query_worksize(const char JOBZ, const char UPLO, const int_type N, T *A, const int_type LDA, typename get_real_type<T>::type *W, additional_working & /* working */)
            {
                int_type n = N;
                int_type lda = LDA;
                eig_mode jobz;
                CALL_AND_RETHROW(jobz = get_jobz(JOBZ));
                fill_mode uplo;
                CALL_AND_RETHROW(uplo = get_uplo(UPLO));
                int_type worksize;
                CALL_AND_RETHROW(backend_algebra<cuda_backend>::heev_buffersize(jobz, uplo, n, A, lda, W, &worksize);)
                return worksize;
            }

            static inline eig_mode get_jobz(const char JOBZ)
            {
                switch (JOBZ)
                {
                case ('N'):
                {
                    return eig_mode::no_vectors;
                }
                case ('V'):
                {
                    return eig_mode::vectors;
                }
                default:
                {
                    RAISE_EXCEPTION("Invalid JOBZ argument.");
                }
                };
            }

            static inline fill_mode get_uplo(const char UPLO)
            {
                switch (UPLO)
                {
                case ('U'):
                {
                    return fill_mode::upper;
                }
                case ('L'):
                {
                    return fill_mode::lower;
                }
                default:
                {
                    RAISE_EXCEPTION("Invalid UPLO argument.");
                }
                };
            }
        };
    }   //namespace internal
} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_EIGENSOLVERS_EIGENSOLVER_HERMITIAN_CUH_//
