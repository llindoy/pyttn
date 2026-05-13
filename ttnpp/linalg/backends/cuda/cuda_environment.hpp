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

#ifndef PYTTN_LINALG_BACKENDS_CUDA_ENVIRONMENT_HPP_
#define PYTTN_LINALG_BACKENDS_CUDA_ENVIRONMENT_HPP_

#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <array>

namespace linalg
{
    class cuda_backend;

    // class wrapping the cuda environment.  This allows for easy setup of devices and allocation of global workspace variables
    // that are required for performing linear algebra operations.
    class cuda_environment
    {
    public:
        using size_type = std::size_t;
        using index_type = int32_t;
        // this is a friend of the cuda_backend type
        friend class cuda_backend;

        friend std::ostream &operator<<(std::ostream &out, const cuda_environment &s);

    public:
        cuda_environment() ;
        cuda_environment(int device_id, int nstreams) ;
        cuda_environment(cuda_environment &&other);
        ~cuda_environment();

        cuda_environment &operator=(cuda_environment &&other) noexcept;

        // functions for updating the state of the cuda_environment
        void init(int device_id, int nstreams = 1);
        void destroy();

        bool is_initialised() const;

        void* current_stream() const;
        void increment_stream_id();
        void reset_stream_id();

        // accessors device specific properties required for determining kernel execution parameters
        size_type total_global_memory() const;
        size_type shared_mem_per_block() const;
        int warpsize() const;
        int maximum_threads_per_block() const;
        std::array<int, 3> maximum_dimensions_threads_per_block() const;

        const void* cublas_handle() const;
        const void* cusolver_dn_handle() const;
        const void* cusparse_handle() const;
        const void* cutensor_handle() const;

        void set_device() const;

        // accessors for general properties of the cuda install that do not require a cuda_environment instance
        static int number_of_devices();

        static std::ostream &list_devices(std::ostream &out);
    
    protected:
        struct impl;
        impl* m_impl;
    };

    std::ostream &operator<<(std::ostream &out, const cuda_environment &inst);

} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_CUDA_ENVIRONMENT_HPP_//
