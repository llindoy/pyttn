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

#ifndef PYTTN_LINALG_UTILS_GENRANDOM_CUH_
#define PYTTN_LINALG_UTILS_GENRANDOM_CUH_

#include "../linalg_forward_decl.hpp"
#include "genrandom.hpp"

#include <curand.h>
#include "../backends/cuda/curand_wrapper.cuh"
#include "../backends/cuda/cuda_backend.hpp"


namespace linalg
{
    template <typename T>
    struct generate_norm_vec;

    template <>
    struct generate_norm_vec<float>
    {
        static inline uint64_t generate(curandGenerator_t gen, float *buffer, size_t n)
        {
            float mean = 0;
            float stdev = 1;
            curand_safe_call(curandGenerateNormal(gen, buffer, n, mean, stdev));
            return n;
        }
    };

    template <>
    struct generate_norm_vec<double>
    {
        static inline uint64_t generate(curandGenerator_t gen, double *buffer, size_t n)
        {
            double mean = 0;
            double stdev = 1;
            curand_safe_call(curandGenerateNormalDouble(gen, buffer, n, mean, stdev));
            return n;
        }
    };

    // in order to generate normal distributed complex numbers we generate 2 times as many real numbers
    // with a standard deviation that is sqrt(2) smaller.
    template <>
    struct generate_norm_vec<std::complex<float>>
    {
        static inline uint64_t generate(curandGenerator_t gen, std::complex<float> *buffer, size_t n)
        {
            float mean = 0;
            float stdev = 1 / std::sqrt(2.0);
            curand_safe_call(curandGenerateNormal(gen, reinterpret_cast<float *>(buffer), 2 * n, mean, stdev));
            return 2 * n;
        }
    };

    template <>
    struct generate_norm_vec<std::complex<double>>
    {
        static inline uint64_t generate(curandGenerator_t gen, std::complex<double> *buffer, size_t n)
        {
            double mean = 0;
            double stdev = 1 / std::sqrt(2.0);
            curand_safe_call(curandGenerateNormalDouble(gen, reinterpret_cast<double *>(buffer), 2 * n, mean, stdev));
            return 2 * n;
        }
    };

    template <>
    class random_engine<cuda_backend>
    {
    public:
        // constructors for random number generation
        random_engine() : m_active(false) { initialise(); }
        template <typename I>
        random_engine(I seed) : m_active(false)
        {
            initialise();
            set_seed(seed);
        }

        random_engine(const random_engine &o)
        {
            if (o.active())
            {
                m_active = false;
                initialise();
            }
            set_seed(o.seed());
            m_ngenerated = o.ngenerated();
            m_active = o.active();
        }
        random_engine(random_engine &&o)
        {
            if (o.active())
            {
                m_active = false;
                initialise();
            }
            set_seed(o.seed());
            m_ngenerated = o.ngenerated();
            m_active = o.active();

            o.clear();
        }
        ~random_engine() { clear(); }
        random_engine &operator=(const random_engine &o)
        {
            clear();
            if (o.active())
            {
                m_active = false;
                initialise();
            }
            set_seed(o.seed());
            m_ngenerated = o.ngenerated();
            m_active = o.active();
            return *this;
        }
        random_engine &operator=(random_engine &&o)
        {
            clear();
            if (o.active())
            {
                m_active = false;
                initialise();
            }
            set_seed(o.seed());
            m_ngenerated = o.ngenerated();
            m_active = o.active();

            o.clear();
            return *this;
        }

        bool active() const { return m_active; }
        std::uint64_t ngenerated() const { return m_ngenerated; }
        unsigned long long seed() const { return m_seed; }

        template <typename I>
        void set_seed(I seed)
        {
            ASSERT(m_active, "Cannot set seed of inactive pseudo random number generator.");
            m_seed = seed;
            curand_safe_call(curandSetPseudoRandomGeneratorSeed(m_gen, m_seed));
        }

        template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, cuda_backend>::value, void>::type>
        void fill_normal(ArrType &array)
        {
            using T = typename traits<ArrType>::value_type;
            m_ngenerated += generate_norm_vec<T>::generate(m_gen, array.buffer(), array.size());
        }

        template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, cuda_backend>::value, void>::type>
        void fill_normal(ArrType &&array)
        {
            using T = typename traits<ArrType>::value_type;
            m_ngenerated += generate_norm_vec<T>::generate(m_gen, array.buffer(), array.size());
        }

    protected:
        void initialise()
        {
            if (!m_active)
            {
                m_ngenerated = 0;
                m_active = true;
                curand_safe_call(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT));
            }
        }

        void clear()
        {
            if (m_active)
            {
                curand_safe_call(curandDestroyGenerator(m_gen));
            }
            m_active = false;
            m_ngenerated = 0;
            m_seed = 0;
        }

    protected:
        curandGenerator_t m_gen;
        bool m_active;
        std::uint64_t m_ngenerated;
        unsigned long long m_seed;
    };

} // namespace linalg

#endif // PYTTN_LINALG_UTILS_GENRANDOM_CUH_
