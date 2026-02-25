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

#ifndef PYTTN_LINALG_UTILS_GENRANDOM_HPP_
#define PYTTN_LINALG_UTILS_GENRANDOM_HPP_

#include <random>
#include "../linalg_forward_decl.hpp"
#include "../linalg_type_traits.hpp"

namespace linalg
{
    template <typename backend>
    class random_engine
    {
    public:

        // constructors for random number generation
        random_engine();
        template <typename I>
        random_engine(I seed);
        random_engine(const random_engine &o);
        random_engine(random_engine &&o);
        ~random_engine();
        random_engine &operator=(const random_engine &o);
        random_engine &operator=(random_engine &&o);

        bool active() const;
        std::uint64_t ngenerated() const ;
        unsigned long long seed() const ;

        template <typename I>
        void set_seed(I seed);

        template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, backend>::value, void>::type>
        void fill_normal(ArrType &array);

        template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, backend>::value, void>::type>
        void fill_normal(ArrType &&array);

    };

    template <>
    class random_engine<blas_backend>
    {
    protected:
        template <typename T, typename = void>
        class random_normal;

        template <typename T>
        class random_normal<T, typename std::enable_if<not is_complex<T>::value, void>::type>
        {
        public:
            using real_type = T;

        public:
            static inline T generate(std::mt19937 &rng, std::normal_distribution<T> &dist)
            {
                return dist(rng);
            }
        };

        template <typename T>
        class random_normal<T, typename std::enable_if<is_complex<T>::value, void>::type>
        {
        public:
            using real_type = typename get_real_type<T>::type;

        public:
            static inline T generate(std::mt19937 &rng, std::normal_distribution<real_type> &dist)
            {
                real_type div = 1.0 / std::sqrt(2.0);
                real_type a = dist(rng) / div;
                real_type b = dist(rng) / div;
                return T(a, b);
            }
        };

    public:
        // constructors for random number generation
        random_engine() {}
        template <typename sseq>
        random_engine(sseq &seed) { m_rng.seed(seed); }
        random_engine(const random_engine &o) = default;
        random_engine(random_engine &&o) = default;
        random_engine &operator=(const random_engine &o) = default;
        random_engine &operator=(random_engine &&o) = default;

        template <typename sseq>
        void set_seed(sseq &seed) { m_rng.seed(seed); }

        template <typename T>
        T generate_normal()
        {
            using real_type = typename get_real_type<T>::type;
            std::normal_distribution<real_type> dist(0, 1);

            return random_normal<T>::generate(m_rng, dist);
        }

        template <typename T>
        void fill_normal(std::vector<T> &arr)
        {
            using real_type = typename get_real_type<T>::type;
            std::normal_distribution<real_type> dist(0, 1);

            for (size_t i = 0; i < arr.size(); ++i)
            {
                arr[i] = random_normal<T>::generate(m_rng, dist);
            }
        }

        template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, blas_backend>::value, void>::type>
        void fill_normal(ArrType &arr)
        {
            using T = typename traits<ArrType>::value_type;
            using real_type = typename get_real_type<T>::type;
            std::normal_distribution<real_type> dist(0, 1);

            for (size_t i = 0; i < arr.size(); ++i)
            {
                arr(i) = random_normal<T>::generate(m_rng, dist);
            }
        }

        template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, blas_backend>::value, void>::type>
        void fill_normal(ArrType &&arr)
        {
            using T = typename traits<ArrType>::value_type;
            using real_type = typename get_real_type<T>::type;
            std::normal_distribution<real_type> dist(0, 1);

            for (size_t i = 0; i < arr.size(); ++i)
            {
                arr(i) = random_normal<T>::generate(m_rng, dist);
            }
        }

        const std::mt19937 &rng() const { return m_rng; }
        std::mt19937 &rng() { return m_rng; }

    protected:
        std::mt19937 m_rng;
    };
} // namespace linalg

#endif // PYTTN_LINALG_UTILS_GENRANDOM_HPP_
