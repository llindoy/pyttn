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

#ifndef PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_PRIMITIVE_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_PRIMITIVE_OPERATOR_HPP_

#include <complex>
#include <string>
#include <memory>

#include <linalg/linalg.hpp>
#include <common/tmp_funcs.hpp>

#include "serialisation_helper.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/portable_binary.hpp>
#endif

namespace ttns
{
    namespace ops
    {

        template <typename T, typename backend = linalg::blas_backend>
        class primitive
        {
        public:
            using vector_type = linalg::vector<T, backend>;
            using matrix_type = linalg::matrix<T, backend>;
            using size_type = typename backend::size_type;
            using matrix_ref = matrix_type &;
            using const_matrix_ref = const matrix_type &;
            using vector_ref = vector_type &;
            using const_vector_ref = const vector_type &;
            using real_type = typename tmp::get_real_type<T>::type;

            // objects needed for applying operators to views
            using matview = linalg::reinterpreted_tensor<const T, 2, backend>;
            using resview = linalg::reinterpreted_tensor<T, 2, backend>;
            using tensview = linalg::reinterpreted_tensor<const T, 3, backend>;
            using restensview = linalg::reinterpreted_tensor<T, 3, backend>;

        protected:
            size_type m_size;
            bool m_is_identity;

        public:
            primitive() : m_size(0), m_is_identity(false) {}
            primitive(size_type _size) : m_size(_size), m_is_identity(false) {}
            primitive(size_type _size, bool _is_identity) : m_size(_size), m_is_identity(_is_identity) {}
            primitive(const primitive &o) = default;
            primitive(primitive &&o) = default;
            virtual ~primitive() {}

            primitive &operator=(const primitive &o) = default;
            primitive &operator=(primitive &&o) = default;

            virtual std::shared_ptr<primitive> transpose() const = 0;

            virtual linalg::matrix<T> todense() const = 0;

            // apply to matrix
            virtual void apply(const_matrix_ref A, matrix_ref HA) = 0;
            virtual void apply(const_matrix_ref A, matrix_ref HA, real_type t, real_type dt) = 0;
            virtual void apply(const_vector_ref A, vector_ref HA) = 0;
            virtual void apply(const_vector_ref A, vector_ref HA, real_type t, real_type dt) = 0;

            // apply to rank 2 views
            virtual void apply(const resview &A, resview &HA) = 0;
            virtual void apply(const resview &A, resview &HA, real_type t, real_type dt) = 0;
            virtual void apply(const matview &A, resview &HA) = 0;
            virtual void apply(const matview &A, resview &HA, real_type t, real_type dt) = 0;

            // apply to rank 3 views
            virtual void apply(const restensview &A, restensview &HA) = 0;
            virtual void apply(const restensview &A, restensview &HA, real_type t, real_type dt) = 0;
            virtual void apply(const tensview &A, restensview &HA) = 0;
            virtual void apply(const tensview &A, restensview &HA, real_type t, real_type dt) = 0;

            // function for allowing you to update time-dependent Hamiltonians
            virtual void update(real_type t, real_type dt) = 0;

            virtual bool is_resizable() const { return true; }
            virtual void resize(size_type n) { m_size = n; }
            virtual std::shared_ptr<primitive> clone() const = 0;
            virtual std::string to_string() const = 0;

            size_type size() const { return m_size; }
            bool is_identity() const { return m_is_identity; }

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void save(archive &ar) const
            {
                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("size", m_size)),
                    "Failed to primitive operator.  Failed to serialise its size.");

                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("is_identity", m_is_identity)),
                    "Failed to primitive operator.  Failed to serialise whether or not it is the identity operator.");
            }

            template <typename archive>
            void load(archive &ar)
            {
                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("size", m_size)),
                    "Failed to primitive operator.  Failed to serialise its size.");
                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("is_identity", m_is_identity)),
                    "Failed to primitive operator.  Failed to serialise whether or not it is the identity operator.");
            }
#endif
        };

        // implementation of the identity ops ops
        template <typename T, typename backend = linalg::blas_backend>
        class identity final : public primitive<T, backend>
        {
        public:
            using base_type = primitive<T, backend>;

            // use the parent class type aliases
            using typename base_type::const_matrix_ref;
            using typename base_type::const_vector_ref;
            using typename base_type::matrix_ref;
            using typename base_type::matrix_type;
            using typename base_type::matview;
            using typename base_type::real_type;
            using typename base_type::restensview;
            using typename base_type::resview;
            using typename base_type::size_type;
            using typename base_type::tensview;
            using typename base_type::vector_ref;
            using typename base_type::vector_type;

        public:
            identity() : base_type() {}
            identity(size_type size) : base_type(size, true) {}
            identity(const identity &o) = default;
            identity(identity &&o) = default;
            ~identity() {}

            std::shared_ptr<base_type> transpose() const { return std::make_shared<identity>(base_type::m_size); }

            linalg::matrix<T> todense() const
            {
                linalg::matrix<T> ret(base_type::m_size, base_type::m_size, [](size_type i, size_type j)
                                      { return i == j ? T(1) : T(0); });
                return ret;
            }

            // apply to matrices
            void apply(const_matrix_ref A, matrix_ref HA) final { HA = A; }
            void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final { HA = A; }
            void apply(const_vector_ref A, vector_ref HA) final { HA = A; }
            void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final { HA = A; }

            // apply to rank 2 views
            void apply(const resview &A, resview &HA) final { HA = A; }
            void apply(const resview &A, resview &HA, real_type /* t */, real_type /* dt */) final { HA = A; }
            void apply(const matview &A, resview &HA) final { HA = A; }
            void apply(const matview &A, resview &HA, real_type /* t */, real_type /* dt */) final { HA = A; }

            // functions for applying the operator to rank 3 tensor views
            void apply(const restensview &A, restensview &HA) final { HA = A; }
            void apply(const restensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { HA = A; }
            void apply(const tensview &A, restensview &HA) final { HA = A; }
            void apply(const tensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { HA = A; }

            void update(real_type /* t */, real_type /* dt */) final{};

            std::shared_ptr<base_type> clone() const { return std::make_shared<identity>(base_type::m_size); }
            std::string to_string() const final { return std::string("I_") + std::to_string(this->m_size); }

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void save(archive &ar) const
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise identity operator object.  Error when serialising the base object.");
            }

            template <typename archive>
            void load(archive &ar)
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise identity operator object.  Error when serialising the base object.");
            }
#endif
        };

    } // namespace ops
} // namespace ttns

#ifdef CEREAL_LIBRARY_FOUND
#ifdef PYTTN_BUILD_CUDA
#define SERIALIZE_CUDA_TYPES
#endif
TTNS_REGISTER_SERIALIZATION(ttns::ops::identity, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::test_operator, ttns::ops::primitive)
#endif

#endif // PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_PRIMITIVE_OPERATOR_HPP_//
