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

#ifndef PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_MATRIX_OPERATORS_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_MATRIX_OPERATORS_HPP_

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"

namespace ttns
{
    namespace ops
    {

        template <typename T, typename backend = linalg::blas_backend>
        class dense_matrix_operator : public primitive<T, backend>
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
            dense_matrix_operator() : base_type() {}
            template <typename... Args>
            dense_matrix_operator(Args &&...args)
            try : base_type(), m_operator(std::forward<Args>(args)...)
            {
                ASSERT(
                    m_operator.shape(0) == m_operator.shape(1),
                    "Failed to construct dense operator. The operator to be bound must be a square matrix.");
                base_type::m_size = m_operator.shape(0);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
            }

            dense_matrix_operator(const dense_matrix_operator &o) = default;
            dense_matrix_operator(dense_matrix_operator &&o) = default;

            dense_matrix_operator &operator=(const dense_matrix_operator &o) = default;
            dense_matrix_operator &operator=(dense_matrix_operator &&o) = default;

            bool is_resizable() const final { return false; }
            void resize(size_type /*n*/) { ASSERT(false, "This shouldn't be called."); }

            std::shared_ptr<base_type> clone() const { return std::make_shared<dense_matrix_operator>(m_operator); }
            std::shared_ptr<base_type> transpose() const { return std::make_shared<dense_matrix_operator>(linalg::trans(m_operator)); }
            std::shared_ptr<dense_matrix_operator> transpose_matrix() const { return std::make_shared<dense_matrix_operator>(linalg::trans(m_operator)); }

            linalg::matrix<T> todense() const
            {
                linalg::matrix<T> ret(m_operator);
                return ret;
            }

            void apply(const resview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const resview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            void apply(const_matrix_ref A, matrix_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            // functions for applying the operator to rank 3 tensor views
            void apply(const restensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const restensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }

        protected:
            template <typename A1, typename A2>
            void apply_rank_2(const A1 &A, A2 &HA)
            {
                CALL_AND_HANDLE(HA = m_operator * A, "Failed to apply dense matrix operator.  Failed to rank 2 tensor.");
            }
            template <typename A1, typename A2>
            void apply_rank_3(const A1 &A, A2 &HA)
            {
                CALL_AND_HANDLE(HA = linalg::contract(m_operator, 1, A, 1), "Failed to apply dense matrix operator to rank 3 tensor.");
            }

        public:
            void update(real_type /*t*/, real_type /*dt*/) final {}
            const matrix_type &mat() const { return m_operator; }

            std::string to_string() const final
            {
                std::stringstream oss;
                oss << "dense matrix operator: " << std::endl;
                oss << m_operator << std::endl;
                return oss.str();
            }

        protected:
            matrix_type m_operator;

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void save(archive &ar) const
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise dense_matrix operator object.  Error when serialising the base object.");

                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("matrix", m_operator)),
                    "Failed to serialise dense_matrix operator object.  Error when serialising the matrix.");
            }

            template <typename archive>
            void load(archive &ar)
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise dense_matrix operator object.  Error when serialising the base object.");
                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("matrix", m_operator)),
                    "Failed to serialise dense_matrix operator object.  Error when serialising the matrix.");
            }
#endif
        };

        template <typename T, typename backend>
        class csr_matrix_rank_3_op;

        template <typename T>
        class csr_matrix_rank_3_op<T, linalg::blas_backend>
        {
        public:
            template <typename At1, typename At2>
            static inline void apply(const linalg::csr_matrix<T, linalg::blas_backend> &m_operator, const At1 &A, At2 &HA)
            {
                for (size_t i = 0; i < A.shape(0); ++i)
                {
                    CALL_AND_HANDLE(HA[i] = m_operator * A[i], "Failed to apply sparse matrix operator.");
                }
            }
        };

#ifdef PYTTN_BUILD_CUDA
        template <typename T>
        class csr_matrix_rank_3_op<T, linalg::cuda_backend>
        {
        public:
            template <typename At1, typename At2>
            static inline void apply(const linalg::csr_matrix<T, linalg::cuda_backend> &m_operator, const At1 &A, At2 &HA)
            {
                // TODO: Parallelise this function over cuda streams.
                for (size_t i = 0; i < A.shape(0); ++i)
                {
                    CALL_AND_HANDLE(HA[i] = m_operator * A[i], "Failed to apply sparse matrix operator.");
                }

                // CALL_AND_HANDLE
                //(
                //     linalg::cuda_backend::async_for(0, A.shape(0), [&A, &HA, &m_operator](size_t i){CALL_AND_HANDLE(HA[i] = m_operator*A[i], "Failed to apply sparse matrix operator.");}),
                //     "Error when applying rank 3 applicative."
                //);
            }
        };
#endif

        // A class for wrapping the multiplication by a sparse matrix.  This stores the matrix in csr form to make it easy to perform all of the required operations.
        template <typename T, typename backend = linalg::blas_backend>
        class sparse_matrix_operator : public primitive<T, backend>
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
            sparse_matrix_operator() : base_type() {}
            sparse_matrix_operator(const sparse_matrix_operator &o) = default;
            sparse_matrix_operator(sparse_matrix_operator &&o) = default;
            template <typename... Args>
            sparse_matrix_operator(Args &&...args)
            try : base_type(), m_operator(std::forward<Args>(args)...)
            {
                ASSERT(m_operator.dims(0) == m_operator.dims(1), "The operator to be bound must be a square matrix.");
                base_type::m_size = m_operator.dims(0);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct sparse matrix operator object.");
            }

            bool is_resizable() const final { return false; }
            void resize(size_type /*n*/) { ASSERT(false, "This shouldn't be called."); }

            linalg::matrix<T> todense() const
            {
                return m_operator.todense();
            }

            std::shared_ptr<base_type> clone() const { return std::make_shared<sparse_matrix_operator>(m_operator); }
            std::shared_ptr<base_type> transpose() const
            {
                linalg::csr_matrix<T, backend> mat;
                m_operator.transpose(mat);
                return std::make_shared<sparse_matrix_operator>(mat);
            }

            void apply(const resview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const resview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            void apply(const_matrix_ref A, matrix_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            // functions for applying the operator to rank 3 tensor views
            void apply(const restensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const restensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }

        protected:
            template <typename A1, typename A2>
            void apply_rank_2(const A1 &A, A2 &HA)
            {
                ASSERT(this->size() == A.shape(0), "Failed to apply rank 2 contraction.");
                CALL_AND_HANDLE(HA = m_operator * A, "Failed to apply sparse matrix operator.  Failed to rank 2 tensor.");
            }
            template <typename At1, typename At2>
            void apply_rank_3(const At1 &A, At2 &HA)
            {
                using applicative = csr_matrix_rank_3_op<T, backend>;
                ASSERT(this->size() == A.shape(1), "Failed to apply rank 3 contraction.");
                CALL_AND_HANDLE(applicative::apply(m_operator, A, HA), "Failed to apply rank 3 contraction.");
            }

        public:
            void update(real_type /*t*/, real_type /*dt*/) final {}
            const linalg::csr_matrix<T, backend> &mat() const { return m_operator; }

            std::string to_string() const final
            {
                std::stringstream oss;
                oss << "sparse matrix operator: " << std::endl;
                oss << m_operator << std::endl;
                return oss.str();
            }

        protected:
            linalg::csr_matrix<T, backend> m_operator;
#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void save(archive &ar) const
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise sparse_matrix operator object.  Error when serialising the base object.");
                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("matrix", m_operator)),
                    "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
            }

            template <typename archive>
            void load(archive &ar)
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise sparse_matrix operator object.  Error when serialising the base object.");

                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("matrix", m_operator)),
                    "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
            }
#endif
        };

        template <typename T, typename backend = linalg::blas_backend>
        class diagonal_matrix_operator : public primitive<T, backend>
        {
        public:
            using base_type = primitive<T, backend>;

            // use the parent type aliases
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
            diagonal_matrix_operator() : base_type() {}
            template <typename... Args>
            diagonal_matrix_operator(Args &&...args)
            try : base_type(), m_operator(std::forward<Args>(args)...)
            {
                ASSERT(
                    m_operator.shape(0) == m_operator.shape(1),
                    "The operator to be bound must be a square matrix.");
                base_type::m_size = m_operator.shape(0);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct diagonal matrix operator object.");
            }

            diagonal_matrix_operator(const diagonal_matrix_operator &o) = default;
            diagonal_matrix_operator(diagonal_matrix_operator &&o) = default;

            diagonal_matrix_operator &operator=(const diagonal_matrix_operator &o) = default;
            diagonal_matrix_operator &operator=(diagonal_matrix_operator &&o) = default;

            bool is_resizable() const final { return false; }
            void resize(size_type /* n */) { ASSERT(false, "This shouldn't be called."); }

            std::shared_ptr<base_type> clone() const { return std::make_shared<diagonal_matrix_operator>(m_operator); }
            std::shared_ptr<base_type> transpose() const { return std::make_shared<diagonal_matrix_operator>(m_operator); }

            linalg::matrix<T> todense() const
            {
                return m_operator.todense();
            }

            void apply(const resview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const resview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            void apply(const_matrix_ref A, matrix_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            // functions for applying the operator to rank 3 tensor views
            void apply(const restensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const restensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }

        protected:
            template <typename A1, typename A2>
            void apply_rank_2(const A1 &A, A2 &HA)
            {
                CALL_AND_HANDLE(HA = m_operator * A, "Failed to apply dense matrix operator.  Failed to rank 2 tensor.");
            }
            template <typename At1, typename At2>
            void apply_rank_3(const At1 &A, At2 &HA)
            {
                for (size_t i = 0; i < A.shape(0); ++i)
                {
                    CALL_AND_RETHROW(HA[i] = m_operator * A[i]);
                }
            }

        public:
            void update(real_type /*t*/, real_type /*dt*/) final {}
            const linalg::diagonal_matrix<T, backend> &mat() const { return m_operator; }

            std::string to_string() const final
            {
                std::stringstream oss;
                oss << "diagonal matrix operator: " << std::endl;
                oss << m_operator << std::endl;
                return oss.str();
            }

        protected:
            linalg::diagonal_matrix<T, backend> m_operator;

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void save(archive &ar) const
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise diagonal_matrix operator object.  Error when serialising the base object.");

                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("matrix", m_operator)),
                    "Failed to serialise diagonal_matrix operator object.  Error when serialising the matrix.");
            }

            template <typename archive>
            void load(archive &ar)
            {
                CALL_AND_HANDLE(
                    ar(cereal::base_class<primitive<T, backend>>(this)),
                    "Failed to serialise diagonal_matrix operator object.  Error when serialising the base object.");

                CALL_AND_HANDLE(
                    ar(cereal::make_nvp("matrix", m_operator)),
                    "Failed to serialise diagonal_matrix operator object.  Error when serialising the matrix.");
            }
#endif
        };

    } // namespace ops
} // namespace ttns

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::dense_matrix_operator, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::sparse_matrix_operator, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::diagonal_matrix_operator, ttns::ops::primitive)
#endif

#endif // PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_MATRIX_OPERATORS_HPP_
