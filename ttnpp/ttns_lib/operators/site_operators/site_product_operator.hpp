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

#ifndef PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SITE_PRODUCT_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SITE_PRODUCT_OPERATOR_HPP_

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"
#include "../kron.hpp"

namespace ttns
{
    namespace ops
    {

        // need to implement the kronecker product operator object
        template <typename T, typename backend = linalg::blas_backend>
        class site_product_operator : public primitive<T, backend>
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
            using mode_container_type = std::vector<std::shared_ptr<base_type>>;
            using container_type = std::vector<std::pair<size_type, mode_container_type>>;

        public:
            site_product_operator() : base_type() {}

            site_product_operator(const std::vector<size_type> &mdims, const std::vector<mode_container_type> &ops)
            try : base_type()
            {
                CALL_AND_RETHROW(initialise(mdims, ops));
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
            }

            site_product_operator(const std::vector<size_type> &mdims, const container_type &ops)
            try : base_type()
            {
                CALL_AND_RETHROW(initialise(mdims, ops));
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
            }

            void initialise(const std::vector<size_type> &mdims, const std::vector<mode_container_type> &ops)
            {
                ASSERT(mdims.size() == ops.size(), "Failed to initialise site product operator.  Mode dims and operator array do not have the same size.");

                // now go through and validate sizes
                for (size_type i = 0; i < mdims.size(); ++i)
                {
                    for (const auto &op : ops[i])
                    {
                        ASSERT(op->size() == mdims[i], "Failed to initialise site product operator.  Mode operator has incorrect dimension.");
                    }
                }

                m_dims = mdims;
                size_type size = 1;
                for (size_type i = 0; i < mdims.size(); ++i)
                {
                    if (ops[i].size() != 0)
                    {
                        std::pair<size_type, mode_container_type> v;
                        m_operators.push_back(std::make_pair(i, ops[i]));
                    }
                    size *= m_dims[i];
                }
                base_type::m_size = size;
            }

            void initialise(const std::vector<size_type> &mdims, const container_type &ops)
            {
                // now go through and validate sizes
                for (const auto &op : ops)
                {
                    ASSERT(op.first < mdims.size(), "Failed to initialise site product operator. Mode is out of bounds.");
                    for (const auto &_op : op.second)
                    {
                        ASSERT(_op->size() == mdims[op.first], "Failed to initialise site product operator.  Mode operator has incorrect dimension.");
                    }
                }

                m_dims = mdims;

                m_operators = ops;
                // now sort the operator array based on the mode index
                std::sort(m_operators.begin(), m_operators.end(), [](const std::pair<size_type, mode_container_type> &a, const std::pair<size_type, mode_container_type> &b)
                          { return a.first < b.first; });

                size_type size = 1;
                for (size_type i = 0; i < m_dims.size(); ++i)
                {
                    size *= m_dims[i];
                }
                base_type::m_size = size;
            }

            site_product_operator(const site_product_operator &o) = default;
            site_product_operator(site_product_operator &&o) = default;

            site_product_operator &operator=(const site_product_operator &o) = default;
            site_product_operator &operator=(site_product_operator &&o) = default;

            bool is_resizable() const final { return false; }
            void resize(size_type /*n*/) { ASSERT(false, "This shouldn't be called."); }

            void update(real_type /*t*/, real_type /*dt*/) final {}
            std::shared_ptr<base_type> clone() const { return std::make_shared<site_product_operator>(m_dims, m_operators); }

            std::shared_ptr<base_type> transpose() const
            {
                container_type operators;
                operators.resize(m_operators.size());
                for (size_type i = 0; i < m_operators.size(); ++i)
                {
                    // set the mode index
                    operators[i].first = m_operators[i].first;

                    // reserve storage for the operator elements
                    operators[i].second.reserve(m_operators[i].second.size());

                    // now iterate through all operators acting on this mode and bind
                    for (size_type j = 0; j < m_operators[i].second.size(); ++j)
                    {
                        operators[i].second.push_back(m_operators[i].second[j]->transpose());
                    }
                }
                return std::make_shared<site_product_operator>(m_dims, operators);
            }

            linalg::matrix<T> todense() const
            {
                // if there are no operators bound then we just return an identity matrix of the correct size
                if (m_operators.size() == 0)
                {
                    return linalg::matrix<T>(this->size(), this->size(), [](size_t i, size_t j)
                                             { return i == j ? T(1.0) : T(0.0); });
                }
                else
                {
                    // if there are operators we start by getting a vector of matrices acting on each mode
                    std::vector<linalg::matrix<T>> matrices(m_dims.size());

                    size_t counter = 0;
                    for (size_type i = 0; i < matrices.size(); ++i)
                    {
                        bool fill_identity = false;
                        if(counter < m_operators.size())
                        {
                            size_type mode = m_operators[counter].first;
                            if (mode == i)
                            {
                                if (m_operators[counter].second.size() == 1)
                                {
                                    matrices[i] = m_operators[counter].second[0]->todense();
                                }
                                // here we need to multiply all the operators acting on the mode in the correct order
                                else
                                {
                                    size_type Nm = m_operators[counter].second.size();
                                    matrices[i] = m_operators[counter].second[Nm-1]->todense();
                                    linalg::matrix<T> temp1, temp2;
                                    for(size_type j = 0; j < Nm; ++j)
                                    {
                                        temp1 = m_operators[counter].second[Nm-(j+1)]->todense();
                                        temp2 = temp1*matrices[i];
                                        matrices[i]=temp2;
                                    }
                                }
                                ++counter;
                            }
                            else
                            {
                                fill_identity = true;
                            }
                        }
                        else
                        {
                            fill_identity = true;
                        }
                        if(fill_identity)
                        {
                            matrices[i] = linalg::matrix<T>(m_dims[i], m_dims[i], [](size_t x, size_t y)
                                                            { return x == y ? T(1.0) : T(0.0); });
                        }
                    }

                    //we now have a vector of each of the operators acting on the individual primitive modes
                    //now we have to kron each of the matrices in order and return the result
                    linalg::matrix<T> ret(this->size(), this->size());

                    CALL_AND_HANDLE(kron::eval(matrices, ret), "Failed to evaluate kron prod");
                    return ret;
                }
            }

            void apply(const resview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const resview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const matview &A, resview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }

            void apply(const_matrix_ref A, matrix_ref HA) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_2(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA) final { CALL_AND_RETHROW(apply_rank_1(A, HA)); }
            void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final { CALL_AND_RETHROW(apply_rank_1(A, HA)); }

            // functions for applying the operator to rank 3 tensor views
            void apply(const restensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const restensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }
            void apply(const tensview &A, restensview &HA, real_type /* t */, real_type /* dt */) final { CALL_AND_RETHROW(apply_rank_3(A, HA)); }

        protected:
            template <typename T1, typename T2>
            void apply_rank_1(const T1 &A, T2 &HA)
            {
                CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
                ASSERT(m_operators.size() > 0, "Invliad operator object.");

                bool HA_set = apply_internal(A, HA);
                if (!HA_set)
                {
                    CALL_AND_HANDLE(HA = m_temp, "Failed to copy temp array.");
                }
            }

            template <typename T1, typename T2>
            void apply_rank_2(const T1 &A, T2 &HA)
            {
                CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
                ASSERT(m_operators.size() > 0, "Invliad operator object.");

                bool HA_set = apply_internal(A, HA);
                if (!HA_set)
                {
                    CALL_AND_HANDLE(HA = m_temp.reinterpret_shape(HA.shape(0), HA.shape(1)), "Failed to copy temp array.");
                }
            }

            template <typename T1, typename T2>
            void apply_rank_3(const T1 &A, T2 &HA)
            {
                CALL_AND_HANDLE(m_temp.resize(A.size()), "Failed to resize temporary array object.");
                ASSERT(m_operators.size() > 0, "Invliad operator object.");

                bool HA_set = apply_internal(A, HA, A.shape(0));
                if (!HA_set)
                {
                    CALL_AND_HANDLE(HA = m_temp.reinterpret_shape(HA.shape(0), HA.shape(1), HA.shape(2)), "Failed to copy temp array.");
                }
            }

            template <typename T1, typename T2>
            bool apply_internal(const T1 &A, T2 &HA, size_type d1 = 1)
            {
                std::array<size_type, 3> mdims = {{d1, 1, A.size() / d1}};

                bool HA_set = false;
                bool applied = false;
                size_t counter = 0;

                for (size_type i = 0; i < m_dims.size(); ++i)
                {
                    // set up the tensor dimensions
                    mdims[0] *= mdims[1];
                    mdims[1] = m_dims[i];
                    mdims[2] /= mdims[1];

                    // if the current counter is less than the total size of the operators array
                    if (counter < m_operators.size())
                    {
                        // check to see if the currently active operator acts on the current mode
                        if (m_operators[counter].first == i)
                        {
                            // and if it does we apply all of the operators as required
                            auto At = A.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
                            auto HAt = HA.reinterpret_shape(mdims[0], mdims[1], mdims[2]);
                            auto Tt = m_temp.reinterpret_shape(mdims[0], mdims[1], mdims[2]);

                            // now iterate over all of the modes acting on this term and apply them in reverse order
                            for (size_t j = 0; j < m_operators[counter].second.size(); ++j)
                            {
                                size_t ind = m_operators[counter].second.size() - (j + 1);

                                if (!applied)
                                {
                                    CALL_AND_HANDLE(m_operators[counter].second[ind]->apply(At, HAt), "Failed to compute kronecker product contraction.");
                                    HA_set = true;
                                    applied = true;
                                }
                                else
                                {
                                    if (HA_set)
                                    {
                                        CALL_AND_HANDLE(m_operators[counter].second[ind]->apply(HAt, Tt), "Failed to compute kronecker product contraction.");
                                        HA_set = false;
                                    }
                                    else
                                    {
                                        CALL_AND_HANDLE(m_operators[counter].second[ind]->apply(Tt, HAt), "Failed to compute kronecker product contraction.");
                                        HA_set = true;
                                    }
                                }
                            }
                            // increment the counter now that we have applied this operator
                            ++counter;
                        }
                    }
                }
                if (!applied)
                {
                    HA = A;
                    HA_set = true;
                }
                return HA_set;
            }

        public:
            std::string to_string() const final
            {
                std::stringstream oss;
                oss << "site product operator: " << std::endl;
                for (size_type i = 0; i < m_operators.size(); ++i)
                {
                    oss << "mode: " << m_operators[i].first << std::endl;
                    for (const auto &op : m_operators[i].second)
                    {
                        oss << op->to_string() << std::endl;
                    }
                }
                return oss.str();
            }

        protected:
            container_type m_operators;
            std::vector<size_type> m_dims;
            vector_type m_temp;

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void save(archive &ar) const
            {
                CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend>>(this)), "Failed to serialise site_product operator object.  Error when serialising the base object.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise site_product operator object.  Error when serialising the matrix.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("temp", m_temp)), "Failed to serialise site_product operator object.  Error when serialising the matrix.");
            }

            template <typename archive>
            void load(archive &ar)
            {
                CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend>>(this)), "Failed to serialise site_product operator object.  Error when serialising the base object.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operators)), "Failed to serialise site_product operator object.  Error when serialising the matrix.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("temp", m_temp)), "Failed to serialise site_product operator object.  Error when serialising the matrix.");
            }
#endif
        };
    }
}

#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::site_product_operator, ttns::ops::primitive)
#endif

#endif // PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SITE_PRODUCT_OPERATOR_HPP_
