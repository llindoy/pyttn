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

#ifndef PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SITE_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SITE_OPERATOR_HPP_

#include <complex>
#include <string>
#include <memory>

#include <linalg/linalg.hpp>
#include <common/tmp_funcs.hpp>

#include "primitive_operator.hpp"
#include "site_product_operator.hpp"

#include "../../sop/sSOP.hpp"
#include "../../sop/system_information.hpp"
#include "../../sop/operator_dictionaries/operator_dictionary.hpp"

namespace ttns
{

    /*
     * A class for handling individual site operators. Here we use type erasure to construct a type that is easier to work with for the
     * python side of the code.
     * */
    template <typename T, typename backend>
    class site_operator
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

    protected:
        mutable std::shared_ptr<ops::primitive<T, backend>> m_op;
        size_type m_mode;

    public:
        site_operator() : m_op(nullptr) {}

        site_operator(std::shared_ptr<ops::primitive<T, backend>> op) : m_op(op->clone()), m_mode(0) {}
        site_operator(std::shared_ptr<ops::primitive<T, backend>> op, size_type mode) : m_op(op->clone()), m_mode(mode) {}

        template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
        site_operator(const OpType &op) : m_op(std::make_shared<OpType>(op)), m_mode(0) {}

        template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
        site_operator(OpType &&op) : m_op(std::make_shared<OpType>(std::move(op))), m_mode(0) {}

        template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
        site_operator(const OpType &op, size_type mode) : m_op(std::make_shared<OpType>(op)), m_mode(mode) {}

        template <typename OpType, typename = typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, void>::type>
        site_operator(OpType &&op, size_type mode) : m_op(std::make_shared<OpType>(std::move(op))), m_mode(mode) {}

        site_operator(const sOP &sop, const system_modes &sys, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(sop, sys, use_sparse), "Failed to construct sop operator.");
        }
        site_operator(const sOP &sop, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(sop, sys, opdict, use_sparse), "Failed to construct sop operator.");
        }

        site_operator(const site_operator &o) = default;
        site_operator(site_operator &&o) = default;
        ~site_operator() {}

        site_operator &operator=(const site_operator &o) = default;
        site_operator &operator=(site_operator &&o) = default;

    protected:
        void setup_operator(const system_modes &sys, size_t mode, size_t lmode, std::shared_ptr<ops::primitive<T, backend>> op, bool is_composite_mode)
        {
            // if this isn't a composite mode then we just bind the operator
            if (!is_composite_mode)
            {
                m_op = op;
                op = nullptr;
            }
            // otherwise we use this to construct a site operator object
            else
            {
                std::vector<size_t> mode_dims(sys[mode].nmodes());
                for (size_t lmi = 0; lmi < sys[mode].nmodes(); ++lmi)
                {
                    mode_dims[lmi] = sys[mode][lmi].lhd();
                }
                std::vector<std::vector<std::shared_ptr<ops::primitive<T, backend>>>> ops(sys[mode].nmodes());
                ops[lmode].push_back(op);
                m_op = std::make_shared<ops::site_product_operator<T, backend>>(mode_dims, ops);
            }
        }

    public:
        // resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
        // This implementation does not support composite modes currently.  To do add mode combination
        void initialise(const sOP &sop, const system_modes &sys, bool use_sparse = true)
        {
            // get the primitive mode index associated with this sop term
            size_type nu = sop.mode();
            std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
            size_t mode = std::get<0>(mode_info);
            size_t lmode = std::get<1>(mode_info);

            // now get the composite mode index associated with this primitive mode
            m_mode = sys.mode_index(mode);

            bool is_composite_mode = sys[mode].nmodes() > 1;
            size_t hilbert_space_dimension = sys[mode][lmode].lhd();
            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

            std::string label = sop.op();
            using opdictype = operator_from_default_dictionaries<T, backend>;
            std::shared_ptr<ops::primitive<T, backend>> op;
            CALL_AND_HANDLE(op = opdictype::query(label, basis, sys.primitive_mode(nu).type(), use_sparse), "Failed to insert new element in mode operator.");

            ASSERT(op != nullptr, "Failed to construct site operator object.");

            setup_operator(sys, mode, lmode, op, is_composite_mode);
        }

        void initialise(const sOP &sop, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            // get the primitive mode index associated with this sop term
            size_type nu = sop.mode();
            std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
            size_t mode = std::get<0>(mode_info);
            size_t lmode = std::get<1>(mode_info);

            // now get the composite mode index associated with this primitive mode
            m_mode = sys.mode_index(mode);

            bool is_composite_mode = sys[mode].nmodes() > 1;
            size_t hilbert_space_dimension = sys[mode][lmode].lhd();
            std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

            using opdictype = operator_from_default_dictionaries<T, backend>;
            std::string label = sop.op();

            std::shared_ptr<ops::primitive<T, backend>> op = nullptr;
            if (nu < opdict.nmodes())
            {
                // first start to access element from opdict
                CALL_AND_HANDLE(op = opdict.query(nu, label), "Failed to query operator from user defined dictionary.");

                if (op != nullptr)
                {
                    ASSERT(op->size() == hilbert_space_dimension, "Invalid operator size in default operator dictionary.");
                }
            }

            if (op == nullptr)
            {
                CALL_AND_HANDLE(op = opdictype::query(label, basis, sys.primitive_mode(nu).type(), use_sparse), "Failed to insert new element in mode operator.");
                ASSERT(op != nullptr, "Failed to construct site operator object.");
            }

            setup_operator(sys, mode, lmode, op, is_composite_mode);
        }

        template <typename OpType>
        typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator &>::type
        operator=(const OpType &op)
        {
            m_op = std::make_shared<OpType>(op);
            return *this;
        }

        template <typename OpType>
        typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator &>::type
        operator=(OpType &&op)
        {
            m_op = std::make_shared<OpType>(std::move(op));
            return *this;
        }

        site_operator &operator=(std::shared_ptr<ops::primitive<T, backend>> op)
        {
            m_op = op->clone();
            return *this;
        }

        template <typename OpType>
        typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator &>::type
        bind(const OpType &op)
        {
            m_op = std::make_shared<OpType>(op);
            return *this;
        }

        template <typename OpType>
        typename std::enable_if<std::is_base_of<ops::primitive<T, backend>, OpType>::value, site_operator &>::type
        bind(OpType &&op)
        {
            m_op = std::make_shared<OpType>(std::move(op));
            return *this;
        }

        site_operator &bind(std::shared_ptr<ops::primitive<T, backend>> op)
        {
            m_op = op->clone();
            return *this;
        }

        template <typename Atype, typename HAtype>
        void apply(const Atype &A, HAtype &HA)
        {
            ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
            CALL_AND_RETHROW(m_op->apply(A, HA));
        }

        template <typename Atype, typename HAtype>
        void apply(const Atype &A, HAtype &HA, real_type t, real_type dt)
        {
            ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
            CALL_AND_RETHROW(m_op->apply(A, HA, t, dt));
        }
        template <typename Atype, typename HAtype>
        void apply(const Atype &A, HAtype &HA) const
        {
            ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
            CALL_AND_RETHROW(m_op->apply(A, HA));
        }

        template <typename Atype, typename HAtype>
        void apply(const Atype &A, HAtype &HA, real_type t, real_type dt) const
        {
            ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
            CALL_AND_RETHROW(m_op->apply(A, HA, t, dt));
        }

        // function for allowing you to update time-dependent Hamiltonians
        void update(real_type t, real_type dt)
        {
            ASSERT(m_op != nullptr, "Cannot apply empty site operator.");
            CALL_AND_RETHROW(m_op->update(t, dt));
        }

        bool is_resizable() const
        {
            ASSERT(m_op != nullptr, "Cannot check if empty site operator is resizable.");
            return m_op->is_resizable();
        }
        void resize(size_type n)
        {
            ASSERT(m_op != nullptr, "Cannot resize empty site operator.");
            CALL_AND_RETHROW(m_op->resize(n));
        }
        std::string to_string() const
        {
            ASSERT(m_op != nullptr, "Cannot convert empty site operator to string.");
            return m_op->to_string();
        };

        size_type mode_dimension() const { return m_op->size(); }
        size_type size() const { return m_op->size(); }
        bool is_identity() const { return m_op->is_identity(); }

        std::shared_ptr<ops::primitive<T, backend>> op() { return m_op; }
        std::shared_ptr<ops::primitive<T, backend>> op() const { return m_op->clone(); }

        size_t mode() const { return m_mode; }
        size_t &mode() { return m_mode; }

        site_operator transpose() const
        {
            site_operator ret(m_op->transpose(), m_mode);
            return ret;
        }

        void todense(const std::vector<size_type>& mode_dims, linalg::matrix<T>& ret) const
        {
            ASSERT(mode_dims.size() > m_mode, "Cannot convert siteoperator to dense matrix if the operator is not the correct size.");
            ASSERT(mode_dims[m_mode] == m_op->size(), "Cannot convert site operator to dense matrix.  The specified mode dims are not compatible with the current size.");
            std::vector<linalg::matrix<T>> mats(mode_dims.size());
            for(size_type i=0; i<mode_dims.size(); ++i)
            {
                if (i != m_mode)
                {
                    mats[i] = linalg::matrix<T>(mode_dims[i], mode_dims[i], [](size_t x, size_t y){return x==y ? T(1.0) : T(0.0);});
                }
                else
                {
                    mats[m_mode] = m_op->todense();
                }
            }
            T coeff(1.0);
            //now construct the dense matrix from these operators
            CALL_AND_HANDLE(kron::eval(coeff, mats, ret), "Failed to evaluate kron prod");
        }

        void todense(linalg::matrix<T>& ret) const
        {
            ret = m_op->todense();
        }

        linalg::matrix<T> todense(const std::vector<size_type>& mode_dims) const
        {
            ASSERT(mode_dims.size() > m_mode, "Cannot convert siteoperator to dense matrix if the operator is not the correct size.");
            ASSERT(mode_dims[m_mode] == m_op->size(), "Cannot convert site operator to dense matrix.  The specified mode dims are not compatible with the current size.");
            std::vector<linalg::matrix<T>> mats(mode_dims.size());
            for(size_type i=0; i<mode_dims.size(); ++i)
            {
                if (i != m_mode)
                {
                    mats[i] = linalg::matrix<T>(mode_dims[i], mode_dims[i], [](size_t x, size_t y){return x==y ? T(1.0) : T(0.0);});
                }
                else
                {
                    mats[m_mode] = m_op->todense();
                }
            }
            linalg::matrix<T> ret;

            T coeff(1.0);
            //now construct the dense matrix from these operators
            CALL_AND_HANDLE(kron::eval(coeff, mats, ret), "Failed to evaluate kron prod");
            return ret;
        }

        linalg::matrix<T> todense() const
        {
            return m_op->todense();
        }
#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void serialize(archive &ar) const
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("op", m_op)), "Failed to primitive operator.  Failed to serialise its size.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("mode", m_mode)), "Failed to primitive operator.  Failed to serialise its size.");
        }

#endif
    };

} // namespace ttns

#endif // PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SITE_OPERATOR_HPP_//
