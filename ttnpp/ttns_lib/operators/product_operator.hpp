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

#ifndef PYTTN_TTNS_LIB_OPERATORS_PRODUCT_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_PRODUCT_OPERATOR_HPP_

#include <linalg/linalg.hpp>
#include <common/zip.hpp>

#include <memory>
#include <list>
#include <vector>
#include <algorithm>
#include <map>

#include <linalg/linalg.hpp>
#include "site_operators/site_operator.hpp"
#include "../sop/system_information.hpp"
#include "../sop/sSOP.hpp"
#include "../sop/operator_dictionaries/operator_dictionary.hpp"
#include "../sop/coeff_type.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/vector.hpp>
#endif

namespace ttns
{

    // a generic sum of product operator object.  This stores all of the operator and indexing required for the evaluation of the sop on a TTN state but
    // doesn't store the buffers needed to perform the required contractions
    template <typename T, typename backend = linalg::blas_backend>
    class product_operator
    {
    public:
        using size_type = typename backend::size_type;
        using real_type = typename tmp::get_real_type<T>::type;

        using op_type = ops::primitive<T, backend>;
        using element_type = site_operator<T, backend>;

        using container_type = std::vector<element_type>;

        using value_type = T;
        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;
        using reverse_iterator = typename container_type::reverse_iterator;
        using const_reverse_iterator = typename container_type::const_reverse_iterator;

    protected:
        container_type m_mode_operators;
        literal::coeff<T> _m_coeff = T(1.0);
        T m_coeff = T(1.0);

    public:
        product_operator() {}
        product_operator(const product_operator &o) = default;
        product_operator(product_operator &&o) = default;
        product_operator(const sOP &op, const system_modes &sys, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(op, sys, use_sparse), "Failed to construct sop operator.");
        }
        product_operator(const sOP &op, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(op, sys, opdict, use_sparse), "Failed to construct sop operator.");
        }

        product_operator(const sPOP &op, const system_modes &sys, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(op, sys, use_sparse), "Failed to construct sop operator.");
        }
        product_operator(const sPOP &op, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(op, sys, opdict, use_sparse), "Failed to construct sop operator.");
        }
        product_operator(const sNBO<T> &op, const system_modes &sys, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(op, sys, use_sparse), "Failed to construct sop operator.");
        }
        product_operator(const sNBO<T> &op, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            CALL_AND_HANDLE(initialise(op, sys, opdict, use_sparse), "Failed to construct sop operator.");
        }
        product_operator &operator=(const product_operator &o) = default;
        product_operator &operator=(product_operator &&o) = default;

    protected:
        // function for unpacking a product operator into operators acting on the same modes, preserving the ordering of the modes.
        // Note that this assumes that operators acting on different modes commute and so will generally give incorrect operators
        // if applied to a fermionic operator that has not already been mapped to qubit operators
        std::map<size_t, std::list<std::pair<size_t, std::string>>> unpack_pop(const system_modes &sys, const sPOP &pop)
        {
            std::map<size_t, std::list<std::pair<size_t, std::string>>> ret;
            for (const auto &op : pop)
            {
                std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(op.mode());
                size_t mode = std::get<0>(mode_info);
                ret[mode].push_back(std::make_pair(op.mode(), op.op()));
            }
            return ret;
        }

        void setup_mode_operator(const system_modes &sys, size_t mode, size_t lmode, size_t mode_index, std::shared_ptr<op_type> op, bool is_composite_mode)
        {
            size_t smode = sys.mode_index(mode);
            // if this isn't a composite mode then we just bind the operator
            if (!is_composite_mode)
            {
                m_mode_operators[mode_index] = element_type(op, smode);
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
                std::vector<std::vector<std::shared_ptr<op_type>>> ops(sys[mode].nmodes());
                ops[lmode].push_back(op);
                CALL_AND_HANDLE(m_mode_operators[mode_index] = element_type(ops::site_product_operator<T, backend>{mode_dims, ops}, smode), "Failed to insert new element in mode operator");
            }
        }

    public:
        // resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
        // This implementation does not support composite modes currently.  To do add mode combination
        void initialise(const sPOP &pop, const system_modes &sys, bool use_sparse = true)
        {
            m_coeff = T(1.0);
            _m_coeff = T(1.0);

            // take the product operator which acts on the primitive modes and unpack it into a map
            // with key the tree mode index.  And key storing a list of the primitive mode index and
            // the label of each term
            auto up = unpack_pop(sys, pop);
            m_mode_operators.resize(up.size());

            using dfop_dict = operator_from_default_dictionaries<T, backend>;

            size_type mode_index = 0;
            // now iterate over each of the separate mode terms in the object
            for (const auto &x : up)
            {
                // get the mode index and determine if it is a composite mode
                size_t mode = x.first;
                bool is_composite_mode = sys[mode].nmodes() > 1;

                // now get the list of terms acting on this mode
                const auto &t = x.second;

                // if there is only one such term
                if (t.size() == 1)
                {
                    // work out the primitive mode it acts on and its label
                    size_t nu = std::get<0>(t.front());
                    std::string label = std::get<1>(t.front());

                    // see how this corresponds to local mode indices of composite modes
                    std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
                    size_t lmode = std::get<1>(mode_info);

                    // construct the basis object for this mode
                    size_t hilbert_space_dimension = sys[mode][lmode].lhd();
                    std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

                    // query the mode from the default dictionaries
                    std::shared_ptr<op_type> op;
                    CALL_AND_HANDLE(op = dfop_dict::query(label, basis, sys.primitive_mode(nu).type(), use_sparse), "Failed to query operator from dictionary.");

                    // and set up the mode operator
                    setup_mode_operator(sys, mode, lmode, mode_index, op, is_composite_mode);
                }
                // now if there are multiple terms acting on the mode we will always want to construct a site product operator
                else
                {
                    // determine the mode dimensions for this mode
                    std::vector<size_t> mode_dims(sys[mode].nmodes());
                    for (size_t lmi = 0; lmi < sys[mode].nmodes(); ++lmi)
                    {
                        mode_dims[lmi] = sys[mode][lmi].lhd();
                    }

                    // and build the list of all operators acting on it
                    std::vector<std::vector<std::shared_ptr<op_type>>> ops(sys[mode].nmodes());

                    // iterate over the set of modes
                    for (const auto &term : t)
                    {
                        // get there primitive mode dimension and label
                        size_t nu = std::get<0>(term);
                        std::string label = std::get<1>(term);

                        // from this construct the local mode index
                        std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
                        size_t lmode = std::get<1>(mode_info);

                        // construct the local basis object
                        size_t hilbert_space_dimension = sys[mode][lmode].lhd();
                        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

                        // and add the new primitive mode operator to act on the correct local mode of the composite mode operator
                        CALL_AND_HANDLE(ops[lmode].push_back(dfop_dict::query(label, basis, sys.primitive_mode(nu).type(), use_sparse)), "Failed to insert new element in mode operator.");
                    }
                    size_t smode = sys.mode_index(mode);
                    // now construct the mode operator object from this local mode information.
                    CALL_AND_HANDLE(m_mode_operators[mode_index] = element_type(ops::site_product_operator<T, backend>{mode_dims, ops}, smode), "Failed to insert new element in mode operator");
                }
                ++mode_index;
            }
        }

        void initialise(const sPOP &pop, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            m_coeff = T(1.0);
            _m_coeff = T(1.0);

            // take the product operator which acts on the primitive modes and unpack it into a map
            // with key the tree mode index.  And key storing a list of the primitive mode index and
            // the label of each term
            auto up = unpack_pop(sys, pop);
            m_mode_operators.resize(up.size());

            using dfop_dict = operator_from_default_dictionaries<T, backend>;

            size_type mode_index = 0;
            // now iterate over each of the separate mode terms in the object
            for (const auto &x : up)
            {
                // get the mode index and determine if it is a composite mode
                size_t mode = x.first;
                bool is_composite_mode = sys[mode].nmodes() > 1;

                // now get the list of terms acting on this mode
                const auto &t = x.second;

                // if there is only one such term
                if (t.size() == 1)
                {
                    // work out the primitive mode it acts on and its label
                    size_t nu = std::get<0>(t.front());
                    std::string label = std::get<1>(t.front());

                    // see how this corresponds to local mode indices of composite modes
                    std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
                    size_t lmode = std::get<1>(mode_info);

                    // construct the basis object for this mode
                    size_t hilbert_space_dimension = sys[mode][lmode].lhd();
                    std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

                    // query the mode from the user defined and default dictionaries
                    std::shared_ptr<op_type> op;
                    CALL_AND_HANDLE(op = opdict.query(nu, label), "Failed to query operator dictionary from user defined mode.");
                    if (op != nullptr)
                    {
                        ASSERT(op->size() == hilbert_space_dimension, "Invalid operator size in default operator dictionary.");
                    }
                    else
                    {
                        CALL_AND_HANDLE(op = dfop_dict::query(label, basis, sys.primitive_mode(nu).type(), use_sparse), "Failed to query operator from dictionary.");
                    }

                    // and set up the mode operator
                    setup_mode_operator(sys, mode, lmode, mode_index, op, is_composite_mode);
                }
                // now if there are multiple terms acting on the mode we will always want to construct a site product operator
                else
                {
                    // determine the mode dimensions for this mode
                    std::vector<size_t> mode_dims(sys[mode].nmodes());
                    for (size_t lmi = 0; lmi < sys[mode].nmodes(); ++lmi)
                    {
                        mode_dims[lmi] = sys[mode][lmi].lhd();
                    }

                    // and build the list of all operators acting on it
                    std::vector<std::vector<std::shared_ptr<op_type>>> ops(sys[mode].nmodes());

                    // iterate over the set of modes
                    for (const auto &term : t)
                    {
                        // get there primitive mode dimension and label
                        size_t nu = std::get<0>(term);
                        std::string label = std::get<1>(term);

                        // from this construct the local mode index
                        std::pair<size_t, size_t> mode_info = sys.primitive_mode_index(nu);
                        size_t lmode = std::get<1>(mode_info);

                        // construct the local basis object
                        size_t hilbert_space_dimension = sys[mode][lmode].lhd();
                        std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(hilbert_space_dimension, 1);

                        std::shared_ptr<op_type> op;

                        CALL_AND_HANDLE(op = opdict.query(nu, label), "Failed to query operator dictionary from user defined mode.");
                        if (op != nullptr)
                        {
                            ASSERT(op->size() == hilbert_space_dimension, "Invalid operator size in default operator dictionary.");
                        }
                        else
                        {
                            CALL_AND_HANDLE(op = dfop_dict::query(label, basis, sys.primitive_mode(nu).type(), use_sparse), "Failed to query operator from dictionary.");
                        }

                        // and add the new primitive mode operator to act on the correct local mode of the composite mode operator
                        CALL_AND_HANDLE(ops[lmode].push_back(op), "Failed to insert new element in mode operator.");
                    }
                    size_t smode = sys.mode_index(mode);
                    // now construct the mode operator object from this local mode information.
                    CALL_AND_HANDLE(m_mode_operators[mode_index] = element_type(ops::site_product_operator<T, backend>{mode_dims, ops}, smode), "Failed to insert new element in mode operator");
                }
                ++mode_index;
            }
        }

        // resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
        // This implementation does not support composite modes currently.  To do add mode combination
        void initialise(const sOP &_op, const system_modes &sys, bool use_sparse = true)
        {
            m_coeff = T(1.0);
            _m_coeff = T(1.0);
            m_mode_operators.resize(1);

            m_mode_operators[0].initialise(_op, sys, use_sparse);
        }

        void initialise(const sOP &_op, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            m_mode_operators.resize(1);
            m_coeff = T(1.0);
            _m_coeff = T(1.0);

            m_mode_operators[0].initialise(_op, sys, opdict, use_sparse);
        }

        // resize this object from a tree structure, a SOP object, a system info class and an optional operator dictionary.
        // This implementation does not support composite modes currently.
        void initialise(const sNBO<T> &pop, const system_modes &sys, bool use_sparse = true)
        {
            CALL_AND_RETHROW(initialise(pop.pop(), sys, use_sparse));
            _m_coeff = pop.coeff();
            update_coefficients(0);
        }

        void initialise(const sNBO<T> &pop, const system_modes &sys, const operator_dictionary<T, backend> &opdict, bool use_sparse = true)
        {
            CALL_AND_RETHROW(initialise(pop.pop(), sys, opdict, use_sparse));
            _m_coeff = pop.coeff();
            update_coefficients(0);
        }

        void clear()
        {
            m_mode_operators.clear();
        }

        const element_type &operators(size_type nu) const
        {
            return m_mode_operators[nu];
        }

        const element_type &operator[](size_type nu) const
        {
            return m_mode_operators[nu];
        }

        const element_type &operator()(size_type nu) const
        {
            return m_mode_operators[nu];
        }

        std::ostream &print(std::ostream &os) const
        {
            os << "prod_op: [" << m_coeff << " ";
            for (size_t i = 0; i < m_mode_operators.size(); ++i)
            {
                os << m_mode_operators[i].to_string() << (i + 1 != this->nmodes() ? ", " : "]");
            }
            return os;
        }

        void update_coefficients(real_type t)
        {
            m_coeff = _m_coeff(t);
        }

        const T &coeff() const { return m_coeff; }
        T &coeff() { return m_coeff; }

        const container_type &mode_operators() const { return m_mode_operators; }
        container_type &mode_operators() { return m_mode_operators; }

        size_type nmodes() const { return m_mode_operators.size(); }

        iterator begin() { return iterator(m_mode_operators.begin()); }
        iterator end() { return iterator(m_mode_operators.end()); }
        const_iterator begin() const { return const_iterator(m_mode_operators.begin()); }
        const_iterator end() const { return const_iterator(m_mode_operators.end()); }

        reverse_iterator rbegin() { return reverse_iterator(m_mode_operators.rbegin()); }
        reverse_iterator rend() { return reverse_iterator(m_mode_operators.rend()); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(m_mode_operators.rbegin()); }
        const_reverse_iterator rend() const { return const_reverse_iterator(m_mode_operators.rend()); }


        void todense(const std::vector<size_type>& mode_dims, linalg::matrix<T>& ret) const
        {
            std::vector<linalg::matrix<T>> mats(mode_dims.size());

            size_t counter = 0;
            for(size_type i=0; i<mode_dims.size(); ++i)
            {
                if(counter > m_mode_operators.size())
                {
                    mats[i] = linalg::matrix<T>(mode_dims[i], mode_dims[i], [](size_t x, size_t y){return x==y ? T(1.0) : T(0.0);});
                }
                else
                {
                    if (i == m_mode_operators[counter].mode())
                    {
                        ASSERT(mode_dims[i] == m_mode_operators[counter].mode_dimension(), "Cannot convert site operator to dense matrix.  The specified mode dims are not compatible with the current size.");
                        CALL_AND_HANDLE(m_mode_operators[counter].todense(mats[i]), "Failed to construct dense matrix from product operator.  Failed to convert a site operator.");
                        ++counter;
                    }
                    else
                    {
                        mats[i] = linalg::matrix<T>(mode_dims[i], mode_dims[i], [](size_t x, size_t y){return x==y ? T(1.0) : T(0.0);});
                    }
                }
            }
            ASSERT(counter == m_mode_operators.size(), "Invalid conversion of mode operator to dense matrix.  Likely insufficient mode dimensions passed to function.");

            //now construct the dense matrix from these operators
            CALL_AND_HANDLE(kron::eval(m_coeff, mats, ret), "Failed to evaluate kron prod");
        }

#ifdef CEREAL_LIBRARY_FOUND
    public:
        template <typename archive>
        void serialize(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("operators", m_mode_operators)), "Failed to serialise sum of product operator.  Failed to serialise array of operators.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("coeff", m_coeff)), "Failed to serialise sum of product operator.  Failed to serialise array of operators.");
        }
#endif
    }; // class product_operator

    template <typename T, typename backend>
    std::ostream &operator<<(std::ostream &os, const product_operator<T, backend> &t)
    {
        return t.print(os);
    }

} // namespace ttns

#endif // PYTTN_TTNS_LIB_OPERATORS_PRODUCT_OPERATOR_HPP_
