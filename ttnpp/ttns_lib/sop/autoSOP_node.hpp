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

#ifndef PYTTN_TTNS_LIB_SOP_AUTOSOP_NODE_HPP_
#define PYTTN_TTNS_LIB_SOP_AUTOSOP_NODE_HPP_

#include <utility>
#include <utils/product_iterator.hpp>
#include <utils/bipartite_graph.hpp>
#include <utils/term_indexing_array.hpp>

#include "../operators/operator_contraction_info.hpp"
#include "../ttn/ttn_nodes/ttn_node.hpp"

namespace ttns
{

    template <typename T>
    class autoSOP;

    namespace auto_sop
    {

        struct opinfo
        {
            std::vector<std::vector<size_t>> m_indices;
            bool m_is_identity;

            opinfo() : m_is_identity(false) {}
            opinfo(const std::vector<std::vector<size_t>> &inds) : m_indices(inds), m_is_identity(false) {}
            opinfo(std::vector<std::vector<size_t>> &&inds) : m_indices(std::move(inds)), m_is_identity(false) {}

            opinfo(const std::vector<std::vector<size_t>> &inds, bool is_identity) : m_indices(inds), m_is_identity(is_identity) {}
            opinfo(std::vector<std::vector<size_t>> &&inds, bool &&is_identity) : m_indices(std::move(inds)), m_is_identity(std::move(is_identity)) {}
            opinfo(const opinfo &o) = default;
            opinfo(opinfo &&o) = default;

            ~opinfo() { clear(); }

            opinfo &operator=(const opinfo &o) = default;
            opinfo &operator=(opinfo &&o) = default;

            bool is_identity() const { return m_is_identity; }
            bool &is_identity() { return m_is_identity; }

            void push_back(const std::vector<size_t> &i) { m_indices.push_back(i); }
            size_t size() const { return m_indices.size(); }

            const std::vector<size_t> &operator[](size_t i) const { return m_indices[i]; }

            void clear()
            {
                m_is_identity = false;
                for (size_t i = 0; i < m_indices.size(); ++i)
                {
                    m_indices[i].clear();
                    m_indices[i].shrink_to_fit();
                }
                m_indices.clear();
                m_indices.shrink_to_fit();
            }

            bool operator==(const opinfo &o) const
            {
                return m_indices == o.m_indices && m_is_identity == o.m_is_identity;
            }

            bool operator!=(const opinfo &o) const
            {
                return !this->operator==(o);
            }

            bool operator<(const opinfo &o) const
            {
                return m_indices < o.m_indices;
            }

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void serialize(archive &ar)
            {
                CALL_AND_HANDLE(ar(cereal::make_nvp("inds", m_indices)), "Failed to serialise operator_data object.  Error when serialising operator definitions.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("is_identity", m_is_identity)), "Failed to serialise operator_data object.  Error when serialising term indices.");
            }
#endif
        };

        inline std::ostream &operator<<(std::ostream &os, const opinfo &op)
        {
            for (size_t i = 0; i < op.m_indices.size(); ++i)
            {
                os << "[";
                for (size_t j = 0; j < op.m_indices[i].size(); ++j)
                {
                    os << op.m_indices[i][j] << " ";
                }
                os << "]";
            }
            os << "is identity: " << (op.m_is_identity ? "true" : "false");
            return os;
        }

        template <typename T>
        class node_op_info;

        template <typename T>
        class operator_data
        {
        public:
            operator_data() {}
            operator_data(const opinfo &op) : m_opdef(op) {}
            operator_data(const opinfo &op, const utils::term_indexing_array<size_t> &ind) : m_opdef(op), m_inds(ind) {}
            operator_data(const opinfo &op, const utils::term_indexing_array<size_t> &ind, const std::vector<literal::coeff<T>> &coeff) : m_opdef(op), m_inds(ind), m_accum_coeff(coeff) {}
            operator_data(const operator_data &o) = default;
            operator_data(operator_data &&o) = default;

            ~operator_data() { clear(); }

            operator_data &operator=(const operator_data &o) = default;
            operator_data &operator=(operator_data &&o) = default;

            const opinfo &operator_definition() const { return m_opdef; }
            const utils::term_indexing_array<size_t> &indices() const { return m_inds; }
            const std::vector<literal::coeff<T>> &accumulation_coefficients() const { return m_accum_coeff; }
            void set_accumulation_coefficients(const std::vector<literal::coeff<T>> &coeffs) const { m_accum_coeff = coeffs; }
            const bool &is_identity() const { return m_opdef.m_is_identity; }
            bool &is_identity() { return m_opdef.m_is_identity; }

            size_t nterms() const { return m_opdef.size(); }

            void clear()
            {
                m_opdef.clear();
                m_inds.clear();
            }

            /*
            void stash()
            {
            }

            void fetch()
            {
            }
            */

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void serialize(archive &ar)
            {
                CALL_AND_HANDLE(ar(cereal::make_nvp("opdef", m_opdef)), "Failed to serialise operator_data object.  Error when serialising operator definitions.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("inds", m_inds)), "Failed to serialise operator_data object.  Error when serialising term indices.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("accum_coeff", m_accum_coeff)), "Failed to serialise operator_data object.  Error when serialising accumulation coefficients.");
            }
#endif

        protected:
            opinfo &operator_definition() { return m_opdef; }
            utils::term_indexing_array<size_t> &indices() { return m_inds; }
            std::vector<literal::coeff<T>> &accumulation_coefficients() { return m_accum_coeff; }

        protected:
            opinfo m_opdef;
            utils::term_indexing_array<size_t> m_inds;

            // the coefficients used when accumulating the terms together to form a composite operator.
            // If the m_opdef only contains a single element then this is simply set to vector of length 1
            // containing static_cast<T>(1.0), as we never need to accumulate this term
            std::vector<literal::coeff<T>> m_accum_coeff;

            friend class node_op_info<T>;
            friend class autoSOP<T>;
        };

        template <typename T>
        class node_op_info
        {
        public:
            node_op_info() {}
            node_op_info(const std::vector<operator_data<T>> &spf, const std::vector<operator_data<T>> &mf) : m_spf(spf), m_mf(mf) {}
            node_op_info(const node_op_info &o) = default;
            node_op_info(node_op_info &&o) = default;

            node_op_info &operator=(const node_op_info &o) = default;
            node_op_info &operator=(node_op_info &&o) = default;

            const std::vector<operator_data<T>> &spf() const { return m_spf; }
            const std::vector<operator_data<T>> &mf() const { return m_mf; }
            const std::vector<literal::coeff<T>> &coeff() const { return m_coeff; }

            std::vector<operator_data<T>> &spf() { return m_spf; }
            std::vector<operator_data<T>> &mf() { return m_mf; }
            std::vector<literal::coeff<T>> &coeff() { return m_coeff; }

            size_t nterms() const { return m_spf.size(); }

            bool valid_bipartition() const
            {
                return (m_spf.size() == m_mf.size() && m_coeff.size() == m_mf.size());
            }

            void clear()
            {
                for (auto &sp : m_spf)
                {
                    sp.clear();
                }
                m_spf.clear();
                m_spf.shrink_to_fit();
                for (auto &m : m_mf)
                {
                    m.clear();
                }
                m_mf.clear();
                m_mf.shrink_to_fit();
            }

#ifdef CEREAL_LIBRARY_FOUND
        public:
            template <typename archive>
            void serialize(archive &ar)
            {
                CALL_AND_HANDLE(ar(cereal::make_nvp("spf", m_spf)), "Failed to serialise node_op_info object.  Error when serialising spf definitions.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("mf", m_mf)), "Failed to serialise node_op_info object.  Error when serialising mf definitions.");
                CALL_AND_HANDLE(ar(cereal::make_nvp("rcoeff", m_coeff)), "Failed to serialise node_op_info object.  Error when serialising coefficients.");
            }

#endif

        protected:
            std::vector<operator_data<T>> m_spf;
            std::vector<operator_data<T>> m_mf;
            std::vector<literal::coeff<T>> m_coeff;

            friend class autoSOP<T>;

            friend class tree_node<tree_base<auto_sop::node_op_info<T>>>;

        protected:
            static inline void generate_r_array(const std::vector<operator_data<T>> &op, size_t nterms, std::vector<size_t> &rs)
            {
                ASSERT(op.size() > 0, "Cannot generate r array object if this object has no operators bound.")
                if (rs.size() != nterms)
                {
                    rs.resize(nterms);
                }

                for (size_t ind = 0; ind < op.size(); ++ind)
                {
                    for (size_t i : op[ind].m_inds)
                    {
                        rs[i] = ind;
                    }
                }
            }

            static inline void setup_r_indexing(const std::vector<size_t> &rs, std::vector<operator_data<T>> &op)
            {
                for (auto &o : op)
                {
                    o.m_inds.reserve(rs.size());
                }
                for (size_t r = 0; r < rs.size(); ++r)
                {
                    op[rs[r]].m_inds.insert(r);
                }
            }

        public:
            void generate_mfr(std::vector<size_t> &rs, size_t nterms) const
            {
                CALL_AND_RETHROW(generate_r_array(m_mf, nterms, rs));
            }

            void setup_mf_r_indexing(const std::vector<size_t> &rs)
            {
                setup_r_indexing(rs, m_mf);
            }

            void generate_spfr(std::vector<size_t> &rs, size_t nterms) const
            {
                CALL_AND_RETHROW(generate_r_array(m_spf, nterms, rs));
            }

            void setup_spf_r_indexing(const std::vector<size_t> &rs)
            {
                setup_r_indexing(rs, m_spf);
            }
        };

    }

    // TODO implement write to disk storage for the tree node object to avoid memory issues.
    template <typename T>
    class tree_node<tree_base<auto_sop::node_op_info<T>>> : public tree_node_base<tree_base<auto_sop::node_op_info<T>>>
    {
    public:
        using base_type = tree_node_base<tree_base<auto_sop::node_op_info<T>>>;
        using coeff_type = T;

        template <typename iter_type, typename dtype>
        class _product_iterator
        {
        protected:
            using iter_vec = std::vector<iter_type>;
            using ref_vec = std::vector<dtype *>;

            using self_type = _product_iterator<iter_type, dtype>;

        public:
            _product_iterator(iter_vec &state, const iter_vec &begin, const iter_vec &end) : m_state(state), m_begin(begin), m_end(end)
            {
                ASSERT(state.size() == begin.size() && begin.size() == end.size(), "Cannot construct this iterator.");
            }

            bool operator==(const self_type &other) const { return m_state == other.m_state; }
            bool operator!=(const self_type &other) const { return m_state != other.m_state; }

            ref_vec operator*() const
            {
                ref_vec ret;
                for (auto v : m_state)
                {
                    ret.push_back(std::reference_wrapper<dtype>(*v));
                }
                return ret;
            }

            dtype &operator[](size_t i)
            {
                ASSERT(i < m_state.size(), "Failed to access element index out of bounds.");
                return *m_state[i];
            }

            iter_vec state() { return m_state; }

            self_type &operator++()
            {
                for (size_t i = 0; i < m_state.size(); ++i)
                {
                    size_t ind = m_state.size() - i - 1;
                    if (ind != 0 && m_state[ind] + 1 == m_end[ind])
                    {
                        m_state[ind] = m_begin[ind];
                    }
                    else if (ind == 0 && m_state[0] + 1 == m_end[0])
                    {
                        m_state = m_end;
                    }
                    else
                    {
                        ++m_state[ind];
                        return *this;
                    }
                }
                return *this;
            }

            self_type operator++(int)
            {
                self_type ret(*this);
                ++(*this);
                return ret;
            }

        protected:
            iter_vec m_state;
            iter_vec m_begin;
            iter_vec m_end;
        };

        using iter_vec_type = typename std::vector<auto_sop::operator_data<T>>::iterator;
        using const_iter_vec_type = typename std::vector<auto_sop::operator_data<T>>::const_iterator;

        friend class tree<auto_sop::node_op_info<T>>;
        friend class tree_base<auto_sop::node_op_info<T>>;

    protected:
        using base_type::m_children;
        using base_type::m_data;
        using base_type::m_parent;

    protected:
        using spfind = typename operator_contraction_info<T>::spf_index_type;
        using mfind = typename operator_contraction_info<T>::mf_index_type;

        template <typename SPType>
        void get_contraction_info_spf_term(const SPType &spf, spfind &sp, bool &identity_spf, bool exploit_identity_opt) const
        {
            // if the current term is the identity term
            if (spf.is_identity() && exploit_identity_opt)
            {
                identity_spf = true;
            }

            // if it is not the identity and we are making use of this fact
            if (!identity_spf)
            {
                sp.reserve(spf.nterms());

                const auto &op = spf.operator_definition();

                if (!this->is_leaf())
                {
                    // iterate over each of the terms in this sum
                    for (size_t spi = 0; spi < op.size(); ++spi)
                    {
                        std::vector<std::array<size_t, 2>> term;
                        // iterate over each mode of the terms
                        for (size_t d = 0; d < op[spi].size(); ++d)
                        {
                            size_t ct = op[spi][d];
                            // and if the child term that acts on this mode is not the idendity
                            if (!exploit_identity_opt || !this->operator[](d)().spf()[ct].is_identity())
                            {
                                // then bind the operator and the child index to this term.
                                term.push_back({{d, ct}});
                            }
                        }
                        sp.push_back(term);
                    }
                }
                else
                {
                    // iterate over each of the terms in this sum
                    for (size_t spi = 0; spi < op.size(); ++spi)
                    {
                        std::vector<std::array<size_t, 2>> term;
                        // iterate over each mode of the terms
                        for (size_t d = 0; d < op[spi].size(); ++d)
                        {
                            size_t ct = op[spi][d];
                            // then bind the operator and the child index to this term.
                            term.push_back({{this->leaf_index(), ct}});
                        }
                        sp.push_back(term);
                    }
                }
            }
        }

        template <typename MFType>
        void get_contraction_info_mf_term(const MFType &mf, mfind &m, bool &identity_mf, bool exploit_identity_opt) const
        {
            if ((mf.is_identity() && exploit_identity_opt) || this->is_root())
            {
                identity_mf = true;
            }

            // if the current term is not the identity term
            if (!identity_mf)
            {
                m.reserve(mf.nterms());
                const auto &op = mf.operator_definition();

                if (!this->is_root())
                {
                    auto &np = this->parent();
                    // iterate over each of the terms in this sum
                    for (size_t mi = 0; mi < op.size(); ++mi)
                    {
                        size_t ctp = op[mi][0];
                        size_t parent_index = ctp;

                        std::list<std::array<size_t, 2>> indexing;

                        size_t d = 1;
                        for (size_t ci = 0; ci < np.size(); ++ci)
                        {
                            if (ci != this->child_id())
                            {
                                size_t ri = op[mi][d];
                                if (!exploit_identity_opt || !np[ci]().spf()[ri].is_identity())
                                {
                                    indexing.push_back({{ci, ri}});
                                }
                                ++d;
                            }
                        }
                        m.push_back(mf_index<size_t>{parent_index, {indexing.begin(), indexing.end()}});
                    }
                }
            }
            if (this->is_root() && !mf.is_identity())
            {
                m.reserve(mf.nterms());
                const auto &op = mf.operator_definition();

                mf_index<size_t> empty(0);
                empty.parent_index() = 0;
                for (size_t mi = 0; mi < op.size(); ++mi)
                {
                    m.push_back(empty);
                }
            }
        }

    public:
        template <typename U>
        void get_contraction_info(std::vector<operator_contraction_info<U>> &inf, bool exploit_identity_opt = true) const
        {
            ASSERT(m_data.valid_bipartition(), "Cannot construct contraction info object from node_op_info.  Invalid bipartition.");
            inf.clear();

            const auto &spf = m_data.m_spf;
            const auto &mf = m_data.m_mf;

            inf.reserve(spf.size());

            for (size_t i = 0; i < spf.size(); ++i)
            {
                bool identity_spf = false;
                bool identity_mf = false;

                spfind sp;
                this->get_contraction_info_spf_term(spf[i], sp, identity_spf, exploit_identity_opt);

                mfind m;
                this->get_contraction_info_mf_term(mf[i], m, identity_mf, exploit_identity_opt);

                // set up the outer coefficient for this term
                inf.push_back(operator_contraction_info<U>(sp, m, m_data.m_coeff[i], spf[i].accumulation_coefficients(), mf[i].accumulation_coefficients(), identity_spf, identity_mf));
            }
        }

        // template <typename T>
        // void as_ttn_node(ttn_node_data<T>& n) const
        //{
        //     ASSERT(m_data.valid_bipartition(), "Cannot construct contraction info object from node_op_info.  Invalid bipartition.");
        //     n.clear();

        //    const auto& spf = m_data.m_spf;
        //    const auto& mf = m_data.m_mf;

        //    size_t hrank = m_data.nterms();
        //    std::vector<size_t> mode_dims;
        //    if(n().is_leaf())
        //    {
        //        mode_dims.resize(0);
        //        for(size_t i = 0; i < spf.
        //    }

        //    n().resize(hrank(), mode_dims);

        // std::vector<size_type> m_mode_dims;
        // std::vector<size_type> m_mode_capacity;
        // size_type m_max_hrank;
        // size_type m_max_dimen;
        // void resize(size_type hrank, const std::vector<size_type>& mode_dims)

        //}

        using product_iterator = _product_iterator<iter_vec_type, auto_sop::operator_data<T>>;
        using const_product_iterator = _product_iterator<const_iter_vec_type, const auto_sop::operator_data<T>>;

        product_iterator spf_prod_begin()
        {
            std::vector<iter_vec_type> state(m_children.size());
            std::vector<iter_vec_type> begin(m_children.size());
            std::vector<iter_vec_type> end(m_children.size());
            for (size_t i = 0; i < m_children.size(); ++i)
            {
                state[i] = m_children[i]->data().spf().begin();
                begin[i] = m_children[i]->data().spf().begin();
                end[i] = m_children[i]->data().spf().end();
            }

            return product_iterator(state, begin, end);
        }
        product_iterator spf_prod_end()
        {
            std::vector<iter_vec_type> state(m_children.size());
            std::vector<iter_vec_type> begin(m_children.size());
            std::vector<iter_vec_type> end(m_children.size());
            for (size_t i = 0; i < m_children.size(); ++i)
            {
                state[i] = m_children[i]->data().spf().end();
                begin[i] = m_children[i]->data().spf().begin();
                end[i] = m_children[i]->data().spf().end();
            }

            return product_iterator(state, begin, end);
        }
        const_product_iterator spf_prod_begin() const
        {
            std::vector<const_iter_vec_type> state(m_children.size());
            std::vector<const_iter_vec_type> begin(m_children.size());
            std::vector<const_iter_vec_type> end(m_children.size());
            for (size_t i = 0; i < m_children.size(); ++i)
            {
                state[i] = m_children[i]->data().spf().begin();
                begin[i] = m_children[i]->data().spf().begin();
                end[i] = m_children[i]->data().spf().end();
            }

            return const_product_iterator(state, begin, end);
        }

        const_product_iterator spf_prod_end() const
        {
            std::vector<const_iter_vec_type> state(m_children.size());
            std::vector<const_iter_vec_type> begin(m_children.size());
            std::vector<const_iter_vec_type> end(m_children.size());
            for (size_t i = 0; i < m_children.size(); ++i)
            {
                state[i] = m_children[i]->data().spf().end();
                begin[i] = m_children[i]->data().spf().begin();
                end[i] = m_children[i]->data().spf().end();
            }

            return const_product_iterator(state, begin, end);
        }
    };

    namespace node_data_traits
    {
        // clear traits for the operator node data object
        template <typename T>
        struct clear_traits<auto_sop::node_op_info<T>>
        {
            void operator()(auto_sop::node_op_info<T> &t) { CALL_AND_RETHROW(t.clear()); }
        };

    } // namespace node_data_traits
} // namespace ttns

#endif // PYTTN_TTNS_LIB_SOP_AUTOSOP_NODE_HPP_
