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

#ifndef PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_NLEVEL_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_NLEVEL_OPERATOR_HPP_

#include <linalg/linalg.hpp>
#include "single_site_operator.hpp"

#include <regex>
#include <string>
#include <tuple>

namespace ttns
{
    namespace nlevel
    {
        inline std::tuple<size_t, size_t, bool> extract_indices(const std::string &str)
        {
            std::regex r("\\|(\\d+)\\>\\<(\\d+)\\|");
            std::smatch m;
            bool match = std::regex_search(str, m, r);
            if (match)
            {
                return std::tuple<size_t, size_t, bool>(std::stoi(m[1].str()), std::stoi(m[2].str()), true);
            }
            else
            {
                return std::tuple<size_t, size_t, bool>(0, 0, false);
            }
        }

        inline bool valid_two_level(const std::string &str)
        {
            std::tuple<size_t, size_t, bool> inds = extract_indices(str);
            if (std::get<0>(inds) < 2 && std::get<1>(inds) < 2 && std::get<2>(inds))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        template <typename T>
        class nlevel_op : public single_site_operator<T>
        {
        protected:
            size_t m_i;
            size_t m_j;

        public:
            nlevel_op(size_t i, size_t j) : m_i(i), m_j(j) {}
            nlevel_op(const std::string &label)
            {
                std::tuple<size_t, size_t, bool> inds = extract_indices(label);
                ASSERT(std::get<2>(inds), "Failed to construct nlevel_op.  The input string is not valid.");

                m_i = std::get<0>(inds);
                m_j = std::get<1>(inds);
            }
            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::diagonal_matrix<T> &mat) const
            {
                if (m_i == m_j)
                {
                    ASSERT(m_i < op->dim(index) && m_j < op->dim(index), "Cannot form matrix representation of operator.  State indices are out of bounds.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        if (n == m_i)
                        {
                            mat(i, i) = 1.0;
                        }
                    }
                }
                else
                {
                    RAISE_EXCEPTION("Cannot form requested nlevel operator as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
                }
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(m_i < op->dim(index) && m_j < op->dim(index), "Cannot form matrix representation of operator.  State indices are out of bounds.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    std::vector<size_t> state_i(op->nmodes());
                    std::fill(state_i.begin(), state_i.end(), 0);
                    std::vector<size_t> state_j(op->nmodes());
                    std::fill(state_j.begin(), state_j.end(), 0);

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        op->get_state(i, state_i);
                        if (state_i[index] == m_i)
                        {
                            for (size_t j = 0; j < op->nstates(); ++j)
                            {
                                op->get_state(j, state_j);
                                if (state_j[index] == m_j)
                                {
                                    bool all_same = true;
                                    for (size_t k = 0; k < op->nmodes(); ++k)
                                    {
                                        if (k != index && state_i[k] != state_j[k])
                                        {
                                            all_same = false;
                                        }
                                    }
                                    if (all_same)
                                    {
                                        ++nnz;
                                    }
                                }
                            }
                        }
                    }

                    mat.resize(nnz, op->nstates(), op->nstates());
                    auto rowptr = mat.rowptr();
                    rowptr[0] = 0;
                    auto colind = mat.colind();
                    auto buffer = mat.buffer();

                    size_t counter = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        op->get_state(i, state_i);
                        if (state_i[index] == m_i)
                        {
                            for (size_t j = 0; j < op->nstates(); ++j)
                            {
                                op->get_state(j, state_j);
                                if (state_j[index] == m_j)
                                {
                                    bool all_same = true;
                                    for (size_t k = 0; k < op->nmodes(); ++k)
                                    {
                                        if (k != index && state_i[k] != state_j[k])
                                        {
                                            all_same = false;
                                        }
                                    }
                                    if (all_same)
                                    {
                                        buffer[counter] = 1.0;
                                        colind[counter] = j;
                                        ++counter;
                                    }
                                }
                            }
                        }
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct generic nlevel operator as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                ASSERT(m_i < op->dim(index) && m_j < op->dim(index), "Cannot form matrix representation of operator.  State indices are out of bounds.");
                ASSERT(index < op->nmodes(), "Index out of bounds.");
                mat.resize(op->nstates(), op->nstates());
                mat.fill_zeros();

                std::vector<size_t> state_i(op->nmodes());
                std::fill(state_i.begin(), state_i.end(), 0);
                std::vector<size_t> state_j(op->nmodes());
                std::fill(state_j.begin(), state_j.end(), 0);

                for (size_t i = 0; i < op->nstates(); ++i)
                {
                    op->get_state(i, state_i);
                    if (state_i[index] == m_i)
                    {
                        for (size_t j = 0; j < op->nstates(); ++j)
                        {
                            op->get_state(j, state_j);
                            if (state_j[index] == m_j)
                            {
                                bool all_same = true;
                                for (size_t k = 0; k < op->nmodes(); ++k)
                                {
                                    if (k != index && state_i[k] != state_j[k])
                                    {
                                        all_same = false;
                                    }
                                }
                                if (all_same)
                                {
                                    mat(i, j) = 1.0;
                                }
                            }
                        }
                    }
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("|") + std::to_string(m_j) + std::string("><") + std::to_string(m_i) + std::string("|"));
                return ret;
            }
        };

    } // namespace nlevel

} // namespace ttns
#endif // PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_NLEVEL_OPERATOR_HPP_
