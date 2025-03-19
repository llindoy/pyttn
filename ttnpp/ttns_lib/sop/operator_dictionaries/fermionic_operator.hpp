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

#ifndef PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_FERMIONIC_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_FERMIONIC_OPERATOR_HPP_

#include <linalg/linalg.hpp>
#include "single_site_operator.hpp"

namespace ttns
{

    namespace fermion
    {

        // form the one body fermion creation operator.  This operator does not correctly satisfy the fermion anticommutation operators
        template <typename T>
        class creation : public single_site_operator<T>
        {
        public:
            creation() {}
            virtual bool is_sparse() const { return true; }
            virtual bool changes_parity() const { return true; }
            virtual bool jw_sign_change() const { return false; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form fermion creation operator as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            ++nnz;
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
                        if (op->contains_lowered_state(i, index))
                        {
                            // size_t n = op->get_occupation(i, index);
                            buffer[counter] = 1.0;
                            colind[counter] = op->get_lowered_index(i, index);
                            ++counter;
                        }
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion creation operator as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                ASSERT(index < op->nmodes(), "Index out of bounds.");
                mat.resize(op->nstates(), op->nstates());
                mat.fill_zeros();

                try
                {
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_raised_state(i, index))
                        {
                            mat(op->get_raised_index(i, index), i) = 1.0;
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion creation operator as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("c"));
                return ret;
            }
        };

        template <typename T>
        class annihilation : public single_site_operator<T>
        {
        public:
            annihilation() {}
            virtual bool changes_parity() const { return true; }
            virtual bool jw_sign_change() const { return true; }

            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form fermion annihilation operator as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_raised_state(i, index))
                        {
                            ++nnz;
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
                        if (op->contains_raised_state(i, index))
                        {
                            // size_t n = op->get_occupation(i, index);
                            buffer[counter] = 1.0;
                            colind[counter] = op->get_raised_index(i, index);
                            ++counter;
                        }
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion annihilation operator as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            mat(op->get_lowered_index(i, index), i) = 1.0;
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion annihilation operator as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("cdag"));
                return ret;
            }
        };

        template <typename T>
        class number : public single_site_operator<T>
        {
        public:
            number() {}

            virtual bool is_sparse() const { return true; }
            virtual bool is_diagonal() const { return true; }
            virtual bool changes_parity() const { return false; }
            virtual bool jw_sign_change() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::diagonal_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        mat(i, i) = (n == 0 ? 0.0 : 1.0);
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion number operator as dense.");
                }
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        if (n != 0)
                        {
                            ++nnz;
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
                        size_t n = op->get_occupation(i, index);

                        if (n != 0)
                        {
                            buffer[counter] = T(1.0);
                            colind[counter] = i;
                            ++counter;
                        }
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion number operator as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        mat(i, i) = (n == 0 ? 0.0 : 1.0);
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion number operator as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("n"));
                return ret;
            }
        };

        template <typename T>
        class vacancy : public single_site_operator<T>
        {
        public:
            vacancy() {}

            virtual bool is_sparse() const { return true; }
            virtual bool is_diagonal() const { return true; }
            virtual bool changes_parity() const { return false; }
            virtual bool jw_sign_change() const { return false; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::diagonal_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        mat(i, i) = (n == 0 ? 1.0 : 0.0);
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion number operator as dense.");
                }
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        if (n == 0)
                        {
                            ++nnz;
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
                        size_t n = op->get_occupation(i, index);
                        if (n == 0)
                        {
                            buffer[counter] = T(1.0);
                            colind[counter] = i;
                            ++counter;
                        }
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion number operator as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        mat(i, i) = (n == 0 ? 1.0 : 0.0);
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct fermion number operator as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("v"));
                return ret;
            }
        };

        template <typename T>
        class jordan_wigner : public single_site_operator<T>
        {
        public:
            jordan_wigner() {}

            virtual bool is_sparse() const { return true; }
            virtual bool is_diagonal() const { return true; }
            virtual bool changes_parity() const { return false; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::diagonal_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        mat(i, i) = (n == 0 ? 1.0 : -1.0);
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct jordan_wigner as dense.");
                }
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    mat.resize(op->nstates(), op->nstates(), op->nstates());
                    auto rowptr = mat.rowptr();
                    rowptr[0] = 0;
                    auto colind = mat.colind();
                    auto buffer = mat.buffer();

                    size_t counter = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        buffer[counter] = (n == 0 ? 1.0 : -1.0);
                        colind[counter] = i;
                        ++counter;
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct jordan_wigner as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create fermion operator matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        size_t n = op->get_occupation(i, index);
                        mat(i, i) = (n == 0 ? 1.0 : -1.0);
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct jordan_wigner as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("jw"));
                return ret;
            }
        };
    } // namespace fermion
} // namespace ttns

#endif // PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_FERMIONIC_OPERATOR_HPP_
