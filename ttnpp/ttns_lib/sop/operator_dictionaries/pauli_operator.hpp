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

#ifndef PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_PAULI_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_PAULI_OPERATOR_HPP_

#include <linalg/linalg.hpp>
#include "single_site_operator.hpp"

// TODO: Need to alter the functions for forming the operators so that if the operator is explicitly complex valued attempting to
// initialise it with a real variable leads to a runtime error not a compile time error.
namespace ttns
{
    namespace pauli
    {

        template <typename T>
        class sigma_p : public single_site_operator<T>
        {
        public:
            sigma_p() {}
            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_+ as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
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
                    RAISE_EXCEPTION("Failed to construct sigma_+ as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
                ASSERT(index < op->nmodes(), "Index out of bounds.");
                mat.resize(op->nstates(), op->nstates());
                mat.fill_zeros();

                try
                {
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_raised_state(i, index))
                        {
                            mat(i, op->get_raised_index(i, index)) = 1.0;
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct sigma_+ as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("s-"));
                return ret;
            }
        };

        template <typename T>
        class sigma_m : public single_site_operator<T>
        {
        public:
            sigma_m() {}

            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_- as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
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
                    RAISE_EXCEPTION("Failed to construct sigma_- as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            mat(i, op->get_lowered_index(i, index)) = 1.0;
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct sigma_- as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("s+"));
                return ret;
            }
        };

        template <typename T>
        class sigma_x : public single_site_operator<T>
        {
        public:
            sigma_x() {}

            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_x as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            ++nnz;
                        }
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
                        if (op->contains_lowered_state(i, index))
                        {
                            buffer[counter] = 1.0;
                            colind[counter] = op->get_lowered_index(i, index);
                            ++counter;
                        }
                        if (op->contains_raised_state(i, index))
                        {
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
                    RAISE_EXCEPTION("Failed to construct sigma_x as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            mat(i, op->get_lowered_index(i, index)) = 1.0;
                        }
                        if (op->contains_raised_state(i, index))
                        {
                            mat(i, op->get_raised_index(i, index)) = 1.0;
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct sigma_x as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("sx"));
                return ret;
            }
        };

        template <typename T, bool is_complex = linalg::is_complex<T>::value>
        class sigma_y;

        template <typename T>
        class sigma_y<T, false> : public single_site_operator<T>
        {
        public:
            sigma_y() {}

            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_y as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::csr_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_y as a real valued operator.");
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_y as a real valued operator.");
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(-1), std::string("sy"));
                return ret;
            }
        };

        template <typename RT>
        class sigma_y<linalg::complex<RT>, true> : public single_site_operator<linalg::complex<RT>>
        {
        public:
            using T = linalg::complex<RT>;

        public:
            sigma_y() {}

            virtual bool is_sparse() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> & /* op */, size_t /* index */, linalg::diagonal_matrix<T> & /* mat */) const
            {
                RAISE_EXCEPTION("Cannot form sigma_y as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());

                    size_t nnz = 0;
                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            ++nnz;
                        }
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
                        if (op->contains_lowered_state(i, index))
                        {
                            buffer[counter] = T(0, 1.0);
                            colind[counter] = op->get_lowered_index(i, index);
                            ++counter;
                        }
                        if (op->contains_raised_state(i, index))
                        {
                            buffer[counter] = T(0.0, -1.0);
                            colind[counter] = op->get_raised_index(i, index);
                            ++counter;
                        }
                        rowptr[i + 1] = counter;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct sigma_y as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
                    ASSERT(index < op->nmodes(), "Index out of bounds.");
                    mat.resize(op->nstates(), op->nstates());
                    mat.fill_zeros();

                    for (size_t i = 0; i < op->nstates(); ++i)
                    {
                        if (op->contains_lowered_state(i, index))
                        {
                            mat(i, op->get_lowered_index(i, index)) = T(0, 1);
                        }
                        if (op->contains_raised_state(i, index))
                        {
                            mat(i, op->get_raised_index(i, index)) = T(0, -1);
                        }
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to construct sigma_y as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(-1), std::string("sy"));
                return ret;
            }
        };

        template <typename T>
        class sigma_z : public single_site_operator<T>
        {
        public:
            sigma_z() {}

            virtual bool is_sparse() const { return true; }
            virtual bool is_diagonal() const { return true; }

            virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::diagonal_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
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
                    RAISE_EXCEPTION("Failed to construct sigma_z as dense.");
                }
            }

            virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::csr_matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
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
                    RAISE_EXCEPTION("Failed to construct sigma_z as csr.");
                }
            }

            virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis> &op, size_t index, linalg::matrix<T> &mat) const
            {
                try
                {
                    ASSERT(op->dim(index) == 2, "Unable to create pauli matrix for mode with dimension greater than 2.");
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
                    RAISE_EXCEPTION("Failed to construct sigma_z as dense.");
                }
            }
            virtual std::pair<T, std::string> transpose() const
            {
                std::pair<T, std::string> ret = std::make_pair(T(1), std::string("sz"));
                return ret;
            }
        };

    } // namespace pauli

} // namespace ttns
#endif // PYTTN_TTNS_LIB_SOP_OPERATOR_DICTIONARIES_PAULI_OPERATOR_HPP_
