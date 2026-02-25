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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_CSR_MATRIX_HPP_
#define PYTTN_LINALG_TENSOR_SPARSE_CSR_MATRIX_HPP_

#include <vector>
#include <tuple>
#include <algorithm>

#include "../../../backends/blas/blas_backend.hpp"
#include "../csr_matrix.hpp"

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/types/array.hpp>
#endif

namespace linalg
{

    template <typename T>
    class csr_matrix<T, blas_backend> : public csr_matrix_base<csr_matrix<T, blas_backend>>
    {
    public:
        using self_type = csr_matrix<T, blas_backend>;
        using base_type = csr_matrix_base<self_type>;
        using coo_type = typename base_type::coo_type;
        using size_type = typename base_type::size_type;
        using real_type = typename base_type::real_type;
        using index_type = typename base_type::index_type;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &out, const csr_matrix<U, blas_backend> &mat);

    public:
        template <typename... Args>
        csr_matrix(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct csr matrix object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        inline size_type nnz_in_row(size_type i) const
        {
            ASSERT(i < this->nrows(), "Unable to get number of terms in row.  Index out of bounds.");
            auto rowptr = this->rowptr();
            return (rowptr[i + 1] - rowptr[i]);
        }

        inline bool contains_diagonal(size_type i) const
        {
            ASSERT(i < this->nrows(), "Unable to get number of terms in row.  Index out of bounds.");
            auto rowptr = this->rowptr();
            auto colind = this->colind();
            for (size_type j = static_cast<size_type>(rowptr[i]); j < static_cast<size_type>(rowptr[i + 1]); ++j)
            {
                if (static_cast<size_t>(colind[j]) == i)
                {
                    return true;
                }
            }
            return false;
        }

        // a function for pruning zeros from the csr matrix.  This iterates over the tree and if a value has magnitude less than the tolerance we remove it.
        // This doesn't change the size of any buffers at all
        inline void prune(real_type tol = 1e-12)
        {
            if (tol > 0)
            {
                size_type counter = 0;
                size_t rpi = 0;

                auto buffer = this->buffer();
                auto rowptr = this->rowptr();
                auto colind = this->colind();
                for (size_type i = 0; i < this->nrows(); ++i)
                {
                    size_t rpi1 = static_cast<size_type>(rowptr[i + 1]);
                    for (size_type j = rpi; j < rpi1; ++j)
                    {
                        // if the absolute value of the current value type is greater than the pruning tolerance we are going to reinsert it
                        // at position counter and incement counter. If a term isn't greater than the pruning tolerance then we are not incrementing
                        // counter and so at a later stage it will be overwritten.
                        if (tol <= std::abs(buffer[j]))
                        {
                            buffer[counter] = buffer[j];
                            colind[counter] = colind[j];
                            ++counter;
                        }
                    }
                    rowptr[i + 1] = counter;
                    rpi = rpi1;
                }
                this->resize(counter);
            }
        }

        inline void transpose(csr_matrix<T, blas_backend> &o, T alpha = T(1)) const
        {
            transpose(*this, o, alpha);
        }

        static inline void transpose(const csr_matrix<T, blas_backend> &imat, csr_matrix<T, blas_backend> &o, T alpha = T(1))
        {
            CALL_AND_HANDLE(o.resize(imat.nnz(), imat.ncols(), imat.nrows()), "Failed to allocate storage for new array.");
            auto ibuffer = imat.buffer();
            auto irowptr = imat.rowptr();
            auto icolind = imat.colind();

            auto obuffer = o.buffer();
            auto orowptr = o.rowptr();
            auto ocolind = o.colind();

            // pad the rowptr array with zeros
            std::fill(orowptr, orowptr + (imat.ncols() + 1), index_type(0));
            // iterate over each term in the buffer and use its column index to set the rowptr of the output result
            for (size_type i = 0; i < imat.nnz(); ++i)
            {
                // get the column index in the original array and increment the rowptr object of the transposed array corresponding to this index
                ++orowptr[icolind[i] + 1];
            }
            // now increment the rowptr objects so that it is the cumulative sum rather than number of elements in row.
            // Following this step the output rowptr is completed, but we will be editing it in the next step
            for (size_type i = 0; i < imat.ncols(); ++i)
            {
                orowptr[i + 1] += orowptr[i];
            }

            size_type rpi = 0;
            // now set up the output data and column indices.  We do this by iterating over each row of the original array.
            for (size_type i = 0; i < imat.nrows(); ++i)
            {
                size_type rpi1 = static_cast<size_type>(irowptr[i + 1]);
                for (size_type j = rpi; j < rpi1; ++j)
                {
                    // for this column index.  Get the current output rowptr value.  Incrementing the results after
                    // we have extracted the value.  Now set the output value and column index
                    index_type index = orowptr[icolind[j]]++;
                    obuffer[index] = alpha * ibuffer[j];
                    ocolind[index] = i;
                }
                rpi = rpi1;
            }

            // now each output rowptr has been incremented by the number of elements in the row so we need to shift everything
            // along to get the correct structure
            for (size_t i = 0; i < imat.ncols(); ++i)
            {
                size_t j = imat.ncols() - i;
                orowptr[j] = orowptr[j - 1];
            }
            orowptr[0] = 0;
        }

        inline tensor<T, 2, blas_backend> todense() const
        {
            tensor<T, 2, blas_backend> ret(this->shape(0), this->shape(1), T(0));
            auto buffer = this->buffer();
            auto rowptr = this->rowptr();
            auto colind = this->colind();

            size_type rpi = 0;
            for (size_t i = 0; i < this->nrows(); ++i)
            {
                size_type rpi1 = static_cast<size_type>(rowptr[i + 1]);
                for (size_type j = rpi; j < rpi1; ++j)
                {
                    index_type index = colind[j];
                    ret(i, index) = buffer[j];
                }
                rpi = rpi1;
            }
            return ret;
        }
    }; // csr_matrix<T, blas_backend>

    template <typename T>
    std::ostream &operator<<(std::ostream &out, const csr_matrix<T, blas_backend> &mat)
    {
        using size_type = typename csr_matrix<T, blas_backend>::size_type;
        using const_index_pointer = typename csr_matrix<T, blas_backend>::const_index_pointer;
        const_index_pointer rowptr = mat.rowptr();
        const_index_pointer colind = mat.colind();
        for (size_type i = 0; i < mat.nrows(); ++i)
        {
            for (size_type j = static_cast<size_type>(rowptr[i]); j < static_cast<size_type>(rowptr[i + 1]); ++j)
            {
                out << i << " " << colind[j] << " " << mat.m_vals[j] << std::endl;
            }
        }
        return out;
    }
} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_CSR_MATRIX_HPP_//
