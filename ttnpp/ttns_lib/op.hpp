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

#ifndef PYTTN_TTNS_LIB_OP_HPP_
#define PYTTN_TTNS_LIB_OP_HPP_

#include <linalg/linalg.hpp>

namespace ttns
{

    template <typename T, typename backend = linalg::blas_backend>
    class Op
    {
    public:
        using real_type = typename linalg::get_real_type<T>::type;
        using size_type = typename linalg::traits<backend>::size_type;

        Op() {}
        Op(const Op &o) : m_op(o.m_op), m_indices(o.m_indices), m_dims(o.m_dims) {}
        Op(Op &&o) noexcept : m_op(std::move(o.m_op)), m_indices(std::move(o.m_indices)), m_dims(std::move(o.m_dims)) {}
        template <typename U, typename be>
        Op(const Op<U, be> &other) : m_op(other.matrix()), m_indices(other.indices()), m_dims(other.dims()){}
        template <typename I1, typename I2>
        Op(const linalg::matrix<T, backend> &m, const std::vector<I1> &indices, const std::vector<I2> &dims)
        {
            ASSERT(indices.size() > 0, "Failed to construct operator.  Must specify at least one index.");
            ASSERT(indices.size() == dims.size(), "Failed to construct operator the dimensions and indices arrays are not the same size.");

            std::set<I1> s(indices.begin(), indices.end());
            ASSERT(s.size() == indices.size(), "Failed to construct operator.  Repeated index.")

            size_t index = indices[0];
            for(size_t i = 1; i < indices.size(); ++i)
            {
                ASSERT(index < indices[i], "The op object requires indices to be specified in ascending order.");
                index = indices[i];
            }

            size_type prod = 1;
            for (size_type i = 0; i < dims.size(); ++i)
            {
                ASSERT(dims[i] > 0 && indices[i] >= 0, "Failed to construct object indicees or dims invalid.");
                prod *= dims[i];
            }
            ASSERT(prod == m.shape(0) && prod == m.shape(1), "Dimensions array is not compatible with specified matrix.");

            m_op = m;
            m_indices.resize(indices.size());
            m_dims.resize(indices.size());
            for (size_type i = 0; i < indices.size(); ++i)
            {
                m_indices[i] = indices[i];
                m_dims[i] = dims[i];
            }
        }

        template <typename I1, typename I2>
        Op(const linalg::matrix<T, backend> &m, std::initializer_list<I1> indices, std::initializer_list<I2> dims)
        try : Op(m, std::vector<I1>(indices), std::vector<I2>(dims))
        {
        }
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct Op object.");
        }

        ~Op() {}

        template <typename U, typename be>
        Op &operator=(const Op<U, be> &other)
        {
            m_op = other.matrix();
            m_indices = other.indices();
            m_dims = other.dims();
            return *this;
        }

        Op &&operator=(Op &&other)
        {
            m_op = std::move(other.m_op);
            m_indices = std::move(other.m_indices);
            m_dims = std::move(other.m_dims);
            return *this;
        }

        Op &set_buffer(const T *src, size_type n)
        {
            CALL_AND_RETHROW(m_op.set_buffer(src, n));
            return *this;
        }

        void clear()
        {
            m_op.clear();
            m_indices.clear();
            m_dims.clear();
        }

        const linalg::matrix<T, backend> &operator()() const { return m_op; }
        linalg::matrix<T, backend> &operator()() { return m_op; }

        const linalg::matrix<T, backend> &matrix() const { return m_op; }
        linalg::matrix<T, backend> &matrix() { return m_op; }

        const std::vector<size_type> &indices() const { return m_indices; }
        std::vector<size_type> &indices() { return m_indices; }

        const std::vector<size_type> &dims() const { return m_dims; }
        std::vector<size_type> &dims() { return m_dims; }

        size_type nmodes() const { return m_dims.size(); }
        size_type ndim() const { return m_dims.size(); }
        size_type size() const {return m_op.shape(0);}

        bool valid_parameters() const
        {
            if(m_indices.size() != m_dims.size()){return false;}

            std::set<size_t> s(m_indices.begin(), m_indices.end());
            if(s.size() != m_indices.size()){return false;}
            ASSERT(s.size() == m_indices.size(), "Failed to construct operator.  Repeated index.");

            size_t index = m_indices[0];
            for(size_t i = 1; i < m_indices.size(); ++i)
            {
                if(index >= m_indices[i]){return false;}
                index = m_indices[i];
            }

            size_type prod = 1;
            for (size_type i = 0; i < m_dims.size(); ++i)
            {
                if(m_dims[i] < 1 ){return false;}
                prod *= m_dims[i];
            }
            if(prod != m_op.shape(0) || prod != m_op.shape(1)){return false;}
            return true;
        }

        // function for converting the operator into a list of vector which when contracted together give this vector
        std::vector<linalg::tensor<T, 4, backend>> as_mpo(int nbmax = -1, real_type tol = -1) const
        {
            ASSERT(this->valid_parameters(), "Failed to form mpo representation of op object. The op object is not correctly formed.")

            std::vector<size_type> strides(m_dims.size());
            strides[m_dims.size() - 1] = 1;
            for (size_type i = 1; i < m_dims.size(); ++i)
            {
                strides[m_dims.size() - (i + 1)] = strides[m_dims.size() - i] * m_dims[m_dims.size() - i];
            }

            // setup the vector of rank 4 tensors that will store the result
            std::vector<linalg::tensor<T, 4, backend>> ret(m_dims.size());

            if (m_dims.size() == 1)
            {
                ret[0] = linalg::tensor<T, 4, backend>(1, m_dims[0], m_dims[0], 1);
                ret[0].set_buffer(m_op);

            }
            else
            {
                size_type dims1 = 1;
                linalg::matrix<T, backend> Umat(m_op.shape(0), m_op.shape(1));

                Umat.set_buffer(m_op);

                linalg::tensor<T, 5, backend> Umt(dims1, m_dims[0], m_dims[0], strides[0], strides[0]);

                linalg::singular_value_decomposition<linalg::matrix<T, backend>, false> m_svd;

                linalg::matrix<T, backend> _U, _Vh;
                linalg::diagonal_matrix<real_type, backend> S;
                for (size_type i = 0; i + 1 < m_dims.size(); ++i)
                {
                    // swap the tensor indices so that it is in the correct format to SVD
                    auto Ut = Umat.reinterpret_shape(dims1, m_dims[i], strides[i], m_dims[i], strides[i]);
                    Umt = linalg::trans(Ut, {0, 1, 3, 2, 4});

                    // reshape the current tensor to a matrix of the correct dimension
                    auto Um = Umt.reinterpret_shape(dims1 * m_dims[i] * m_dims[i], strides[i] * strides[i]);

                    // now compute the singular values decomposition of the matrix representation of the tensor
                    CALL_AND_HANDLE(m_svd(Um, S, _U, _Vh, false), "Failed to compute singular values decomposition of MPO.")

                    // now attempt to truncate the result if needed
                    if (nbmax >= 0 || tol >= 0)
                    {
                    }

                    // and extract the site tensor from it while passing the result on to be contracted
                    size_type dims2 = _Vh.shape(0);

                    ret[i] = linalg::tensor<T, 4, backend>(dims1, m_dims[i], m_dims[i], dims2);
                    ret[i].set_buffer(_U);

                    // reshape Ut so that it can take the result of the required contraction
                    Umat.resize(_Vh.shape(0), _Vh.shape(1));
                    Umat = S * _Vh;

                    dims1 = dims2;
                }

                // now we handle the final matrix
                size_type ind = m_dims.size() - 1;
                size_type d = m_dims[ind];
                ret[ind].resize(dims1, d, d, 1);
                ret[ind].set_buffer(Umat);

            }
            return ret;
        }

        std::ostream &print(std::ostream &os) const
        {
            os << "op: [" << m_op << " ";
            for (size_t i = 0; i < m_indices.size(); ++i)
            {
                os << "ind: " << m_indices[i] << ", dim: " << m_dims[i] << (i + 1 != this->ndim() ? ", " : "]");
            }
            return os;
        }
    protected:
        linalg::matrix<T, backend> m_op;
        std::vector<size_type> m_indices;
        std::vector<size_type> m_dims;
    };

    template <typename T, typename backend>
    std::ostream &operator<<(std::ostream &os, const Op<T, backend> &t)
    {
        return t.print(os);
    }

}

#endif // PYTTN_TTNS_LIB_OP_HPP_
