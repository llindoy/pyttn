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

#ifndef PYTTN_UTILS_QUADRATURE_GAUSSIAN_QUADRATURE_GAUSS_JACODI_QUADRATURE_HPP_
#define PYTTN_UTILS_QUADRATURE_GAUSSIAN_QUADRATURE_GAUSS_JACODI_QUADRATURE_HPP_

#include <cstdint>
#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>

namespace utils
{
    namespace quad
    {
        namespace gauss
        {

            template <typename T, typename backend>
            class jacobi_quad;

            template <typename T>
            class jacobi_quad<T, linalg::blas_backend>
            {
                using RT = typename linalg::get_real_type<T>::type;

            public:
                template <typename F>
                static auto impl(F &&f, RT a, RT b, RT /* alpha */, RT /* beta */, const linalg::vector<RT, linalg::blas_backend> &w, const linalg::vector<RT, linalg::blas_backend> &x) -> decltype(f(a))
                {
                    RT c = (b - a) / 2.0;
                    RT d = (b + a) / 2.0;
                    decltype(f(a)) ret = 0.0;
                    for (size_t i = 0; i < w.size(); ++i)
                    {
                        RT xp = c * x(i) + d;
                        ret += f(xp) * w(i);
                    }
                    return ret * c; // std::pow(c, alpha+beta+1);
                }
            };

            // constructs and applies a gaussian quadrature rule of a given order for a specific orthogonal polynomial
            template <typename T, typename backend = linalg::blas_backend>
            class jacobi
            {
                using RT = typename linalg::get_real_type<T>::type;

            public:
                jacobi(size_t N, RT alpha, RT beta, bool normalise_weights = true) : m_w(N), m_x(N), m_alpha(alpha), m_beta(beta), m_normalise_weights(normalise_weights)
                {
                    ASSERT(alpha > -1 && beta > -1, "Invalid parameters to jacobi polynomial quadrature engine.");
                    construct_quadrature_rule();
                }

                void construct_quadrature_rule()
                {
                    linalg::symmetric_tridiagonal_matrix<RT, linalg::blas_backend> poly_recurrence(m_w.size(), m_w.size());
                    linalg::eigensolver<linalg::symmetric_tridiagonal_matrix<RT, linalg::blas_backend>> eigsolver(m_w.size());
                    linalg::vector<RT, linalg::blas_backend> vals(m_w.size());
                    linalg::matrix<RT, linalg::blas_backend> vecs(m_w.size(), m_w.size());

                    RT a = m_alpha;
                    RT b = m_beta;
                    if (std::abs(a + 0.5) < 1e-14 && std::abs(b + 0.5) < 1e-14)
                    {
                        for (size_t i = 0; i < m_w.size(); ++i)
                        {
                            poly_recurrence(i, i) = 0.0;
                            if (i + 1 < m_w.size())
                            {
                                poly_recurrence(i, i + 1) = 0.5;
                            }
                            poly_recurrence(0, 1) = 1 / sqrt(2.0);
                        }
                    }
                    else if (std::abs(a) < 1e-14 && std::abs(b) < 1e-14)
                    {
                        for (size_t i = 0; i < m_w.size(); ++i)
                        {
                            size_t k = i + 1;
                            poly_recurrence(i, i) = 0.0;
                            if (i + 1 < m_w.size())
                            {
                                poly_recurrence(i, i + 1) = sqrt((4 * k * (k + a) * (k + b) * (k + a + b)) / ((2 * k + a + b - 1) * (2 * k + a + b) * (2 * k + a + b) * (2 * k + a + b + 1)));
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < m_w.size(); ++i)
                        {
                            size_t k = i + 1;
                            poly_recurrence(i, i) = (b * b - a * a) / ((2 * k + a + b) * (2 * k + a + b - 2));
                            if (i + 1 < m_w.size())
                            {
                                poly_recurrence(i, i + 1) = sqrt((4 * k * (k + a) * (k + b) * (k + a + b)) / ((2 * k + a + b - 1) * (2 * k + a + b) * (2 * k + a + b) * (2 * k + a + b + 1)));
                            }
                        }
                    }

                    eigsolver(poly_recurrence, vals, vecs);
                    m_x = vals;
                    for (size_t i = 0; i < m_w.size(); ++i)
                    {
                        vals(i) = vecs(0, i) * vecs(0, i) * (m_normalise_weights ? weight_normalisation() : 1.0);
                    }
                    m_w = vals;
                }

                void resize(size_t N)
                {
                    if (N != m_w.size())
                    {
                        m_w.resize(N);
                        m_x.resize(N);
                        construct_quadrature_rule();
                    }
                }

                size_t size() const { return m_w.size(); }

                const RT &alpha() const { return m_alpha; }

                void set_alpha(const RT &alpha)
                {
                    ASSERT(alpha > -1, "Invalid alpha parameter for jacobi polynomial.");
                    if (alpha != m_alpha)
                    {
                        m_alpha = alpha;
                        construct_quadrature_rule();
                    }
                }

                const RT &beta() const { return m_beta; }

                void set_beta(const RT &beta)
                {
                    ASSERT(beta > -1, "Invalid beta parameter for jacobi polynomial.");
                    if (beta != m_beta)
                    {
                        m_beta = beta;
                        construct_quadrature_rule();
                    }
                }

                template <typename F>
                auto operator()(F &&f, RT a = -1, RT b = 1) const -> decltype(f(a))
                {
                    RT aa = a;
                    RT bb = b;
                    if (a > b)
                    {
                        bb = a;
                        aa = b;
                    }

                    return jacobi_quad<T, backend>::impl(std::forward<F>(f), aa, bb, m_alpha, m_beta, m_w, m_x) * (m_normalise_weights ? 1.0 : weight_normalisation());
                }

                const linalg::vector<RT, backend> &w() const { return m_w; }
                const linalg::vector<RT, backend> &x() const { return m_x; }
                RT w(size_t i, RT a = -1, RT b = 1) const
                {
                    ASSERT(i < m_w.size(), "Index out of bounds.");
                    RT aa = a;
                    RT bb = b;
                    if (a > b)
                    {
                        bb = a;
                        aa = b;
                    }

                    RT c = (b - a) / 2.0;
                    return std::pow(c, m_alpha + m_beta + 1) * m_w(i) * (m_normalise_weights ? 1.0 : weight_normalisation());
                }

                RT x(size_t i, RT a = -1, RT b = 1) const
                {
                    ASSERT(i < m_x.size(), "Index out of bounds.");
                    RT aa = a;
                    RT bb = b;
                    if (a > b)
                    {
                        bb = a;
                        aa = b;
                    }

                    RT c = (b - a) / 2.0;
                    RT d = (b + a) / 2.0;
                    return c * m_x(i) + d;
                }

                RT weight_function(RT x) { return std::pow(1.0 - x, m_alpha) * std::pow(1.0 + x, m_beta); }

            protected:
                RT weight_normalisation() const
                {
                    return std::exp((m_alpha + m_beta + 1) * std::log(2.0) + std::lgamma(m_alpha + 1) + std::lgamma(m_beta + 1) - std::lgamma(m_alpha + m_beta + 2));
                }

                // the quadrature points
                linalg::vector<RT, backend> m_w;
                linalg::vector<RT, backend> m_x;

                RT m_alpha;
                RT m_beta;

                bool m_normalise_weights;
            };

        }
    }
}

#endif // PYTTN_UTILS_QUADRATURE_GAUSSIAN_QUADRATURE_GAUSS_JACODI_QUADRATURE_HPP_
