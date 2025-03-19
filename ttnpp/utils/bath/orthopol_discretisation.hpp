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

#ifndef PYTTN_UTILS_BATH_ORTHOPOL_DISCRETISATION_HPP_
#define PYTTN_UTILS_BATH_ORTHOPOL_DISCRETISATION_HPP_

#include <type_traits>
#include <linalg/linalg.hpp>
#include <cmath>
#include <limits>

#include "../quadrature/adaptive_integrate.hpp"
#include "../orthopol.hpp"
#include "../find_root.hpp"

namespace utils
{

    // a function for applying the AAA algorithm to
    class orthopol_discretisation
    {
    public:
        template <typename Jfunc, typename real_type>
        static void get_poly(Jfunc &&J, orthopol<real_type> &poly, orthopol<real_type> &n_poly, real_type wrange = 1.0, real_type moment_scaling = 1.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100)
        {
            ASSERT(poly.Nmax() % 2 == 0, "Polynomial must have at least 2N terms.");
            size_t N = poly.Nmax() / 2;

            CALL_AND_HANDLE(
                nonclassical_polynomial(n_poly, poly, N, [&J, wrange, moment_scaling](real_type x)
                                        { return J(x * wrange / (2 * moment_scaling)) / M_PI; }, rtol, atol, real_type(1.0), nquad),
                "Failed to construct nonclassical polynomials from weight function.");

            n_poly.scale(wrange / (2 * moment_scaling));
        }

        template <typename Jfunc, typename real_type, typename T>
        static void get_moments(Jfunc &&J, orthopol<real_type> &poly, std::vector<T> &moments, real_type wrange = 1.0, real_type moment_scaling = 1.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100, real_type minbound = 0, real_type maxbound = std::numeric_limits<real_type>::max())
        {
            ASSERT(poly.Nmax() % 2 == 0, "Polynomial must have at least 2N terms.");
            moments.resize(poly.Nmax());
            size_t N = poly.Nmax() / 2;

            CALL_AND_HANDLE(
                polynomial_moments(moments, poly, N, [&J, wrange, moment_scaling](real_type x)
                                   { return J(x * wrange / (2 * moment_scaling)) / M_PI; }, rtol, atol, real_type(1.0), nquad, 100, minbound, maxbound),
                "Failed to construct nonclassical polynomials from weight function.");
        }

    public:
        template <typename Jfunc, typename real_type, typename T>
        static void chain_map(Jfunc &&J, orthopol<real_type> &poly, std::vector<T> &t, std::vector<real_type> &e, real_type wrange = 1.0, real_type moment_scaling = 1.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100)
        {
            orthopol<real_type> n_poly;
            size_t N = poly.Nmax() / 2;
            CALL_AND_RETHROW(get_poly(std::forward<Jfunc>(J), poly, n_poly, wrange, moment_scaling, atol, rtol, nquad));
        }

        template <typename Jfunc, typename real_type, typename T>
        static void chain_map(Jfunc &&J, real_type wmin, real_type wmax, size_t N, std::vector<T> &t, std::vector<real_type> &e, real_type moment_scaling = 1.0, real_type alpha = 1.0, real_type beta = 0.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100)
        {
            ASSERT(wmax > wmin, "wmax <= wmin");
            real_type wrange = (wmax - wmin);

            orthopol<real_type> jacobi;
            jacobi_polynomial(jacobi, 2 * N, alpha, beta);
            jacobi.shift((wmax + wmin) / wrange);
            jacobi.scale(moment_scaling);

            CALL_AND_HANDLE(chain_map(std::forward<Jfunc>(J), jacobi, t, e, wrange, moment_scaling, atol, rtol, nquad), "Failed to chain map bath.");
        }

    public:
        template <typename Jfunc, typename real_type, typename T>
        static void discretise(Jfunc &&J, orthopol<real_type> &poly, std::vector<T> &g, std::vector<real_type> &w, real_type wrange = 1.0, real_type moment_scaling = 1.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100)
        {
            orthopol<real_type> n_poly;
            size_t N = poly.Nmax() / 2;
            CALL_AND_RETHROW(get_poly(std::forward<Jfunc>(J), poly, n_poly, wrange, moment_scaling, atol, rtol, nquad));

            n_poly.compute_nodes_and_weights();

            std::vector<std::pair<T, real_type>> wg(N);
            for (size_t i = 0; i < N; ++i)
            {
                std::get<0>(wg[i]) = n_poly.nodes()(i);
                std::get<1>(wg[i]) = std::sqrt(n_poly.weights()(i));
            }

            std::sort(wg.begin(), wg.end(), [](const std::pair<T, real_type> &a, const std::pair<T, real_type> &b)
                      { return linalg::abs(std::get<0>(a)) < linalg::abs(std::get<0>(b)); });

            g.resize(N);
            w.resize(N);

            for (size_t i = 0; i < N; ++i)
            {
                w[i] = std::get<0>(wg[i]);
                g[i] = std::get<1>(wg[i]);
            }
        }

        template <typename Jfunc, typename real_type, typename T>
        static void discretise(Jfunc &&J, real_type wmin, real_type wmax, size_t N, std::vector<T> &g, std::vector<real_type> &w, real_type moment_scaling = 1.0, real_type alpha = 1.0, real_type beta = 0.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100)
        {
            ASSERT(wmax > wmin, "wmax <= wmin");
            real_type wrange = (wmax - wmin);

            orthopol<real_type> jacobi;
            jacobi_polynomial(jacobi, 2 * N, alpha, beta);
            jacobi.shift((wmax + wmin) / wrange);
            jacobi.scale(moment_scaling);

            CALL_AND_HANDLE(discretise(std::forward<Jfunc>(J), jacobi, g, w, wrange, moment_scaling, atol, rtol, nquad), "Failed to discretise bath.");
        }

    public:
        template <typename Jfunc, typename real_type, typename T>
        static void get_modified_moments(Jfunc &&J, orthopol<real_type> &poly, std::vector<T> &moments, real_type wrange = 1.0, real_type moment_scaling = 1.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100, real_type minbound = 0, real_type maxbound = std::numeric_limits<real_type>::max())
        {
            CALL_AND_RETHROW(get_moments(std::forward<Jfunc>(J), poly, moments, wrange, moment_scaling, atol, rtol, nquad, minbound, maxbound));
        }

        template <typename Jfunc, typename real_type, typename T>
        static void get_modified_moments(Jfunc &&J, real_type wmin, real_type wmax, size_t N, std::vector<T> &moments, real_type moment_scaling = 1.0, real_type alpha = 1.0, real_type beta = 0.0, real_type atol = 1e-12, real_type rtol = 1e-14, size_t nquad = 100, real_type minbound = 0, real_type maxbound = std::numeric_limits<real_type>::max())
        {
            ASSERT(wmax > wmin, "wmax <= wmin");
            real_type wrange = (wmax - wmin);

            orthopol<real_type> jacobi;
            jacobi_polynomial(jacobi, 2 * N, alpha, beta);
            jacobi.shift((wmax + wmin) / wrange);
            jacobi.scale(moment_scaling);

            CALL_AND_HANDLE(get_modified_moments(std::forward<Jfunc>(J), jacobi, moments, wrange, moment_scaling, atol, rtol, nquad, minbound, maxbound), "Failed to discretise bath.");
        }
    };

}

#endif // PYTTN_UTILS_BATH_ORTHOPOL_DISCRETISATION_HPP_
