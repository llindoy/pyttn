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

#ifndef PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_BICGSTABL_HPP_
#define PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_BICGSTABL_HPP_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include "preconditioners/preconditioners.hpp"

/**
 *  This file contains the functions required to solve a sparse system of linear
 * systems using the BiCGSTAB(l) algorithm. These functions act on the
 * underlying arrays of the tensors rather than on the Tensor objects themselves
 */

namespace utils
{

    template <typename T,
              template <typename, typename> class PCtype = preconditioner::identity,
              typename backend = linalg::blas_backend>
    class bicgstabl
    {
        using backend_type = backend;
        using value_type = T;
        using real_type = typename linalg::get_real_type<T>::type;

    protected:
        static_assert(std::is_base_of<preconditioner::preconditioner<T, backend>,
                                      PCtype<T, backend>>::value,
                      "Invalid preconditioner");
        PCtype<T, backend> _pc;

        size_t _L;
        size_t _N;
        linalg::matrix<T, backend> _u;
        linalg::matrix<T, backend> _r;
        linalg::vector<T, backend> _rtilde;

        linalg::vector<T> _gamma;
        linalg::vector<T> _gammap;
        linalg::vector<T> _gammapp;

        linalg::vector<T> _sigma;
        linalg::matrix<T> _tau;

        linalg::vector<real_type> m_residues;

        real_type _tol = std::numeric_limits<real_type>::epsilon() * 1e3;
        size_t _max_iter = 20;

    public:
        bicgstabl() {}
        bicgstabl(size_t L, size_t N) { resize(L, N); }

        template <typename... Args>
        bicgstabl(size_t L, size_t N, Args &&...args)
            : _pc(std::forward<Args>(args)...)
        {
            resize(L, N);
        }

        bicgstabl(const bicgstabl &o) = default;
        bicgstabl(bicgstabl &&o) = default;

        bicgstabl &operator=(const bicgstabl &o) = default;
        bicgstabl &operator=(bicgstabl &&o) = default;

        template <typename... Args>
        void initialise_preconditioner(Args &&...args)
        {
            CALL_AND_RETHROW(_pc.initialise(std::forward<Args>(args)...));
        }

        const real_type &tol() const { return _tol; }
        real_type &tol() { return _tol; }

        const size_t &max_iter() const { return _max_iter; }
        size_t &max_iter() { return _max_iter; }

        const size_t &niter() const { return m_residues.size(); }

        const linalg::vector<real_type> &residues() const { return m_residues; }
        const real_type &residue(size_t i) const
        {
            ASSERT(i < m_residues.size(), "Index out of bounds.");
            return m_residues[i];
        }

        void resize(size_t L, size_t N)
        {
            try
            {
                _L = L;
                _N = N;
                _u.resize(L + 1, N);
                _r.resize(L + 1, N);
                _rtilde.resize(N);

                _gamma.resize(L + 1);
                _gammap.resize(L + 1);
                _gammapp.resize(L + 1);
                _sigma.resize(L + 1);
                _tau.resize(L, L);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to resize bicgstab(l) engine.");
            }
        }

        void clear()
        {
            _L = 0;
            _N = 0;
            m_residues.clear();
            _u.clear();
            _r.clear();
            _rtilde.clear();
            _pc.clear();

            _gamma.clear();
            _gammap.clear();
            _gammapp.clear();
            _sigma.clear();
            _tau.clear();
        }

    public:
        template <typename vec_type, typename... Args>
        bool operator()(vec_type &x, const vec_type &b, Args &&...args)
        {
            try
            {
                ASSERT(b.size() == x.size(), "Failed to allocate vector.");
                if (_rtilde.size() != b.size())
                {
                    CALL_AND_HANDLE(this->resize(this->_L, x.size()),
                                    "Failed to resize bicgstabl object.");
                }

                if (m_residues.size() != _max_iter)
                {
                    m_residues.resize(_max_iter);
                }

                auto a2p = _r[0].reinterpret_shape(x.shape());
                auto rrtilde = _rtilde.reinterpret_shape(x.shape());
                CALL_AND_HANDLE(compute_action(x, a2p, std::forward<Args>(args)...),
                                "Failed to compute the partial krylov subspace.  Failed "
                                "to evaluate action on vector.");
                // r[0] = a2;

                CALL_AND_HANDLE(rrtilde = b - a2p, "Failed to compute rtilde");
                CALL_AND_HANDLE(_pc.apply(_rtilde), "Failed to apply preconditioner.");
                _r[0] = _rtilde;

                real_type bnrm =
                    std::sqrt(linalg::real(linalg::dot_product(linalg::conj(b), b)));
                if (bnrm == 0)
                    bnrm = 1.0;

                // normalise rtilde
                real_type normfac = 1.0 / sqrt(linalg::real(linalg::dot_product(
                                              linalg::conj(_rtilde), _rtilde)));
                _rtilde *= normfac;

                // setup the initial values required before we start the iteration
                _u.fill_zeros();
                T rho = 1.0, alpha = 0.0, omega = 1.0;

                const double bd_err_tol = 1e-30;
                size_t L = _L;

                size_t iter = 0;
                bool broken_down = false;

                // we keep iterating until the solution is converged, we have run into an
                // error or we have reached the maximum number of iterations
                while (std::sqrt(linalg::real(
                           linalg::dot_product(linalg::conj(_r[0]), _r[0]))) > _tol &&
                       iter < _max_iter && !broken_down)
                {
                    m_residues[iter] = std::sqrt(
                        linalg::real(linalg::dot_product(linalg::conj(_r[0]), _r[0])));
                    rho = -omega * rho; // update the value of rho used to compute the BiCG
                                        // coefficients

                    // Perform the BiCG part of the iteration
                    for (size_t j = 0; j < L; ++j)
                    {

                        // update the BiCG coefficients
                        T rho1 = linalg::dot_product(linalg::conj(_rtilde), _r[j]);
                        T beta = alpha * rho1 / rho;
                        rho = rho1;

                        // update the u search directions
                        for (size_t i = 0; i <= j; ++i)
                        {
                            _u[i] = _r[i] - beta * _u[i];
                        }

                        auto ruj = _u[j].reinterpret_shape(x.shape());
                        auto rujp = _u[j + 1].reinterpret_shape(x.shape());

                        CALL_AND_HANDLE(
                            compute_action(ruj, rujp, std::forward<Args>(args)...),
                            "Failed to compute the partial krylov subspace.  Failed to "
                            "evaluate action on vector.");
                        CALL_AND_HANDLE(_pc.apply(rujp), "Failed to apply preconditioner.");

                        T temp = linalg::dot_product(linalg::conj(_rtilde), _u[j + 1]);

                        // if the system has broken down then we terminate and print a warning
                        if (std::fabs(temp) <= bd_err_tol)
                        {
                            broken_down = true;
                            std::cerr << "Warning: bicgstab(l) reached break down condition."
                                      << std::endl;
                            return false;
                        }

                        alpha = rho / temp;

                        // update the residues
                        for (size_t i = 0; i <= j; ++i)
                        {
                            _r[i] -= alpha * _u[i + 1];
                        }

                        auto rrj = _r[j].reinterpret_shape(x.shape());
                        auto rrjp = _r[j + 1].reinterpret_shape(x.shape());

                        CALL_AND_HANDLE(
                            compute_action(rrj, rrjp, std::forward<Args>(args)...),
                            "Failed to compute the partial krylov subspace.  Failed to "
                            "evaluate action on vector.");
                        CALL_AND_HANDLE(_pc.apply(rrjp), "Failed to apply preconditioner.");

                        x += alpha * _u[0];
                    }

                    // Perform the MR part of the iteration.  Compute the minimum residue
                    // polynomial. start by performing a modified gram-schmidt
                    // orthogonalisation
                    for (size_t j = 1; j <= L; ++j)
                    {
                        for (size_t i = 1; i < j; ++i)
                        {
                            T tv = linalg::dot_product(linalg::conj(_r[i]), _r[j]) / _sigma[i];
                            _tau(j - 1, i - 1) = tv; // form the transpose of the tau operator
                                                     // here so that it is easier to use later
                            _r[j] -= tv * _r[i];
                        }
                        _sigma[j] = linalg::dot_product(linalg::conj(_r[j]), _r[j]);
                        _gammap[j] =
                            linalg::dot_product(linalg::conj(_r[j]), _r[0]) / _sigma[j];
                    }

                    // compute _gamma = T^{-1}_gammap
                    _gamma[L] = _gammap[L];
                    omega = _gamma[L];
                    for (size_t j = L - 1; j >= 1; --j)
                    {
                        _gamma[j] = _gammap[j];
                        for (size_t i = j + 1; i <= L; ++i)
                        {
                            _gamma[j] -= _tau(i - 1, j - 1) * _gamma[i];
                        }
                    }

                    // compute _gammapp = TS_gamma
                    for (size_t j = 1; j < L; ++j)
                    {
                        _gammapp[j] = _gamma[j + 1];
                        for (size_t i = j + 1; i < L; ++i)
                        {
                            _gammapp[j] += _tau(i - 1, j - 1) * _gamma[i + 1];
                        }
                    }

                    // update everything
                    x += _gamma[1] * _r[0];
                    _r[0] -= _gammap[L] * _r[L];
                    _u[0] -= _gamma[L] * _u[L];

                    for (size_t j = 1; j < L; ++j)
                    {
                        _u[0] -= _gamma[j] * _u[j];
                        x += _gammapp[j] * _r[j];
                        _r[0] -= _gammap[j] * _r[j];
                    }
                    ++iter;
                }

                m_residues.resize(iter);
                // return whether or not this converged or reached the maximum iteration
                // number
                return (iter < _max_iter && !broken_down);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to solve linear system of equations using "
                                "bicgstab(l) algorithm.");
            }
        }

    protected:
        // action of a matrix on a vector
        template <typename vec_type, typename mat_type, typename res_type>
        typename std::enable_if<linalg::is_dense_tensor<vec_type>::value &&
                                    linalg::is_linalg_object<mat_type>::value,
                                void>::type
        compute_action(const vec_type &vec, res_type &r, const mat_type &mat)
        {
            CALL_AND_RETHROW(r = mat * vec);
        }

        template <typename vtype, typename vec_type, typename mat_type,
                  typename res_type>
        typename std::enable_if<linalg::is_number<vtype>::value &&
                                    linalg::is_dense_tensor<vec_type>::value &&
                                    linalg::is_linalg_object<mat_type>::value,
                                void>::type
        compute_action(const vec_type &vec, res_type &r, const vtype &v,
                       const mat_type &mat)
        {
            CALL_AND_RETHROW(r = v * mat * vec);
        }

        // action of a function on a vector
        template <typename vec_type, typename res_type, typename Func,
                  typename... Args>
        typename std::enable_if<linalg::is_dense_tensor<vec_type>::value &&
                                    !linalg::is_linalg_object<Func>::value,
                                void>::type
        compute_action(const vec_type &vec, res_type &r, Func &&f, Args &&...args)
        {
            CALL_AND_RETHROW(f(vec, std::forward<Args>(args)..., r))
        }
    };

} // namespace utils

#endif // PYTTN_UTILS_ITERATIVE_LINEAR_ALGEBRA_BICGSTABL_HPP_
