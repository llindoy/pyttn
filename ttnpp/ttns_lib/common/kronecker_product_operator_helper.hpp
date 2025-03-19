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

#ifndef PYTTN_TTNS_LIB_COMMON_KRONECKER_PRODUCT_OPERATOR_HELPER_HPP_
#define PYTTN_TTNS_LIB_COMMON_KRONECKER_PRODUCT_OPERATOR_HELPER_HPP_

#include "../ttn/ttn.hpp"
#include "../operators/operator_node.hpp"

#include <linalg/linalg.hpp>

namespace ttns
{

    template <typename T, typename backend>
    class kronecker_product_operator
    {
    private:
        using hdata = ttn_node_data<T, backend>;
        using mat = linalg::matrix<T, backend>;
        using matnode = typename tree<mat>::node_type;

        using optype = operator_node_data<T, backend>;
        using opnode = typename tree<optype>::node_type;

        using boolnode = typename tree<bool>::node_type;
        using size_type = typename backend::size_type;

    public:
        // kronecker product operators for matrix types
        static void apply(const matnode &op, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                for (size_type nu = 0; nu < op.size(); ++nu)
                {
                    auto _A = A.as_rank_3(nu);
                    auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                    auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                    if (first_call)
                    {
                        CALL_AND_HANDLE(_res = contract(op[nu](), 1, _A, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                        first_call = false;
                    }
                    else if (res_set)
                    {
                        CALL_AND_HANDLE(_temp = contract(op[nu](), 1, _res, 1), "Failed to compute kronecker product contraction.");
                        res_set = false;
                    }
                    else
                    {
                        CALL_AND_HANDLE(_res = contract(op[nu](), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                    }
                }
                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }
        static void apply(const matnode &op, size_type nuskip, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                for (size_type nu = 0; nu < op.size(); ++nu)
                {
                    if (nu != nuskip)
                    {
                        auto _A = A.as_rank_3(nu);
                        auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                        auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                        if (first_call)
                        {
                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _A, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                            first_call = false;
                        }
                        else if (res_set)
                        {
                            CALL_AND_HANDLE(_temp = contract(op[nu](), 1, _res, 1), "Failed to compute kronecker product contraction.");
                            res_set = false;
                        }
                        else
                        {
                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                        }
                    }
                }
                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

        // kronecker product operators for matrix types
        static void apply(const matnode &op, const boolnode &is_id, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                for (size_type nu = 0; nu < op.size(); ++nu)
                {
                    if (!is_id[nu]())
                    {
                        auto _A = A.as_rank_3(nu);
                        auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                        auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                        if (first_call)
                        {
                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _A, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                            first_call = false;
                        }
                        else if (res_set)
                        {
                            CALL_AND_HANDLE(_temp = contract(op[nu](), 1, _res, 1), "Failed to compute kronecker product contraction.");
                            res_set = false;
                        }
                        else
                        {
                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                        }
                    }
                }
                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

        static void apply(const matnode &op, const boolnode &is_id, size_type nuskip, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                for (size_type nu = 0; nu < op.size(); ++nu)
                {
                    if (nu != nuskip && !is_id[nu]())
                    {
                        auto _A = A.as_rank_3(nu);
                        auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                        auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                        if (first_call)
                        {
                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _A, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                            first_call = false;
                        }
                        else if (res_set)
                        {
                            CALL_AND_HANDLE(_temp = contract(op[nu](), 1, _res, 1), "Failed to compute kronecker product contraction.");
                            res_set = false;
                        }
                        else
                        {
                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                        }
                    }
                }
                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

    public:
        // kronecker product operators for the operator ype
        static void apply(const opnode &op, size_type ind, size_type ri, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                const auto &spinds = op()[ind].spf_indexing()[ri];
                for (size_type ni = 0; ni < spinds.size(); ++ni)
                {
                    size_type nu = spinds[ni][0];
                    size_type cri = spinds[ni][1];

                    auto _A = A.as_rank_3(nu);
                    auto _res = res.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                    auto _temp = temp.reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                    if (first_call)
                    {
                        CALL_AND_HANDLE(_res = contract(op[nu]()[cri].spf(), 1, _A, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                        first_call = false;
                    }
                    else if (res_set)
                    {
                        CALL_AND_HANDLE(_temp = contract(op[nu]()[cri].spf(), 1, _res, 1), "Failed to compute kronecker product contraction.");
                        res_set = false;
                    }
                    else
                    {
                        CALL_AND_HANDLE(_res = contract(op[nu]()[cri].spf(), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                    }
                }
                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

        // kronecker product operators for the operator ype
        static void apply(const opnode &op, size_type ind, size_type ri, const hdata &B, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                size_type Mshape;
                size_type Rshape;
                ASSERT(op().has_id_matrices(), "Cannot apply rectangular hamiltonian without having identity matrices bound");

                const auto &spinds = op()[ind].spf_indexing()[ri];

                size_t ni = 0;
                for (size_type _nu = 0; _nu < A.nmodes(); ++_nu)
                {
                    size_type nu = spinds[ni][0];
                    size_type m = B.dim(_nu);

                    auto _A = A.as_rank_3(_nu);

                    if (first_call)
                    {
                        Mshape = _A.shape(0);
                        Rshape = m;
                    }
                    else
                    {
                        Mshape *= Rshape;
                        Rshape = m;
                    }

                    if (_nu == nu)
                    {
                        size_type cri = spinds[ni][1];
                        if (first_call)
                        {
                            CALL_AND_HANDLE(res.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                            auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                            CALL_AND_HANDLE(_res = contract(op[nu]()[cri].spf(), 1, _A, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                            first_call = false;
                        }
                        else
                        {
                            if (res_set)
                            {
                                CALL_AND_HANDLE(temp.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                                auto _res = res.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                                auto _temp = temp.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                                CALL_AND_HANDLE(_temp = contract(op[nu]()[cri].spf(), 1, _res, 1), "Failed to compute kronecker product contraction.");
                                res_set = false;
                            }
                            else
                            {
                                CALL_AND_HANDLE(res.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                                auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
                                auto _temp = temp.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));

                                CALL_AND_HANDLE(_res = contract(op[nu]()[cri].spf(), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                                res_set = true;
                            }
                        }
                        ++ni;
                    }
                    else
                    {
                        size_type cri = spinds[ni][1];
                        if (first_call)
                        {
                            CALL_AND_HANDLE(res.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                            auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                            CALL_AND_HANDLE(_res = contract(op[nu]().id_spf(), 1, _A, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                            first_call = false;
                        }
                        else
                        {
                            if (res_set)
                            {
                                CALL_AND_HANDLE(temp.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                                auto _res = res.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                                auto _temp = temp.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                                CALL_AND_HANDLE(_temp = contract(op[nu]().id_spf(), 1, _res, 1), "Failed to compute kronecker product contraction.");
                                res_set = false;
                            }
                            else
                            {
                                CALL_AND_HANDLE(res.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                                auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
                                auto _temp = temp.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));

                                CALL_AND_HANDLE(_res = contract(op[nu]().id_spf(), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                                res_set = true;
                            }
                        }
                    }
                }

                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

        // kronecker product operators for the operator ype
        template <typename Atype, typename dimstype>
        static void apply(const opnode &op, size_type ind, size_type ri, size_type hrank, const dimstype &dims, const Atype &A, mat &temp, mat &res)
        {
            try
            {
                size_type ndim = 1;
                for (size_type i = 0; i < dims.size(); ++i)
                {
                    ndim *= dims[i];
                }

                std::array<size_type, 3> r3dims = {{1, 1, hrank * ndim}};
                bool first_call = true;
                bool res_set = true;

                int64_t nuprev = 0;
                const auto &spinds = op()[ind].spf_indexing()[ri];
                for (size_type ni = 0; ni < spinds.size(); ++ni)
                {
                    size_type nu = spinds[ni][0];
                    size_type cri = spinds[ni][1];

                    for (size_type nuint = static_cast<size_type>(nuprev); nuint <= nu; ++nuint)
                    {
                        r3dims[2] /= dims[nuint];
                        if (nuint != nu)
                        {
                            r3dims[0] *= dims[nuint];
                        }
                    }
                    r3dims[1] = dims[nu];
                    nuprev = nu + 1;

                    auto _A = A.reinterpret_shape(r3dims[0], r3dims[1], r3dims[2]);
                    auto _res = res.reinterpret_shape(r3dims[0], r3dims[1], r3dims[2]);
                    auto _temp = temp.reinterpret_shape(r3dims[0], r3dims[1], r3dims[2]);

                    if (first_call)
                    {
                        CALL_AND_HANDLE(_res = contract(op[nu]()[cri].spf(), 1, _A, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                        first_call = false;
                    }
                    else if (res_set)
                    {
                        CALL_AND_HANDLE(_temp = contract(op[nu]()[cri].spf(), 1, _res, 1), "Failed to compute kronecker product contraction.");
                        res_set = false;
                    }
                    else
                    {
                        CALL_AND_HANDLE(_res = contract(op[nu]()[cri].spf(), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                    }
                    r3dims[0] *= dims[nu];
                }
                if (first_call)
                {
                    res_set = true;
                    res = A;
                }
                if (!res_set)
                {
                    res.swap_buffer(temp);
                }
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

    public:
        // kronecker product operators for matrix types -  currently unsure if this works need to check it in the future.
        static void apply_rectangular(const matnode &op, const hdata &A, mat &temp, mat &res)
        {
            try
            {
                bool first_call = true;
                bool res_set = true;

                size_type Mshape;
                size_type Rshape;

                for (size_type nu = 0; nu < op.size(); ++nu)
                {
                    size_type m = op[nu]().shape(0);
                    auto _A = A.as_rank_3(nu);
                    if (first_call)
                    {
                        Mshape = _A.shape(0);
                        Rshape = m;

                        CALL_AND_HANDLE(res.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                        auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                        CALL_AND_HANDLE(_res = contract(op[nu](), 1, _A, 1), "Failed to compute kronecker product contraction.");
                        res_set = true;
                        first_call = false;
                    }
                    else
                    {
                        Mshape *= Rshape;
                        Rshape = m;
                        if (res_set)
                        {
                            CALL_AND_HANDLE(temp.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                            auto _res = res.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));
                            auto _temp = temp.reinterpret_shape(Mshape, Rshape, _A.shape(2));

                            CALL_AND_HANDLE(_temp = contract(op[nu](), 1, _res, 1), "Failed to compute kronecker product contraction.");
                            res_set = false;
                        }
                        else
                        {
                            CALL_AND_HANDLE(res.resize(1, Mshape * Rshape * _A.shape(2)), "Failed to resize temporary working buffer.");
                            auto _res = res.reinterpret_shape(Mshape, Rshape, _A.shape(2));
                            auto _temp = temp.reinterpret_shape(Mshape, _A.shape(1), _A.shape(2));

                            CALL_AND_HANDLE(_res = contract(op[nu](), 1, _temp, 1), "Failed to compute kronecker product contraction.");
                            res_set = true;
                        }
                    }
                }
                if (first_call)
                {
                    res_set = true;
                    res = A.as_matrix();
                }
                if (!res_set)
                {
                    res = temp;
                }
            }
            catch (const common::invalid_value &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_NUMERIC("applying kronecker product operator.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to apply kronecker product operator.");
            }
        }

    }; // struct kronecker_product_operator

} // namespace ttns

#endif // PYTTN_TTNS_LIB_COMMON_KRONECKER_PRODUCT_OPERATOR_HELPER_HPP_//
