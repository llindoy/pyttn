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

#ifndef PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_ENERGY_DEBUG_UPDATE_HPP_
#define PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_ENERGY_DEBUG_UPDATE_HPP_

#include <common/omp.hpp>
#include "simple_update_parameter_list.hpp"
#include "../sweeping_forward_decl.hpp"
#include "update_buffer.hpp"

namespace ttns
{

    template <typename T, typename backend>
    class energy_debug_engine<T, backend, ttn, sop_environment>
    {
    public:
        using size_type = typename backend::size_type;
        using real_type = typename tmp::get_real_type<T>::type;
        using environment_type = sop_environment<T, backend, ttn>;

        using env_container_type = typename environment_type::container_type;
        using env_node_type = typename env_container_type::node_type;
        using env_data_type = typename env_container_type::value_type;
        using env_type = typename environment_type::environment_type;

        using ttn_type = ttn<T, backend>;
        using hnode = typename ttn_type::node_type;
        using mat_type = linalg::matrix<T, backend>;
        using bond_matrix_type = typename ttn_type::bond_matrix_type;

        using buffer_type = typename environment_type::buffer_type;

        using parameter_list = simple_update_parameter_list;

    public:
        energy_debug_engine() {}
        energy_debug_engine(const ttn_type &) {}
        energy_debug_engine(const energy_debug_engine &o) = default;
        energy_debug_engine(energy_debug_engine &&o) = default;

        energy_debug_engine &operator=(const energy_debug_engine &o) = default;
        energy_debug_engine &operator=(energy_debug_engine &&o) = default;

        void initialise(const ttn_type &) {}

        void initialise(const ttn_type &, const parameter_list &) {}
        void initialise(const ttn_type &, parameter_list &&) {}

        void clear() {}

        void advance_half_step() {}

        size_type update_site_tensor(hnode &A, const environment_type &env, env_node_type &h, env_type &op)
        {
            m_res.resize(A().shape(0), A().shape(1));
            if (!A.is_leaf())
            {
                CALL_AND_HANDLE(
                    env.ceb(A(), h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2, m_res),
                    "Failed to evolve the branch coefficient matrix.");
                auto t1 = m_res.reinterpret_shape(A().as_matrix().size());
                auto a1 = A().as_matrix().reinterpret_shape(A().as_matrix().size());
                std::cerr << "branch: " << A.id() << " " << linalg::dot_product(t1, linalg::conj(a1)) << std::endl;
            }
            else
            {
                CALL_AND_HANDLE(
                    env.cel(A().as_matrix(), h, op, env.buffer().HA, env.buffer().temp, m_res),
                    "Failed to evolve the leaf coefficient matrix.");
                auto t1 = m_res.reinterpret_shape(A().as_matrix().size());
                auto a1 = A().as_matrix().reinterpret_shape(A().as_matrix().size());
                std::cerr << "leaf: " << A.id() << " " << linalg::dot_product(t1, linalg::conj(a1)) << std::endl;
            }
            return 0;
        }
        void update_bond_tensor(bond_matrix_type &r, const environment_type &env, env_node_type &h, env_type &op)
        {
            m_res = r;
            CALL_AND_HANDLE(env.fha(r, h, op, env.buffer().HA, m_res), "Failed to compute action of hamiltonian on node");
            auto t1 = m_res.reinterpret_shape(r.size());
            auto a1 = r.reinterpret_shape(r.size());
            std::cerr << "bond dim: " << linalg::dot_product(t1, linalg::conj(a1)) << std::endl;
        }

        void advance_hamiltonian(ttn_type &, environment_type &, env_container_type &, env_type &) {}

    protected:
        linalg::matrix<T, backend> m_res;
    }; // class energy_debug_engine

    template <typename T, typename backend>
    class energy_debug_engine<T, backend, ms_ttn, sop_environment>
    {
    public:
        using size_type = typename backend::size_type;
        using real_type = typename tmp::get_real_type<T>::type;
        using environment_type = sop_environment<T, backend, ms_ttn>;

        using env_container_type = typename environment_type::container_type;
        using env_node_type = typename env_container_type::node_type;
        using env_data_type = typename env_container_type::value_type;
        using env_type = typename environment_type::environment_type;

        using ttn_type = ms_ttn<T, backend>;
        using hnode = typename ttn_type::node_type;
        using mat_type = linalg::matrix<T, backend>;
        using bond_matrix_type = typename ttn_type::bond_matrix_type;

        using buffer_type = typename environment_type::buffer_type;

        using parameter_list = simple_update_parameter_list;

    public:
        energy_debug_engine() {}
        energy_debug_engine(const ttn_type &) {}
        energy_debug_engine(const energy_debug_engine &o) = default;
        energy_debug_engine(energy_debug_engine &&o) = default;

        energy_debug_engine &operator=(const energy_debug_engine &o) = default;
        energy_debug_engine &operator=(energy_debug_engine &&o) = default;

        void initialise(const ttn_type &A)
        {
            CALL_AND_RETHROW(mbuf.initialise(A));
        }

        void initialise(const ttn_type &, const parameter_list &) {}
        void initialise(const ttn_type &, parameter_list &&) {}

        void clear()
        {
            mbuf.clear();
        }

        void advance_half_step() {}

        size_type update_site_tensor(hnode &A, const environment_type &env, env_node_type &h, env_type &op)
        {
            mbuf.setup(A());

            if (!A.is_leaf())
            {
                env.ceb.set_pointer(&(A()));
                CALL_AND_HANDLE(
                    env.ceb(mbuf.A(), h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2, mbuf.res(), mbuf.resbuf()),
                    "Failed to evolve the branch coefficient matrix.");
                env.ceb.unset_pointer();
            }
            else
            {
                env.cel.set_pointer(&(A()));
                CALL_AND_HANDLE(
                    env.cel(mbuf.A(), h, op, env.buffer().HA, env.buffer().temp, mbuf.res(), mbuf.resbuf()),
                    "Failed to evolve the leaf coefficient matrix.");
                env.cel.unset_pointer();
            }

            mbuf.unpack_results();
            T val(0.0);
            for (size_t i = 0; i < A.nset(); ++i)
            {
                auto t1 = mbuf.res()[i].reinterpret_shape(A()[i].size());
                auto a1 = A()[i].as_matrix().reinterpret_shape(A()[i].size());
                val += linalg::dot_product(linalg::conj(a1), t1);
            }
            std::cout << "site: " << A.id() << " " << val << std::endl;
            return 0;
        }

        void update_bond_tensor(bond_matrix_type &r, const environment_type &env, env_node_type &h, env_type &op)
        {
            mbuf.setup(r);

            env.fha.set_pointer(&(r));
            CALL_AND_HANDLE(env.fha(mbuf.A(), h, op, env.buffer().HA, mbuf.res(), mbuf.resbuf()), "Failed to compute action of hamiltonian on node");
            env.fha.unset_pointer();

            mbuf.unpack_results();

            T val(0.0);
            for (size_t i = 0; i < r.size(); ++i)
            {
                auto t1 = mbuf.res()[i].reinterpret_shape(r[i].size());
                auto a1 = r[i].reinterpret_shape(r[i].size());
                val += linalg::dot_product(linalg::conj(a1), t1);
            }
            std::cout << "bond dim: " << val << std::endl;
        }

        void advance_hamiltonian(ttn_type &, environment_type &, env_container_type &, env_type &) {}

    public:
        multiset_update_buffer<T, backend> mbuf;
    }; // class energy_debug_engine

}

#endif // PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_ENERGY_DEBUG_UPDATE_HPP_
