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

#ifndef PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_GSO_ENGINE_HPP_
#define PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_GSO_ENGINE_HPP_

#include <common/omp.hpp>

#include <utils/iterative_linear_algebra/arnoldi.hpp>
#include "simple_update_parameter_list.hpp"

namespace ttns
{

    template <typename T, typename backend>
    class gso_engine<T, backend, ttn, sop_environment>
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

        using matnode = typename tree<mat_type>::node_type;

        using buffer_type = typename environment_type::buffer_type;
        using eigensolver_type = utils::arnoldi<T, backend>;

        using parameter_list = simple_update_parameter_list;

    public:
        gso_engine() : m_curr_E(0) {}
        gso_engine(const ttn_type &A, size_type krylov_dim = 4) : m_curr_E(0)
        {
            CALL_AND_HANDLE(initialise(A, krylov_dim), "Failed to construct gso_engine.");
        }
        gso_engine(const gso_engine &o) = default;
        gso_engine(gso_engine &&o) = default;

        gso_engine &operator=(const gso_engine &o) = default;
        gso_engine &operator=(gso_engine &&o) = default;

        void initialise(const ttn_type &A, size_type krylov_dim = 4)
        {
            try
            {
                CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

                size_type maxcapacity = 0;
                for (const auto &a : A)
                {
                    size_type capacity = a.buffer_maxcapacity();
                    if (capacity > maxcapacity)
                    {
                        maxcapacity = capacity;
                    }
                }
                CALL_AND_HANDLE(m_eigensolver.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");

                m_eigensolver.mode() = utils::eigenvalue_target::smallest_real;
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
            }
        }

        void initialise(const ttn_type &A, const parameter_list &o) { CALL_AND_RETHROW(initialise(A, o.krylov_dim)); }
        void initialise(const ttn_type &A, parameter_list &&o) { CALL_AND_RETHROW(initialise(A, o.krylov_dim)); }

        void clear()
        {
            try
            {
                CALL_AND_HANDLE(m_eigensolver.clear(), "Failed to clear the krylov subspace engine.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
            }
        }

        void advance_half_step() {}

        const eigensolver_type &eigensolver() const { return m_eigensolver; }
        eigensolver_type &eigensolver() { return m_eigensolver; }

        real_type &eigensolver_tol() { return m_eigensolver.error_tolerance(); }
        const real_type &eigensolver_tol() const { return m_eigensolver.error_tolerance(); }

        real_type &eigensolver_reltol() { return m_eigensolver.rel_tol(); }
        const real_type &eigensolver_reltol() const { return m_eigensolver.rel_tol(); }

        utils::eigenvalue_target &mode() { return m_eigensolver.mode(); }
        const utils::eigenvalue_target &mode() const { return m_eigensolver.mode(); }

        const size_type &restarts() const { return m_eigensolver.max_iter(); }
        size_type &restarts() { return m_eigensolver.max_iter(); }

        T E() const { return m_curr_E; }

        size_type update_site_tensor(hnode &A, const environment_type &env, env_node_type &h, env_type &op)
        {
            size_type ret = 0;
            // update the node tensor.  To do this we solve the effective eigenproblem
            if (!A.is_leaf())
            {
                env.ceb.set_pointer(&(A()));
                CALL_AND_HANDLE(
                    ret = m_eigensolver(A().as_matrix(), m_curr_E, env.ceb, h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2),
                    "Failed to evolve the branch coefficient matrix.");
                env.ceb.unset_pointer();
            }
            else
            {
                CALL_AND_HANDLE(
                    ret = m_eigensolver(A().as_matrix(), m_curr_E, env.cel, h, op, env.buffer().HA, env.buffer().temp),
                    "Failed to evolve the leaf coefficient matrix.");
            }
            return ret;
        }

        void update_bond_tensor(bond_matrix_type & /* r */, const environment_type & /* env */, env_node_type & /* h */, env_type & /* op */) {}
        void advance_hamiltonian(ttn_type &, environment_type &, env_container_type &, env_type &) {}

    protected:
        // the krylov subspace engine
        eigensolver_type m_eigensolver;

        T m_curr_E;
    }; // class gso_engine

    template <typename T, typename backend>
    class gso_engine<T, backend, ms_ttn, sop_environment>
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
        using hdata = typename hnode::value_type;

        using mat_type = linalg::matrix<T, backend>;
        using bond_matrix_type = typename ttn_type::bond_matrix_type;

        using matnode = typename tree<mat_type>::node_type;

        using buffer_type = typename environment_type::buffer_type;
        using eigensolver_type = utils::arnoldi<T, backend>;

        using parameter_list = simple_update_parameter_list;

    public:
        gso_engine() : m_curr_E(0) {}
        gso_engine(const ttn_type &A, size_type krylov_dim = 4) : m_curr_E(0)
        {
            CALL_AND_HANDLE(initialise(A, krylov_dim), "Failed to construct gso_engine.");
        }
        gso_engine(const gso_engine &o) = default;
        gso_engine(gso_engine &&o) = default;

        gso_engine &operator=(const gso_engine &o) = default;
        gso_engine &operator=(gso_engine &&o) = default;

        void initialise(const ttn_type &A, size_type krylov_dim = 4)
        {
            try
            {
                CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

                size_type maxcapacity = 0;
                for (const auto &a : A)
                {
                    size_type capacity = a.buffer_maxcapacity();
                    if (capacity > maxcapacity)
                    {
                        maxcapacity = capacity;
                    }
                }
                CALL_AND_HANDLE(m_eigensolver.resize(krylov_dim, maxcapacity), "Failed to initialise the krylov subspace engine object.");

                m_eigensolver.mode() = utils::eigenvalue_target::smallest_real;
                CALL_AND_RETHROW(mbuf.initialise(A));
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
            }
        }

        void initialise(const ttn_type &A, const parameter_list &o) { CALL_AND_RETHROW(initialise(A, o.krylov_dim)); }
        void initialise(const ttn_type &A, parameter_list &&o) { CALL_AND_RETHROW(initialise(A, o.krylov_dim)); }

        void clear()
        {
            try
            {
                mbuf.clear();
                CALL_AND_HANDLE(m_eigensolver.clear(), "Failed to clear the krylov subspace engine.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
            }
        }

        void advance_half_step() {}

        const eigensolver_type &eigensolver() const { return m_eigensolver; }
        eigensolver_type &eigensolver() { return m_eigensolver; }

        real_type &eigensolver_tol() { return m_eigensolver.error_tolerance(); }
        const real_type &eigensolver_tol() const { return m_eigensolver.error_tolerance(); }

        real_type &eigensolver_reltol() { return m_eigensolver.rel_tol(); }
        const real_type &eigensolver_reltol() const { return m_eigensolver.rel_tol(); }

        utils::eigenvalue_target &mode() { return m_eigensolver.mode(); }
        const utils::eigenvalue_target &mode() const { return m_eigensolver.mode(); }

        const size_type &restarts() const { return m_eigensolver.max_iter(); }
        size_type &restarts() { return m_eigensolver.max_iter(); }

        T E() const { return m_curr_E; }

        size_type update_site_tensor(hnode &A, const environment_type &env, env_node_type &h, env_type &op)
        {
            size_type ret = 0;

            mbuf.setup(A());
            // update the node tensor.  To do this we solve the effective eigenproblem
            if (!A.is_leaf())
            {
                env.ceb.set_pointer(&(A()));
                CALL_AND_HANDLE(
                    ret = m_eigensolver(mbuf.A(), m_curr_E, env.ceb, h, op, env.buffer().HA, env.buffer().temp, env.buffer().temp2, mbuf.res()),
                    "Failed to evolve the branch coefficient matrix.");
                env.ceb.unset_pointer();
            }
            else
            {
                env.cel.set_pointer(&(A()));
                CALL_AND_HANDLE(
                    ret = m_eigensolver(mbuf.A(), m_curr_E, env.cel, h, op, env.buffer().HA, env.buffer().temp, mbuf.res()),
                    "Failed to evolve the leaf coefficient matrix.");
                env.cel.unset_pointer();
            }
            ttn_type::unpack(mbuf.A(), A());
            return ret;
        }

        void update_bond_tensor(bond_matrix_type & /* r */, const environment_type & /* env */, env_node_type & /* h */, env_type & /* op */) {}

        void advance_hamiltonian(ttn_type &, environment_type &, env_container_type &, env_type &) {}

    protected:
        // the krylov subspace engine
        eigensolver_type m_eigensolver;

        T m_curr_E;
        multiset_update_buffer<T, backend> mbuf;
    }; // class gso_engine

}

#endif // PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_UPDATE_GSO_ENGINE_HPP_
