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

#ifndef PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_VARIANCE_SUBSPACE_EXPANSION_ENGINE_FULL_TWO_SITE_HPP_
#define PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_VARIANCE_SUBSPACE_EXPANSION_ENGINE_FULL_TWO_SITE_HPP_

#include <random>

#include <utils/iterative_linear_algebra/arnoldi.hpp>
#include "two_site_energy_variations.hpp"
#include "subspace_expansion_full_two_site.hpp"
#include "../sweeping_forward_decl.hpp"
#include "../environment/sum_of_product_operator_env.hpp"

namespace ttns
{

    template <typename T, typename backend>
    class variance_subspace_expansion_full_two_site<T, backend, ttn, sop_environment>
    {
    public:
        using size_type = typename backend::size_type;
        using real_type = typename tmp::get_real_type<T>::type;
        using environment_type = sop_environment<T, backend, ttn>;

        using env_container_type = typename environment_type::container_type;
        using env_node_type = typename env_container_type::node_type;
        using env_type = typename environment_type::environment_type;

        using ttn_type = ttn<T, backend>;
        using hnode = typename ttn_type::node_type;

        using bond_matrix_type = typename ttn_type::bond_matrix_type;
        using population_matrix_type = typename ttn_type::population_matrix_type;

        using mat_type = linalg::matrix<T, backend>;

        using dmat_type = linalg::diagonal_matrix<real_type, backend>;

        using buffer_type = typename environment_type::buffer_type;

        using twosite = two_site_variations<T, backend>;
        using eigensolver_type = utils::arnoldi<T, backend>;

        struct parameter_list
        {
        };

    public:
        variance_subspace_expansion_full_two_site() : m_ss_expand() {}
        variance_subspace_expansion_full_two_site(const ttn_type &A, const env_type &ham) : m_ss_expand()
        {
            CALL_AND_HANDLE(initialise(A, ham), "Failed to construct variance_subspace_expansion_full_two_site.");
        }
        variance_subspace_expansion_full_two_site(const variance_subspace_expansion_full_two_site &o) = default;
        variance_subspace_expansion_full_two_site(variance_subspace_expansion_full_two_site &&o) = default;

        variance_subspace_expansion_full_two_site &operator=(const variance_subspace_expansion_full_two_site &o) = default;
        variance_subspace_expansion_full_two_site &operator=(variance_subspace_expansion_full_two_site &&o) = default;

        void initialise(const ttn_type &A, const env_type &ham)
        {
            try
            {
                CALL_AND_HANDLE(clear(), "Failed to clear the projector_splitting_intgrator.");

                size_type maxcapacity = 0;
                for (const auto &a : A)
                {
                    size_type capacity = a().capacity();
                    if (capacity > maxcapacity)
                    {
                        maxcapacity = capacity;
                    }
                }
                m_ss_expand.initialise(A, ham);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to initialise the projector_spliting_engine object.");
            }
        }

        void initialise(const ttn_type &A, const env_type &ham, const parameter_list & /* o */) { CALL_AND_RETHROW(initialise(A, ham)); }
        void initialise(const ttn_type &A, const env_type &ham, parameter_list && /* o */) { CALL_AND_RETHROW(initialise(A, ham)); }

        void clear()
        {
            try
            {
                CALL_AND_HANDLE(m_ss_expand.clear(), "Failed to clear subspace expansion object.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear the projector_spliting_engine object.");
            }
        }

        bool &eval_but_dont_apply() { return m_ss_expand.eval_but_dont_apply(); }
        const bool &eval_but_dont_apply() const { return m_ss_expand.eval_but_dont_apply(); }
        bool &only_apply_when_no_unoccupied() { return m_ss_expand.only_apply_when_no_unoccupied(); }
        const bool &only_apply_when_no_unoccupied() const { return m_ss_expand.only_apply_when_no_unoccupied(); }

        real_type &spawning_threshold() { return m_ss_expand.spawning_threshold(); }
        const real_type &spawning_threshold() const { return m_ss_expand.spawning_threshold(); }

        real_type &unoccupied_threshold() { return m_ss_expand.unoccupied_threshold(); }
        const real_type &unoccupied_threshold() const { return m_ss_expand.unoccupied_threshold(); }

        size_type &minimum_unoccupied() { return m_ss_expand.minimum_unoccupied(); }
        const size_type &minimum_unoccupied() const { return m_ss_expand.minimum_unoccupied(); }

        size_type &maximum_bond_dimension() { return m_ss_expand.maximum_bond_dimension(); }
        const size_type &maximum_bond_dimension() const { return m_ss_expand.maximum_bond_dimension(); }

        const real_type &subspace_weighting_factor() const { return m_subspace_weighting_factor; }
        real_type &subspace_weighting_factor() { return m_subspace_weighting_factor; }

        orthogonality::truncation_mode &truncation_mode() { return m_ss_expand.truncation_mode(); }
        const orthogonality::truncation_mode &truncation_mode() const { return m_ss_expand.truncation_mode(); }

        const subspace_expansion<T, backend> &subspace_expander() const { return m_ss_expand; }

    public:
        // perform the subspace expansion as we are moving down a tree.  This requires us to evaluate the optimal functions to add
        // into A2.  For A1 they will be overwriten by the r matrix in the next step so we just
        // here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
        bool subspace_expansion_down(hnode &A1, hnode &A2, bond_matrix_type &r, population_matrix_type &S, env_node_type &h, const env_type &op, environment_type &env, linalg::random_engine<backend> &rng)
        {
            try
            {
                return m_ss_expand.down(A1, A2, r, S, h, op, rng, env.buffer(), m_subspace_weighting_factor);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing down the tree.");
            }
        }

        // here the A1 and A2 tensors must be left and right orthogonal respectively with the non-orthogonal component stored in r
        bool subspace_expansion_up(hnode &A1, hnode &A2, bond_matrix_type &r, population_matrix_type &S, env_node_type &h, const env_type &op, environment_type &env, linalg::random_engine<backend> &rng)
        {
            try
            {
                return m_ss_expand.up(A1, A2, r, S, h, op, rng, env.buffer(), m_subspace_weighting_factor);
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to perform the subspace expansion when traversing up the tree.");
            }
        }

    public:
        const size_type &Nonesite() const { return m_ss_expand.Nonesite(); }
        const size_type &Ntwosite() const { return m_ss_expand.Ntwosite(); }

    protected:
        subspace_expansion_full_two_site<T, backend> m_ss_expand;

        real_type m_subspace_weighting_factor = real_type(1.0);
    }; // class variance_subspace_expansion_full_two_site
} // namespace ttns

#endif // PYTTN_TTNS_LIB_SWEEPING_ALGORITHM_SUBSPACE_EXPANSION_VARIANCE_SUBSPACE_EXPANSION_ENGINE_FULL_TWO_SITE_HPP_//
