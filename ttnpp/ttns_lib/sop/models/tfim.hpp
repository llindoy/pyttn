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

#ifndef PYTTN_TTNS_LIB_SOP_MODELS_TFIM_HPP_
#define PYTTN_TTNS_LIB_SOP_MODELS_TFIM_HPP_

#include "model.hpp"
#include "../SOP.hpp"

namespace ttns
{

    // a class for handling the generation of the second quantised TFIM hamiltonian object.
    template <typename value_type>
    class TFIM : public model<value_type>
    {
    public:
        using real_type = typename linalg::get_real_type<value_type>::type;
        using model<value_type>::hamiltonian;
        using model<value_type>::system_info;

    public:
        TFIM() {}
        TFIM(size_t N, real_type _t, real_type _J, bool open_boundary_condition = false) : m_N(N), m_t(_t), m_J(_J), m_open_boundary_conditions(open_boundary_condition) {}
        virtual ~TFIM() {}

        // functions for building the different sop representations of the Hamiltonian
        virtual void hamiltonian(sSOP<value_type> &H, real_type tol = 1e-14) final
        {
            H.clear();
            H.reserve(2 * m_N - (m_open_boundary_conditions ? 0 : 1));
            build_sop_repr(H, tol);
        }

        virtual void hamiltonian(SOP<value_type> &H, real_type tol = 1e-14) final
        {
            H.clear();
            H.resize(m_N);
            build_sop_repr(H, tol);
        }

        // functions for accessing the size of the model
        size_t N() const { return m_N; }
        size_t &N() { return m_N; }

        // functions for accessing the onsite energy term
        const real_type &t() const { return m_t; }
        real_type &t() { return m_t; }

        // functions for accessing the interaction term
        const real_type &J() const { return m_J; }
        real_type &J() { return m_J; }

        // functions for accessing whether or not to use open boundary conditions
        const bool &open_boundary_conditions() const { return m_open_boundary_conditions; }
        bool &open_boundary_conditions() { return m_open_boundary_conditions; }

        virtual void system_info(system_modes &sysinf) final
        {
            sysinf.resize(m_N);
            for (size_t i = 0; i < m_N; ++i)
            {
                sysinf[i] = spin_mode(2);
            }
        }

    protected:
        template <typename Hop>
        void build_sop_repr(Hop &H, real_type tol)
        {
            if (linalg::abs(m_t) > tol)
            {
                // add on the onsite terms
                for (size_t i = 0; i < m_N; ++i)
                {
                    H += m_t * sOP("sx", i);
                }
            }

            if (linalg::abs(m_J) > tol)
            {
                // add on the onsite terms
                for (size_t i = 1; i < m_N; ++i)
                {
                    H += m_J * sOP("sz", i - 1) * sOP("sz", i);
                }
                if (m_open_boundary_conditions)
                {
                    H += m_J * sOP("sz", 0) * sOP("sz", m_N - 1);
                }
            }
        }

    protected:
        size_t m_N;
        real_type m_t, m_J;
        bool m_open_boundary_conditions;
    };
}

#endif // PYTTN_TTNS_LIB_SOP_MODELS_TFIM_HPP_
