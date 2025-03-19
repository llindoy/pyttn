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

#ifndef PYTTN_TTNS_LIB_SOP_MODELS_MODEL_HPP_
#define PYTTN_TTNS_LIB_SOP_MODELS_MODEL_HPP_

#include "../sSOP.hpp"
#include "../SOP.hpp"
#include "../system_information.hpp"

namespace ttns
{

    // a interface class for handling generic models.  The model classes are designed for offloading
    // the work associated with setting up the Hamiltonian and its parameters from the user.
    template <typename T>
    class model
    {
    public:
        using real_type = typename linalg::get_real_type<T>::type;

    public:
        model() {}
        virtual ~model() {}

        virtual SOP<T> hamiltonian(real_type tol = 1e-14)
        {
            SOP<T> sop;
            this->hamiltonian(sop, tol);
            return sop;
        }
        virtual system_modes system_info()
        {
            system_modes inf;
            this->system_info(inf);
            return inf;
        }

        virtual void hamiltonian(sSOP<T> &H, real_type tol = 1e-14) = 0;
        virtual void hamiltonian(SOP<T> &H, real_type tol = 1e-14) = 0;
        virtual void system_info(system_modes &sysinf) = 0;
    };
}

#endif // PYTTN_TTNS_LIB_SOP_MODELS_MODEL_HPP_
