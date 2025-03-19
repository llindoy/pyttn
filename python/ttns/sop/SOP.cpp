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

#include "SOP.hpp"
#include "../../pyttn_typedef.hpp"

template <>
void initialise_SOP<pyttn_real_type>(py::module &m)
{
    using real_type = pyttn_real_type;
    using complex_type = linalg::complex<real_type>;
#ifdef BUILD_REAL_TTN
    init_SOP<real_type>(m, "SOP_real");
#endif
    init_SOP<complex_type>(m, "SOP_complex");
}
