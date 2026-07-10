

# This files is part of the pyTTN package.
# (C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from ..sop.system_information import SystemInfo
from ..sop.labelled_SOP import lCSOP

def interaction_terms(op: lCSOP,sysinfo: SystemInfo):
    prim_to_comp = {}

    for comp_label, prims in sysinfo.items():
        for prim_label in prims:
            prim_to_comp[prim_label] = comp_label

    opdict = op.expr.get_operator_dictionary()

    for term, coeff in op.expr:

        pop = term.as_sPOP(opdict)

        comps_in_term = set()

        for opi in pop:
            label = op.index_to_label[opi.mode]

            if label not in prim_to_comp:
                raise ValueError(f"Primitive mode '{label}' not found")

            comps_in_term.add(prim_to_comp[label])

        if len(comps_in_term) == 0:
            continue

        yield ( comps_in_term, abs(coeff(0)) ** 2, coeff, term, )
