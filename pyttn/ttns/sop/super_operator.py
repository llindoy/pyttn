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


from .opdictExt import OperatorDictionary
from .symbolicTransposeExt import symbolic_transpose
from .labelled_operator_dictionary import LabelledOperatorDictionary
from .labelled_SOP import lCSOP
from .system_information import SystemInfo

from typing import Optional

class SuperOp:
    @staticmethod
    def _validate_systems(hsys : SystemInfo, lsys : SystemInfo, suffix : str):
        """Enforces that the modes in hsys are a subset of lsys and that the suffix operators of hsys also exist in lsys

        :param hsys: The Hilbert space system definition
        :type hsys: SystemInfo
        :param lsys: The dual space system definition
        :type lsys: SystemInfo
        :param suffix: The suffix used for dual variables
        :type suffix: str

        :raises:
        """

        h_prims = set()
        l_prims = set()

        for _, prims in hsys.items():
            h_prims.update(prims.keys())
        for _, prims in lsys.items():
            l_prims.update(prims.keys())

        missing = []
        for p in h_prims:
            if p not in l_prims:
                missing.append(p)

            if f"{p}{suffix}" not in l_prims:
                missing.append(f"{p}{suffix}")

        if missing:
            raise ValueError("Liouville-space SystemInfo is missing primitive modes: "f"{sorted(missing)}")

    @staticmethod
    def left(op : lCSOP, hsys : SystemInfo, lsys : SystemInfo, opdict: Optional[LabelledOperatorDictionary] = None, suffix : str = "~") -> tuple[lCSOP, Optional[LabelledOperatorDictionary]]:
        """
        Construct the left-acting superoperator. This embeds an operator into the Liouville space defined by ``lsys`` and represents the action

            O_L = O ⊗ I

        where ``O`` acts on the physical Hilbert-space modes and the identity acts on the remaining Liouville-space degrees of freedom. 
        If an operator dictionary is supplied, a corresponding Liouville-space operator dictionary is constructed containingoperators acting on all modes appearing in ``lsys``.

        :param op: Operator acting on the Hilbert space
        :type op: lCSOP
        :param hsys: Hilbert-space system definition
        :type hsys: SystemInfo
        :param lsys: Liouville-space system definition
        :type lsys: SystemInfo
        :param opdict: Optional labelled operator dictionary
        :type opdict: Optional[LabelledOperatorDictionary]
        :param suffix: Tilde suffix used in the Liouville-space system
        :type suffix: str
        :returns: Left-acting superoperator and corresponding Liouville-space operator dictionary
        :rtype: tuple[lCSOP, Optional[LabelledOperatorDictionary]]
        """
        #validate the systems
        SuperOp._validate_systems(hsys, lsys, suffix)

        #set up the operator
        info = lsys.build_system_modes(lsys.composite_labels())
        site_map = info["primitive_label_to_index"]

        #now construct an explicit embedding of the current operator into a SOP
        sop = op.compile(site_map, nmodes=len(site_map))
        lop = lCSOP(sop, {v: k for k, v in site_map.items()})

        if opdict is None:
            return lop, None
        
        #now construct the new dictionary
        hinfo = hsys.build_system_modes(hsys.composite_labels())
        hprim_to_idx = hinfo["primitive_label_to_index"]

        hidx_to_prim = { idx: prim for prim, idx in hprim_to_idx.items()}
        lprim_to_idx = info["primitive_label_to_index"]

        # OperatorDictionary compatible with the full Liouville space
        raw_dict = OperatorDictionary(len(lprim_to_idx), dtype=opdict.dtype)

        Lopdict = LabelledOperatorDictionary(raw_dict)

        for label in opdict.labels():
            for hmode in opdict.modes(label):
                site_op = opdict[(hmode, label)]
                prim = hidx_to_prim[hmode]
                if prim not in lprim_to_idx:
                    raise ValueError(f"Primitive '{prim}' not present in Liouville-space system.")
                Lopdict.insert(lprim_to_idx[prim], label, site_op )

        return lop, Lopdict


    @staticmethod
    def right(op : lCSOP, hsys : SystemInfo, lsys : SystemInfo, opdict: Optional[LabelledOperatorDictionary] = None, suffix : str = "~") -> tuple[lCSOP, Optional[LabelledOperatorDictionary]]:
        """
        Construct the right-acting superoperator. This embeds an operator into the Liouville space defined by ``lsys`` and represents the action

            O_R = I ⊗ Oᵀ

        where ``O`` acts on the physical Hilbert-space modes and the identity acts on the remaining Liouville-space degrees of freedom.  
        If an operator dictionary is supplied, a corresponding Liouville-space operator dictionary is constructed containingoperators acting on all modes appearing in ``lsys``.

        :param op: Operator acting on the Hilbert space
        :type op: lCSOP
        :param hsys: Hilbert-space system definition
        :type hsys: SystemInfo
        :param lsys: Liouville-space system definition
        :type lsys: SystemInfo
        :param opdict: Optional labelled operator dictionary
        :type opdict: Optional[LabelledOperatorDictionary]
        :param suffix: Tilde suffix used in the Liouville-space system
        :type suffix: str
        :returns: Right-acting superoperator and corresponding Liouville-space operator dictionary
        :rtype: tuple[lCSOP, Optional[LabelledOperatorDictionary]]
        """
        #validate the systems
        SuperOp._validate_systems(hsys, lsys, suffix)

        hinfo = hsys.build_system_modes(hsys.composite_labels())
        hprim_to_idx = hinfo["primitive_label_to_index"]
        linfo = lsys.build_system_modes(lsys.composite_labels())
        lprim_to_idx = linfo["primitive_label_to_index"]

        if opdict is None:
            sop_t, _ = symbolic_transpose(op.expr, hinfo["system_modes"])
            Lopdict = None
        else:
            raw_hdict = OperatorDictionary(opdict.nmodes(), dtype=opdict.dtype)
            sop_t, raw_hdict = symbolic_transpose(op.expr, opdict._opdict, hinfo["system_modes"], raw_hdict, suffix=suffix)
            Lopdict = LabelledOperatorDictionary(OperatorDictionary(len(lprim_to_idx), dtype=opdict.dtype))

        tilde_index_to_label = {}
        for idx, label in op.index_to_label.items():
            if label not in hprim_to_idx:
                raise ValueError(f"Label '{label}' is not present in hsys.")
            tilde_label = f"{label}{suffix}"
            if tilde_label not in lprim_to_idx:
                raise ValueError(f"Primitive '{tilde_label}' is not present in lsys.")
            tilde_index_to_label[idx] = tilde_label

        op_tilde = lCSOP(sop_t, tilde_index_to_label)

        rsop = op_tilde.compile(lprim_to_idx, nmodes=len(lprim_to_idx))
        rop = lCSOP(rsop, {v: k for k, v in lprim_to_idx.items()})

        # Move generated transpose operators from Hilbert indices onto tilde indices.
        if opdict is not None:
            hidx_to_prim = { v: k for k, v in hprim_to_idx.items() }
            for label in raw_hdict.labels():
                for mode in raw_hdict.modes(label):
                    site_op = raw_hdict(mode, label)
                    prim = hidx_to_prim[mode]
                    tilde_prim = f"{prim}{suffix}"
                    Lopdict.insert(lprim_to_idx[tilde_prim], label,site_op)

        return rop, Lopdict
    
    @staticmethod
    def _merge_opdicts(left: LabelledOperatorDictionary,right: LabelledOperatorDictionary) -> LabelledOperatorDictionary:
        nmodes = max(left.nmodes(),right.nmodes())

        raw = OperatorDictionary(nmodes,dtype=left.dtype)

        merged = LabelledOperatorDictionary(raw)

        for mode in range(left.nmodes()):
            for label, op in left.site_dictionary(mode).items():
                merged.insert(mode, label, op)

        for mode in range(right.nmodes()):
            for label, op in right.site_dictionary(mode).items():
                merged.insert(mode, label, op)

        return merged

    @staticmethod
    def commutator(op : lCSOP, hsys : SystemInfo, lsys : SystemInfo, opdict: Optional[LabelledOperatorDictionary] = None, suffix : str = "~") -> tuple[lCSOP, Optional[LabelledOperatorDictionary]]:
        """
        Construct the commutator superoperator. This embeds an operator into the Liouville space defined by ``lsys`` and represents the action

            O_C = O ⊗ I - I ⊗ Oᵀ

        where ``O`` acts on the physical Hilbert-space modes and the identity acts on the remaining Liouville-space degrees of freedom.  
        If an operator dictionary is supplied, a corresponding Liouville-space operator dictionary is constructed containingoperators acting on all modes appearing in ``lsys``.

        :param op: Operator acting on the Hilbert space
        :type op: lCSOP
        :param hsys: Hilbert-space system definition
        :type hsys: SystemInfo
        :param lsys: Liouville-space system definition
        :type lsys: SystemInfo
        :param opdict: Optional labelled operator dictionary
        :type opdict: Optional[LabelledOperatorDictionary]
        :param suffix: Tilde suffix used in the Liouville-space system
        :type suffix: str
        :returns: Commutator superoperator and corresponding Liouville-space operator dictionary
        :rtype: tuple[lCSOP, Optional[LabelledOperatorDictionary]]
        """

        OL, Lopdict = SuperOp.left(op, hsys, lsys, opdict, suffix)
        OR, Ropdict = SuperOp.right(op, hsys, lsys, opdict, suffix)

        if opdict is None:
            return OL-OR, None
        else:
            return OL-OR, SuperOp._merge_opdicts(Lopdict, Ropdict)
        
    @staticmethod
    def anticommutator(op : lCSOP, hsys : SystemInfo, lsys : SystemInfo, opdict: Optional[LabelledOperatorDictionary] = None, suffix : str = "~") -> tuple[lCSOP, Optional[LabelledOperatorDictionary]]:
        """
        Construct the anticommutator superoperator. This embeds an operator into the Liouville space defined by ``lsys`` and represents the action

            O_C = O ⊗ I + I ⊗ Oᵀ

        where ``O`` acts on the physical Hilbert-space modes and the identity acts on the remaining Liouville-space degrees of freedom.  
        If an operator dictionary is supplied, a corresponding Liouville-space operator dictionary is constructed containingoperators acting on all modes appearing in ``lsys``.

        :param op: Operator acting on the Hilbert space
        :type op: lCSOP
        :param hsys: Hilbert-space system definition
        :type hsys: SystemInfo
        :param lsys: Liouville-space system definition
        :type lsys: SystemInfo
        :param opdict: Optional labelled operator dictionary
        :type opdict: Optional[LabelledOperatorDictionary]
        :param suffix: Tilde suffix used in the Liouville-space system
        :type suffix: str
        :returns: Anticommutator superoperator and corresponding Liouville-space operator dictionary
        :rtype: tuple[lCSOP, Optional[LabelledOperatorDictionary]]
        """

        OL, Lopdict = SuperOp.left(op, hsys, lsys, opdict, suffix)
        OR, Ropdict = SuperOp.right(op, hsys, lsys, opdict, suffix)

        if opdict is None:
            return OL+OR, None
        else:
            return OL+OR, SuperOp._merge_opdicts(Lopdict, Ropdict)