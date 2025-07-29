import numpy as np
import re
from pyttn.ttnpp import ntree, system_modes, boson_mode, nlevel_mode
from io import TextIOWrapper
from pyttn.ttns.sop.sSOPExt import sOP
from pyttn.ttns.sop.SOPExt import SOP

from operator import add, sub, mul, pow

_energy_unit_dict = {
    'au' : 1,
    'mh' : 1000,
    'ev' : 27.21138386,
    'mev': 27211.38386,
    'cm-1': 2.1947463137e5,
    'kcal/mol': 627.402,
    'kj/mol': 2.6255e3,
    'kelvin': 3.15777e5
}

class quantics_inputs:
    def _extract_section(fp: TextIOWrapper, section_label: str) -> list[str]:
        section = []
        match=False

        #extract the information about the tree from the quantics input file
        for line in fp:
            if re.match(section_label, line.lower()):
                match=True
            elif re.match('end-'+section_label, line.lower()):
                match=False
            elif match:
                line_val = line.strip().split('#')[0]
                if len(line_val.strip()) > 0:
                    section.append(' '.join(line_val.split()))
        return section
    
    def _convert_primitive_modes(mode_info: str) -> tuple[str, str, int]:
        """A function for converting the primitive basis mode

        :param mode_info: The 
        :type mode_info: str
        :return: _description_
        :rtype: tuple[str, str, int]
        """
        minf = mode_info.split(' ')
        label = minf[0]
        dims = int(minf[2])
        type = None
        if minf[1].lower() == 'ho':
            if float(minf[3]) != 0 or float(minf[4]) != 1.0 or float(minf[5]) != 1.0:
                raise RuntimeError("Currently the quantics converter only works with zero displacement frequency 1 harmonic oscillators")
            type = 'boson'
        elif minf[1].lower() == 'el':
            type = 'nlevel'
        else:
            raise RuntimeError("Currently the quantics converter only supports the harmonic oscillator basis set.")
        return label, type, dims
    
    def _get_mode_ordering(tree_info: list[str]) -> tuple[list[list[str]], list[str]]:
        modes = []
        mode_combination = []
        for line in tree_info:
            vals = line.split('>')[1].strip()
            if '[' in vals and ']' in vals:
                comb_modes = vals.split('[')[1].split(']')[0].split(' ')
                mode_combination.append(list(comb_modes))
                for v in comb_modes:
                    modes.append(v)
        return mode_combination, modes

    def _convert_tree_info(tree_info: list[str]) -> ntree:
        #now extract the tree information and the mode combination rules
        topo = ntree('1')
        counter = ntree('0')

        curr_node = topo.root()
        counter_node = counter.root()
        curr_level = None

        for line in tree_info:
            level = int(line.split('>')[0])
            vals = line.split('>')[1].strip()
            if '[' in vals and ']' in vals:
                counter_node.value += 1
            else:
                if curr_level is None:
                    for val in vals.split(' '):
                        curr_node.insert(int(val))
                        counter_node.insert(0)
                    curr_level = level

                elif curr_level < level:
                    curr_node = curr_node[counter_node.value]
                    counter_node = counter_node[counter_node.value]
                    for val in vals.split(' '):
                        curr_node.insert(int(val))
                        counter_node.insert(0)
                    curr_level = level


                elif curr_level == level:
                    curr_node = curr_node.parent()
                    counter_node = counter_node.parent()

                    counter_node.value += 1

                    curr_node = curr_node[counter_node.value]
                    counter_node = counter_node[counter_node.value]
                
                    for val in vals.split(' '):
                        curr_node.insert(int(val))
                        counter_node.insert(0)

                    curr_level = level
                elif curr_level > level:
                    for _ in range(level, curr_level+1):
                        curr_node = curr_node.parent()
                        counter_node = counter_node.parent()

                    counter_node.value += 1

                    curr_node = curr_node[counter_node.value]
                    counter_node = counter_node[counter_node.value]
                
                    for val in vals.split(' '):
                        curr_node.insert(int(val))
                        counter_node.insert(0)
                    curr_level = level

        return topo 
    
    def load_topology(fname: str) -> tuple[ntree, system_modes, list[str]]:
        """_summary_

        :param fname: Path to input quantics file
        :type fname: str
        :return: _description_
        :rtype: tuple[ntree, dict]
        """
        with open(fname, 'r') as fp:
            tree_info = quantics_inputs._extract_section(fp, 'ml-basis-section')
        with open(fname, 'r') as fp:
            mode_info = quantics_inputs._extract_section(fp, 'primitive-basis-section')

        #extract the mode information from the mode_info strings
        mode_dict = {}
        for mode_str in mode_info:
            label, t, d = quantics_inputs._convert_primitive_modes(mode_str)
            mode_dict[label] = {'type': t, 'lhd': d}

        #get the mode combination and mode ordering information
        mode_combination, modes = quantics_inputs._get_mode_ordering(tree_info)

        #extract the base tree structure
        topo = quantics_inputs._convert_tree_info(tree_info)
        #now iterate over the tree nodes and add the primitive nodes to the leaves of the tree
        leaves = topo.leaf_indices()
        for counter, leaf in enumerate(leaves):
            dim = 1
            for imode in mode_combination[counter]:
                dim = dim * mode_dict[imode]['lhd']
            topo.at(leaf).insert(dim)

        #finally set up the system modes information
        sysinf = system_modes(len(mode_combination))
        for i, mc in enumerate(mode_combination):
            combined_mode = []
            for ml in mc:
                if mode_dict[ml]['type'] == 'boson':
                    combined_mode.append(boson_mode(mode_dict[ml]['lhd']))
                elif mode_dict[ml]['type'] == 'nlevel':
                    combined_mode.append(nlevel_mode(mode_dict[ml]['lhd']))
                else:
                    raise RuntimeError("Invalid mode type.")
            sysinf[i] = combined_mode
        return topo, sysinf, modes
    
    def _extract_parameter_dict(parameter_info : list[str]) -> dict:
        params = {}
        for line in parameter_info:
            label = line.split("=")[0].strip()
            expression = line.split("=")[1].strip()
            if "," in expression:
                numeric = expression.split(",")[0].strip()
                unit = expression.split(",")[1].strip()
                val = float(numeric)*_energy_unit_dict[unit]
            else:
                val = float(expression.strip())
            params[label]=val
        return params
    
    def _extract_mode_order(hamiltonian_info : list[str], modes: list[str]) -> list[int]:
        hamiltonian_modes = []
        for line in hamiltonian_info:
            if "modes" in line:
                for x in line.strip().split("|")[1:]:
                    v = x.strip()
                    if len(v) > 0:
                        hamiltonian_modes.append(v)
        hamiltonian_to_tree_mapping = []
        for label in hamiltonian_modes:
            hamiltonian_to_tree_mapping.append(modes.index(label))
        return hamiltonian_to_tree_mapping
    
    def _extract_coeff(coeff: str, params: dict) -> float:
        res_str=""
        mul_split = coeff.split("*")
        for i, mul_str in enumerate(mul_split):
            div_split = mul_str.split("/")
            for j, div_str in enumerate(div_split):
                add_split = div_str.split("+")
                for k, add_str in enumerate(add_split):
                    sub_split = add_str.split("-")
                    for ll, sub_str in enumerate(sub_split):
                        pow_split = sub_str.split("^")
                        for m, pow_str in enumerate(pow_split):
                            if pow_str in params:
                                res_str += str(params[pow_str])
                            else:
                                res_str += pow_str
                            if m+1 < len(pow_split):
                                res_str += "^"
                        if ll+1 < len(sub_split):
                            res_str += "-"
                    if k+1 < len(add_split):
                        res_str += "+"
                if j + 1 < len(div_split):
                    res_str += "/"
            if i + 1 < len(mul_split):
                res_str += "*"
        return eval(res_str)


    def _split_hamiltonian_info(hamiltonian_info: list[str]) -> tuple[list[str], list[str]]:
        mode_info = []
        h_info = []
        mode_section = False
        for line in hamiltonian_info:
            if "---" in line:
                mode_section = not mode_section
            else:
                if mode_section:
                    mode_info.append(line)
                else:
                    h_info.append(line)

        return mode_info, h_info
    
    def _extract_mode_operator(term : str) -> tuple[str, int]:
        #to do - add conversion from quantics format operator labels to pyttn format operator labels
        label = term.split(' ')[1].strip()
        hmode = int(term.split(' ')[0].strip())
        return label, hmode

    def _extract_operator_definition(hamiltonian_info: list[str], params: dict, mode_order: list[int]) -> SOP:
        H = SOP(len(mode_order))
        for line in hamiltonian_info:
            split_line = line.split('|')
            if len(split_line) == 0:
                continue
            coeff = split_line[0].strip()        
            terms = [x.strip() for x in split_line[1:]]
            val = quantics_inputs._extract_coeff(coeff, params)
            
            label, mode = quantics_inputs._extract_mode_operator(terms[0])
            op = sOP(label, mode_order[mode-1])
            if len(terms) > 1:
                for term in terms[1:]:
                    label, mode = quantics_inputs._extract_mode_operator(term)
                    op *= sOP(label, mode_order[mode-1])
            print(op)
            H += op

    def load_operator(fname: str, modes: list[str]) -> SOP:
        with open(fname, 'r') as fp:
            parameter_info = quantics_inputs._extract_section(fp, 'parameter-section')
        with open(fname, 'r') as fp:
            hamiltonian_info = quantics_inputs._extract_section(fp, 'hamiltonian-section')

        mode_info, hamiltonian_info = quantics_inputs._split_hamiltonian_info(hamiltonian_info)
        params = quantics_inputs._extract_parameter_dict(parameter_info)
        mode_order = quantics_inputs._extract_mode_order(mode_info, modes)

        quantics_inputs._extract_operator_definition(hamiltonian_info, params, mode_order)


if __name__ == "__main__":
    topo, sysinf, modes = quantics_inputs.load_topology("QuanticsTest/th21d_s1.inp")
    quantics_inputs.load_operator("QuanticsTest/thio_opt5.op", modes)
