import re
import numpy as np
from io import TextIOWrapper
import functools as ft

from pyttn.ttnpp import ntree, system_modes, boson_mode, nlevel_mode
from pyttn.ttns.sop.sSOPExt import sOP
from pyttn.ttns.sop.SOPExt import SOP
from pyttn.ttns.operators.siteOperatorsExt import site_operator

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

class QuanticsInputs:
    def _extract_section(fp: TextIOWrapper, section_label: str) -> list[str]:
        section = []
        match=False

        #extract the information about the tree from the quantics input file
        for line in fp:
            if re.match(section_label, "".join(line.split("-")).lower()):
                match=True
            elif re.match('end'+section_label, "".join(line.split("-")).lower()):
                match=False
            elif match:
                line_val = line.strip().split('#')[0]
                line_val = line_val.strip()
                if len(line_val) > 0:
                    #handle continuation lines
                    if(line_val.startswith("&&&")):
                        line = "".join(line_val.split("&&&")[1:])
                        section[-1] += line
                    else:
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
    

    
    def _extract_parameter_dict(parameter_info : list[str]) -> dict:
        params = {}
        for line in parameter_info:
            label = line.split("=")[0].strip()
            expression = line.split("=")[1].strip()
            if "," in expression:
                numeric = expression.split(",")[0].strip()
                unit = expression.split(",")[1].strip()
                val = float(numeric)/_energy_unit_dict[unit]
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
            if label in modes:
                hamiltonian_to_tree_mapping.append(modes.index(label))
            else:
                hamiltonian_to_tree_mapping.append(-1)
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
    
    def _extract_mode_operator(term : str) -> tuple[list[str], int]:
        #to do - add conversion from quantics format operator labels to pyttn format operator labels
        label = term.split(' ')[1].strip()
        hmode = int(term.split(' ')[0].strip())

        x = re.split('Z(\\d+)&(\\d+)', label)
        if len(x) == 4:
            label = '|%d><%d|'%(int(x[1])-1, int(x[2])-1)
            return [label.lower()], hmode
        x = re.split('S(\\d+)&(\\d+)', label)
        if len(x) == 4:
            if int(x[1]) != int(x[2]):
                label1 = '|%d><%d|'%(int(x[1])-1, int(x[2])-1)
                label2 = '|%d><%d|'%(int(x[2])-1, int(x[1])-1)

                return [label1.lower(), label2.lower()], hmode
            else:
                label = '|%d><%d|'%(int(x[1])-1, int(x[2])-1)
                return [label.lower()], hmode

        return [label.lower()], hmode

    def _extract_operator_definition(hamiltonian_info: list[str], params: dict, mode_order: list[int]) -> SOP:
        active_modes = []
        for x in mode_order:
            if x >= 0:
                active_modes.append(x)
        H = SOP(len(active_modes))
        for line in hamiltonian_info:
            split_line = line.split('|')
            if len(split_line) == 0:
                continue
            coeff = split_line[0].strip() 
                   
            add_term = True

            terms = [x.strip() for x in split_line[1:]]
            val = QuanticsInputs._extract_coeff(coeff, params)
            contains_q3 = False
            if np.abs(val) < 1e-14:
                add_term = False


            labels, mode = QuanticsInputs._extract_mode_operator(terms[0])
            op = None
            if mode_order[mode-1] >= 0:
                if len(labels) == 1:
                    op = val*sOP(labels[0], mode_order[mode-1])
                    if(labels[0] == "q^2"):
                        H -= 0.5*val    #subtract off the zero point energy term if we have a quadratic coupling term
                else:
                    op = sOP(labels[0], mode_order[mode-1])
                    for i in range(1, len(labels)):
                        op += sOP(labels[i], mode_order[mode-1])
                    op *= val

                if len(terms) > 1:
                    for term in terms[1:]:
                        labels, mode = QuanticsInputs._extract_mode_operator(term)
                        if mode_order[mode-1] >= 0:
                            mop = None
                            if len(labels) == 1:
                                mop = sOP(labels[0], mode_order[mode-1])
                            else:
                                mop = sOP(labels[0], mode_order[mode-1])
                                for i in range(1, len(labels)):
                                    mop += sOP(labels[i],mode_order[mode-1])
                            op = op*mop
                        else:
                            add_term = False
            else:
                add_term = False

            if contains_q3:
                print(op)
            if add_term:
                H += op
        return H
    
    def _extract_wfn_modes(wfn_info: list[str]) -> dict:
        wfn_data = {}
        for line in wfn_info:
            if "build" in line:
                continue
            line = line.strip()
            if line.startswith("init_state"):
                wfn_data["el"] = ('nlevel', int(line.split("=")[1].strip()))
            else:
                data = line.split(" ")
                if data[1] != "HO":
                    raise RuntimeError("Currently the quantics converter only supports the harmonic oscillator basis set.")
                wfn_data[data[0]] = ("boson", ) + tuple(float(data[x]) for x in range(2, len(data)))
        return wfn_data

    def _convert_wfn(wfn_data: dict, sysinf: system_modes, modes: list[str]) -> list[list[float]]:
        #and compute the normalised representation of each mode in the basis we are using
        #that is form the vector |n><n|\psi_0>

        #for each primitive mode in each composite mode construct an array capable of storing the initial wavefunction
        res = [ [np.zeros((sysinf[i][j].lhd)) for j in range(sysinf[i].nmodes()) ] for i in range(sysinf.nmodes())]

        #now iterate over each term
        for k, v in wfn_data.items():

            #get which mode it corresponds to if the mode hasn't been bound in the mode list then
            #we skip adding the term into the Hamiltonian.
            if k in modes:
                primitive_ind = modes.index(k)

                #and extract the composite mode and primitive submode in the system info array\
                i1, i2 = sysinf.primitive_mode_index(primitive_ind)

                #and finally set the value of the term
                if v[0] == "nlevel":
                    res[i1][i2][v[1]-1] = 1.0
                elif v[0] == "boson":
                    centre = float(v[1])
                    momentum = float(v[2])
                    frequency = float(v[3])
                    mass = 1
                    if len(v) > 4:
                        mass = float(v[4])

                    if np.abs(frequency-1) > 1e-12 or np.abs(mass-1) > 1e-12:
                        raise RuntimeError("Currently the quantics converter does not support perturbed harmonic oscillator states.")
                    
                    if np.abs(centre) < 1e-12 and np.abs(momentum) < 1e-12:
                        res[i1][i2][0] = 1.0
                    else:
                        alpha = centre + 1.0j*momentum
                        res[i1][i2][0] = 1.0

                        op = sOP("disp"+str(alpha), 0)
                        si = system_modes(1)
                        si[0] = boson_mode(len(res[i1][i2]))
                        sop = site_operator(op, si)
                        opmat = np.array(sop.todense())

                        res[i1][i2] = opmat@res[i1][i2]
                        norm = np.dot(np.conj(res[i1][i2]), res[i1][i2])
                        res[i1][i2] /= np.sqrt(norm)

        #now for each composite mode construct the effective wavefunction by taking the kronecker product of each primitive modes wavefunction
        return [ft.reduce(np.kron, x) if len(x) > 1 else x[0] for x in res]

    def load_topology(fname: str) -> tuple[ntree, system_modes, list[str]]:
        """Load the tree topology, system information, and order of physical modes in the tree structure from a quantics input file

        :param fname: Path to input quantics file
        :type fname: str
        :return: The tree topology, system information and order of physical modes
        :rtype: tuple[ntree, system_modes, list[str]]
        """
        with open(fname, 'r') as fp:
            tree_info = QuanticsInputs._extract_section(fp, 'mlbasissection')
        with open(fname, 'r') as fp:
            mode_info = QuanticsInputs._extract_section(fp, 'primitivebasissection')

        #extract the mode information from the mode_info strings
        mode_dict = {}
        for mode_str in mode_info:
            label, t, d = QuanticsInputs._convert_primitive_modes(mode_str)
            mode_dict[label] = {'type': t, 'lhd': d}

        #get the mode combination and mode ordering information
        mode_combination, modes = QuanticsInputs._get_mode_ordering(tree_info)
        #extract the base tree structure
        topo = QuanticsInputs._convert_tree_info(tree_info)
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


    def load_operator(fname: str, modes: list[str]) -> SOP:
        """Load the Hamiltonian object defined in a quantics input file.

        :param fname: The quantics input file path
        :type fname: str
        :param modes: A list of the string labels for the modes in the order expected for the tree
        :type modes: list[str]
        :return: The Hamiltonian object defined in the quantics input file
        :rtype: SOP
        """
        with open(fname, 'r') as fp:
            parameter_info = QuanticsInputs._extract_section(fp, 'parametersection')
        with open(fname, 'r') as fp:
            hamiltonian_info = QuanticsInputs._extract_section(fp, 'hamiltoniansection')

        mode_info, hamiltonian_info = QuanticsInputs._split_hamiltonian_info(hamiltonian_info)
        params = QuanticsInputs._extract_parameter_dict(parameter_info)
        mode_order = QuanticsInputs._extract_mode_order(mode_info, modes)

        return QuanticsInputs._extract_operator_definition(hamiltonian_info, params, mode_order)

    def load_wfn(fname: str, sysinf: system_modes, modes: list[str]) -> list[np.ndarray]:
        """Load the direct product initial wavefunction from the quantics file

        :param fname: The quantics input file path
        :type fname: str
        :param sysinf: The system information of the model being considered
        :type sysinf: system_modes
        :param modes: A list of the string labels for the modes in the order expected for the tree
        :type modes: list[str]
        :return: A list of numpy arrays defining the direct product wavefunction
        :rtype: list[np.ndarray]
        """
        with open(fname, 'r') as fp:
            wfn_info = QuanticsInputs._extract_section(fp, 'init_wf')

        #extract the wavefunction information for each mode
        wfn_data = QuanticsInputs._extract_wfn_modes(wfn_info)
        return QuanticsInputs._convert_wfn(wfn_data, sysinf, modes)

    def load_all(fname : str) -> tuple[ntree, system_modes, SOP, list[list[np.ndarray]]]:
        """Function for loading tree topology, system information, hamiltonian and initial product wavefunction 
        from a quantics input file

        :param fname: The quantics input file path
        :type fname: str
        :return: tree topology, system information, hamiltonian and initial product wavefunction
        :rtype: tuple[ntree, system_modes, SOP, list[list[np.ndarray]]]
        """
        topo, sysinf, modes = QuanticsInputs.load_topology(fname)
        H = QuanticsInputs.load_operator(fname, modes)
        wfn = QuanticsInputs.load_wfn(fname, sysinf, modes)

        return topo, sysinf, H, wfn

