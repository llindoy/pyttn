# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import copy
import numpy as np

from pyttn import system_modes


class ModeCombination:
    """A class for automatically combining modes of a system_modes object together to form a new system_modes
    object consisting of composite modes.  The approach implemented within this class involves sweeping from left
    to right and we attempt to combine the next mode into the current mode provided we have not already combined nbmax
    modes together in this mode and the local hilbert space dimension of the composite mode is less than nbmax.

    Constructor arguments

    :param nhilb: The maximum local Hilbert space dimension to allow during the mode combination process (default:1)
    :type nhilb: int, optional
    :param nbmax: The maximum  number of modes to combine(default:1)
    :type nbmax: int, optional
    :param blocksize: An optional blocksize argument.  For a blocksize of X, the current mode will contain a multiple of X modes and the new mode to add will contain X modes provided we have not reached the end of the chain(default:1)
    :type blocksize: int, optional


    Callable arguments
    :param system: the system_modes object defining all mode data
    :type mode_dims: system_modes
    :param blocksize: An optional blocksize argument, used to ignore the globally set blocksize (default is None)
    :type blocksize: int or None, optional
    :returns: The mode indices used to construct each composite mode.  ret[0] contains the indices of the first composite mode
    :rtype: list of list
    """

    def __init__(self, nhilb=1, nbmax=1, blocksize=1):
        self.nbmax = nbmax
        self.nhilb = nhilb
        self.blocksize = blocksize

    def mode_combination_array(self, mode_dims, mode_inds=None, _blocksize=None):
        """Perform the mode combination process on a array of mode dimensions.

        :param mode_dims: the mode dimensions
        :type mode_dims: list or np.ndarray
        :param mode_inds: the index labels of each of the modes (defaults to None) in which case we assume the modes are labelled [x for x in range(len(mode_dims))]
        :type mode_inds: list or np.ndarray, optional
        :param blocksize: An optional blocksize argument, used to ignore the globally set blocksize (default is None)
        :type blocksize: int or None, optional
        :returns: The mode indices used to construct each composite mode.  ret[0] contains the indices of the first composite mode
        :rtype: list of list
        """
        if not isinstance(mode_inds, (np.ndarray, list)):
            if mode_inds is None:
                mode_inds = [x for x in range(len(mode_dims))]

        nbmax = self.nbmax
        nhilbmax = self.nhilb
        blocksize = self.blocksize
        if _blocksize is not None:
            blocksize = _blocksize

        composite_modes = []

        all_modes_traversed = False
        cmode = []
        chilb = 1
        mode = 0
        while not all_modes_traversed:
            # if the current cmode object is empty then we just add the current mode to the composite mode and increment
            if len(cmode) == 0:
                # add in all modes up to the blocksize
                for j in range(blocksize):
                    cmode.append(mode_inds[mode])
                    chilb = chilb * mode_dims[mode]
                    mode += 1
                    # but if we get to the end of the mode dims array exit at this stage
                    if mode == len(mode_dims):
                        all_modes_traversed = True
                        composite_modes.append(copy.deepcopy(cmode))
            else:
                # othewise we check to see if the composite mode could accept the current mode without exceeding the bounds
                # then we add and increment
                nextdims = 1
                for j in range(blocksize):
                    if mode < len(mode_dims):
                        nextdims = nextdims * mode_dims[mode + j]

                if (
                    nbmax is None or len(cmode) < nbmax
                ) and chilb * nextdims <= nhilbmax:
                    # add all modes in the next block
                    for j in range(blocksize):
                        cmode.append(mode_inds[mode])
                        chilb = chilb * mode_dims[mode]
                        mode += 1

                        # bailing out early if we hit the end
                        if mode == len(mode_dims):
                            all_modes_traversed = True
                            composite_modes.append(copy.deepcopy(cmode))

                else:
                    # otherwise we have reached the end of the current composite mode.  We will now reset the composite
                    # mode object and we will not increment the mode so that it start a new composite mode object in the
                    # next iteration
                    composite_modes.append(copy.deepcopy(cmode))
                    cmode = []
                    chilb = 1

        return composite_modes

    def mode_combination_system(self, system, _blocksize=None):
        """Perform the mode combination process on a system_modes object.

        :param system: the system_modes object defining all mode data
        :type mode_dims: system_modes
        :param blocksize: An optional blocksize argument, used to ignore the globally set blocksize (default is None)
        :type blocksize: int or None, optional
        :returns: The mode indices used to construct each composite mode.  ret[0] contains the indices of the first composite mode
        :rtype: list of list
        """
        # extract the composite mode dimensions from the system
        mode_dims = [system[i].lhd() for i in range(len(system))]

        # now perform the mode combination on this array
        composite_modes = self.mode_combination_array(mode_dims, _blocksize=_blocksize)

        # and set up a composite system object using this information
        composite_system = system_modes(len(composite_modes))
        for i in range(len(composite_system)):
            for j in composite_modes[i]:
                composite_system[i].append(system[j])

        return composite_system

    def __call__(self, system, _blocksize=None):
        """Perform the mode combination process on a system_modes object.

        :param system: the system_modes object defining all mode data
        :type mode_dims: system_modes
        :param blocksize: An optional blocksize argument, used to ignore the globally set blocksize (default is None)
        :type blocksize: int or None, optional
        :returns: The mode indices used to construct each composite mode.  ret[0] contains the indices of the first composite mode
        :rtype: list of list
        """
        return self.mode_combination_system(system, _blocksize=_blocksize)
