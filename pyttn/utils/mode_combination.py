import copy
import numpy as np

from pyttn import system_modes


class ModeCombination:
    def __init__(self, nhilb = 1, nbmax = 1, blocksize=1):
        self.nbmax = nbmax
        self.nhilb = nhilb
        self.blocksize = blocksize


    def mode_combination_array(self, mode_dims, mode_inds=None, _blocksize=None):
        if not isinstance(mode_inds, (np.ndarray, list)):
            if mode_inds is None:
                mode_inds = [x for x in range(len(mode_dims))]

        nbmax = self.nbmax
        nhilbmax = self.nhilb
        blocksize = self.blocksize
        if not _blocksize is None:
            blocksize = _blocksize

        composite_modes = []

        all_modes_traversed = False
        cmode = []
        chilb = 1
        mode = 0
        while not all_modes_traversed:
            #if the current cmode object is empty then we just add the current mode to the composite mode and increment
            if(len(cmode) == 0):
                #add in all modes up to the blocksize
                for j in range(blocksize):
                    cmode.append(mode_inds[mode])
                    chilb = chilb*mode_dims[mode]
                    mode += 1
                    #but if we get to the end of the mode dims array exit at this stage
                    if(mode == len(mode_dims)):
                        all_modes_traversed = True
                        composite_modes.append(copy.deepcopy(cmode))
            else:
                #othewise we check to see if the composite mode could accept the current mode without exceeding the bounds
                #then we add and increment
                nextdims = 1
                for j in range(blocksize):
                    if(mode < len(mode_dims)):
                        nextdims = nextdims*mode_dims[mode+j]

                if (nbmax == None or len(cmode) < nbmax) and chilb*nextdims <= nhilbmax:
                    #add all modes in the next block
                    for j in range(blocksize):
                        cmode.append(mode_inds[mode])
                        chilb = chilb*mode_dims[mode]
                        mode += 1

                        #bailing out early if we hit the end
                        if(mode == len(mode_dims)):
                            all_modes_traversed = True
                            composite_modes.append(copy.deepcopy(cmode))

                else:
                    #otherwise we have reached the end of the current composite mode.  We will now reset the composite 
                    #mode object and we will not increment the mode so that it start a new composite mode object in the
                    #next iteration
                    composite_modes.append(copy.deepcopy(cmode))
                    cmode = []
                    chilb = 1

        return composite_modes

    def mode_combination_system(self, system, _blocksize=None):
        #extract the composite mode dimensions from the system
        mode_dims = [system[i].lhd() for i in range(len(system))]

        #now perform the mode combination on this array
        composite_modes = self.mode_combination_array(mode_dims, _blocksize=_blocksize)

        #and set up a composite system object using this information
        composite_system = system_modes(len(composite_modes))
        for i in range(len(composite_system)):
            for j in composite_modes[i]:
                composite_system[i].append(system[j])

        return composite_system

    def __call__(self, system, _blocksize = None):
        return self.mode_combination_system(system, _blocksize=_blocksize)
