# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np


def softmspace(
    start: float, stop: float, N: int, beta: float = 1, endpoint: bool = True
):
    r"""A function for generating a set of points with separations defined by a softmspace function.
    This ensures that at small values we points near a logspace and at large values we get something closer to linspace, allowing for
    logspace resolution at low frequencies but retaining moderate resolution at high frequencies

    :param start: The starting value for the softmspace
    :type start: float
    :param stop: The ending value for the softmspace
    :type stop: float
    :param N: The number of points generated
    :type N: int
    :param beta: The parameter determining how rapidly we transition from 0 to 1 (default: 1)
    :type beta: float, optional
    :param endpoint: Whether or not the top endpoint should be included in the set of generated points
    :type endpoint: bool, optional
    """
    start = (np.log(1 - (np.exp(-beta * start))) + beta * start) / beta
    # start = np.log(np.exp(beta*start)-1)/beta
    stop = (np.log(1 - (np.exp(-beta * stop))) + beta * stop) / beta
    # stop = np.log(np.exp(beta*stop)-1)/beta

    dx = (stop - start) / N
    if endpoint:
        dx = (stop - start) / (N - 1)

    return np.logaddexp(beta * (np.arange(N) * dx + start), 0) / beta
    # return np.log(np.exp(beta*(np.arange(N)*dx  + start))+1)/beta
