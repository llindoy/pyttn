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

from typing import TypeAlias

# import the blas backend
import pyttn.ttnpp.linalg as la

# and attempt to import the cuda backend
try:
    import pyttn.ttnpp.cuda.linalg as cula

    RandomEngine: TypeAlias = la.random_engine | cula.random_engine
    __cuda_import = True
except ImportError:
    RandomEngine: TypeAlias = la.random_engine
    __cuda_import = False


def random_engine(backend: str = "blas") -> RandomEngine:
    """Create a new random engine object associated with a given linear algebra backend

    :param backend: The backend to use for the created random engine object, defaults to "blas"
    :type backend: str, optional
    :return: The random engine object
    :rtype: RandomEngine
    """
    if backend == "blas":
        return la.random_engine()
    elif backend == "cuda" and __cuda_import:
        return cula.random_engine()
    else:
        raise RuntimeError("Backend not recognised.")
