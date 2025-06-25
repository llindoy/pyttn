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

from .bath_fitting import (
    OrthopolDiscretisation,
    DensityDiscretisation,
    BathDiscretisation,
    AAADecomposition,
    ESPRITDecomposition,
    CtExpFitDecomposition,
    SwExpFitDecomposition,
    ExpFitDecomposition,
    PoleDecomposition,
    MatsubaraDecomposition,
    softmspace,
)

from .baths import (
    bosonic_bath,
    fermionic_bath,
    discrete_bath,
    discrete_bosonic_bath,
    discrete_fermionic_bath,
    Bath,
    BosonicBath,
    CorrelatedBosonicBath,
    DiscreteBath,
    DiscreteBosonicBath,
    DiscreteFermionicBath,
    DiscreteOQSBath,
    DiscreteCorrelatedBosonicBath,
    DiscreteCorrelatedOQSBath,
    ExpFitBath,
    ExpFitBosonicBath,
    ExpFitFermionicBath,
    ExpFitOQSBath,
    ExpFitCorrelatedBosonicBath,
    ExpFitCorrelatedOQSBath,
    FermionicBath,
)
from .spectral_density import (
    CorrelatedSpectralDensity,
    RationalFunctionSpectralDensity,
    SumSpectralDensity,
    DebyeSpectralDensity,
    BrownianOscillatorSpectralDensity,
)


__all__: list[str] = [
    "OrthopolDiscretisation",
    "DensityDiscretisation",
    "BathDiscretisation",
    "AAADecomposition",
    "ESPRITDecomposition",
    "CtExpFitDecomposition",
    "SwExpFitDecomposition",
    "ExpFitDecomposition",
    "PoleDecomposition",
    "softmspace",
    "MatsubaraDecomposition",
    "bosonic_bath",
    "fermionic_bath",
    "discrete_bath",
    "discrete_bosonic_bath",
    "discrete_fermionic_bath",
    "Bath",
    "BosonicBath",
    "CorrelatedBosonicBath",
    "DiscreteBath",
    "DiscreteBosonicBath",
    "DiscreteFermionicBath",
    "DiscreteOQSBath",
    "DiscreteCorrelatedBosonicBath",
    "DiscreteCorrelatedOQSBath",
    "ExpFitBath",
    "ExpFitBosonicBath",
    "ExpFitFermionicBath",
    "ExpFitOQSBath",
    "ExpFitCorrelatedBosonicBath",
    "ExpFitCorrelatedOQSBath",
    "FermionicBath",
    "CorrelatedSpectralDensity",
    "RationalFunctionSpectralDensity",
    "SumSpectralDensity",
    "DebyeSpectralDensity",
    "BrownianOscillatorSpectralDensity",
]
