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

import importlib
import pkgutil

__all__ = []
for mod_info in pkgutil.iter_modules(__path__, __name__ + '.'):
    mod = importlib.import_module(mod_info.name)

    # Emulate `from mod import *`
    try:
        names = mod.__dict__['__all__']
    except KeyError:
        names = [k for k in mod.__dict__ if not k.startswith('_')]

    globals().update({k: getattr(mod, k) for k in names})
    __all__ = __all__ + names
