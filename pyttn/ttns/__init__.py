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
