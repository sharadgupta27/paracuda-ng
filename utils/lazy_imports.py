"""
Lazy import helpers.

Heavy scientific libraries (scikit-learn, SciPy, matplotlib, rasterio, XGBoost,
Optuna) each cost hundreds of milliseconds to seconds to import, but they are only
needed when the user actually runs an analysis, draws a plot or loads an image --
never just to display the GUI.  These proxies stand in for those imports at module
load time and transparently import the real module / object the first time it is
used, so the application window appears quickly while behavior stays identical.

A proxy resolves to the *real* object on first call or attribute access.  It also
pickles as the real object (via ``__reduce__``), so it is safe to send across the
process boundary used by scikit-learn's ``n_jobs`` / multiprocessing pools.

NOTE: a proxy is NOT usable as the second argument of ``isinstance``/``issubclass``
(it is an instance, not a class).  The few places that need that must import the
real class locally -- see ``DataIOMixin.view_model``.

@author: Sharad Kumar Gupta
"""
import importlib
import warnings


_sklearn_warnings_silenced = False


def _maybe_silence_sklearn(module_name):
    """Apply the ConvergenceWarning filter the first time scikit-learn is imported.

    The original app set this filter at startup, but that forced an eager (~4s)
    scikit-learn/SciPy import just to launch the GUI. Deferring it here keeps
    startup fast while still applying the filter before any model is fit (the
    first fit necessarily resolves a scikit-learn proxy, which lands here)."""
    global _sklearn_warnings_silenced
    if not _sklearn_warnings_silenced and module_name.split('.', 1)[0] == 'sklearn':
        _sklearn_warnings_silenced = True
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)


def _import_module(name):
    return importlib.import_module(name)


def _import_attr(module_name, attr):
    return getattr(importlib.import_module(module_name), attr)


class LazyModule:
    """Proxy for ``import <name>`` that defers the import until first use."""

    def __init__(self, name):
        self.__dict__["_lazy_name"] = name
        self.__dict__["_lazy_mod"] = None

    def _lazy_resolve(self):
        mod = self.__dict__["_lazy_mod"]
        if mod is None:
            name = self.__dict__["_lazy_name"]
            _maybe_silence_sklearn(name)
            mod = importlib.import_module(name)
            self.__dict__["_lazy_mod"] = mod
        return mod

    def __getattr__(self, item):
        return getattr(self._lazy_resolve(), item)

    def __call__(self, *args, **kwargs):
        return self._lazy_resolve()(*args, **kwargs)

    def __reduce__(self):
        return (_import_module, (self.__dict__["_lazy_name"],))


class LazyCallable:
    """Proxy for ``from <module> import <attr>`` (a class or function)."""

    def __init__(self, module_name, attr):
        self.__dict__["_lazy_module"] = module_name
        self.__dict__["_lazy_attr"] = attr
        self.__dict__["_lazy_obj"] = None

    def _lazy_resolve(self):
        obj = self.__dict__["_lazy_obj"]
        if obj is None:
            module_name = self.__dict__["_lazy_module"]
            _maybe_silence_sklearn(module_name)
            obj = getattr(importlib.import_module(module_name),
                          self.__dict__["_lazy_attr"])
            self.__dict__["_lazy_obj"] = obj
        return obj

    def __call__(self, *args, **kwargs):
        return self._lazy_resolve()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._lazy_resolve(), item)

    def __reduce__(self):
        return (_import_attr, (self.__dict__["_lazy_module"], self.__dict__["_lazy_attr"]))
