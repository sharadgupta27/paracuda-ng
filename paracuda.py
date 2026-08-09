"""
PARACUDA-NG - entry point.

The application has been modularized into topical packages:
    preprocessing/  - spectral pre-processing and data utilities
    models/         - model construction, hyper-parameter tuning, batch scoring
    validation/     - cross-validation
    utils/          - file I/O, image processing, data conversion, help system
    gui/            - the Paracuda window (composed from mixins)

@author: Sharad Kumar Gupta
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# The ConvergenceWarning filter is applied lazily the first time scikit-learn is
# imported (see utils/lazy_imports.py) so that merely starting the GUI does not
# pull in scikit-learn/SciPy (~4s). It is set before any model is fit, so the
# suppression behaves exactly as before.

# Re-export so ``paracuda.Paracuda`` stays importable (headless tests, etc.).
from gui.app import Paracuda

if __name__ == "__main__":
    app = Paracuda()
    app.mainloop()
