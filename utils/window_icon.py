"""
Window / taskbar icon helpers.

``iconbitmap()`` sets the icon Windows draws in the title bar, but it does not
decide what the *taskbar* button shows.  The shell picks that from the process's
Application User Model ID, which for a Python GUI defaults to the interpreter --
so PARACUDA-NG ends up sharing (and displaying) the generic Python icon.  Telling
the shell that this process is its own application makes it fall back to the
window's own icon, and keeps PARACUDA-NG from being grouped with unrelated Python
programs on the taskbar.

The AppUserModelID has to be set *before* the first window is created, so call
``set_app_user_model_id()`` at the very top of a window class's ``__init__``,
ahead of ``tk.Tk.__init__``.

@author: Sharad Kumar Gupta
"""
import os
import sys

__all__ = ["APP_ID", "set_app_user_model_id", "icon_path", "apply_icon"]

# Vendor.Product.Component.Version - the format Microsoft documents for an
# explicit AppUserModelID.
APP_ID = "SharadKumarGupta.ParacudaNG.GUI.1"

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_app_user_model_id(app_id=APP_ID):
    """Declare this process a distinct application to the Windows shell.

    A no-op on other platforms, and on any Windows where the call is
    unavailable -- the icon simply stays as it was.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def icon_path():
    """Return the path of the bundled icon, or ``None`` if it is missing."""
    for name in ("paracuda.ico", "icon.ico"):
        path = os.path.join(_PACKAGE_DIR, name)
        if os.path.exists(path):
            return path
    return None


def apply_icon(window):
    """Give ``window`` the PARACUDA-NG icon, and use it for its dialogs too.

    ``default=`` makes every Toplevel created afterwards inherit the icon; it is
    a Windows-only option, hence the fallback to a plain ``iconbitmap`` call.
    """
    path = icon_path()
    if path is None:
        return
    try:
        window.iconbitmap(default=path)
    except Exception:
        try:
            window.iconbitmap(path)
        except Exception:
            pass
