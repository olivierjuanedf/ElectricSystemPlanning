import ctypes
import os
import sys


def _preload_highspy_native_library():
    """
    Pre-load highspy's bundled C++ shared library (libhighs.so) into the global symbol table
    (RTLD_GLOBAL) before any extension (or scipy/ortools) is loaded.
    This resolves the 'undefined symbol: _ZN5Highs13releaseMemoryEv' dynamic linking conflict on Linux.
    """
    search_paths = list(sys.path)
    try:
        import site
        if hasattr(site, "getsitepackages"):
            search_paths.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            search_paths.append(site.getusersitepackages())
    except Exception:
        pass

    for p in search_paths:
        if not isinstance(p, str) or not os.path.isdir(p):
            continue
        highspy_dir = os.path.join(p, "highspy")
        if not os.path.isdir(highspy_dir):
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(highspy_dir)
            except Exception:
                pass
        for lib_name in ("libhighs.so.1", "libhighs.so.1.15.1", "libhighs.so", "libhighs.1.dylib", "libhighs.dylib"):
            lib_path = os.path.join(highspy_dir, lib_name)
            if os.path.isfile(lib_path):
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    break
                except Exception:
                    pass


_preload_highspy_native_library()

try:
    import highspy  # noqa: F401
except ImportError:
    pass
