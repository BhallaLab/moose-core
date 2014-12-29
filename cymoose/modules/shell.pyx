# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

from bindings.PyShell import PyShell

shell_ = PyShell()
