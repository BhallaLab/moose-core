# distutils: language = c++
# distutils: include_dirs = ., cymoose, ../cymoose/
# distutils: extra_compile_args = -DCYTHON
# distutils: libraries = basecode

cimport Eref as _Eref

cdef extern from "../biophysics/CompartmentBase.h" namespace "moose":
    cdef cppclass CompartmentBase:
        CompartmentBase()
