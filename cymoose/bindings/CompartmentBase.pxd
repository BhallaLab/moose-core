# distutils: language = c++
# distutils: include_dirs = ., cymoose, ../cymoose/
# distutils: extra_compile_args = -DCYTHON

cimport bindings.Eref as _Eref

cdef extern from "../biophysics/CompartmentBase.h" namespace "moose":
    cdef cppclass CompartmentBase:
        CompartmentBase()
