# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

from libcpp.string cimport string
cimport bindings.Eref as _eref

cdef extern from "../shell/Neutral.h":
    cdef cppclass Neutral:
        void setName(const _eref.Eref&, string)

