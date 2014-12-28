# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

cimport Id as _Id 

cdef class PyId:
    cdef _Id.Id* thisptr

    def __init__(self):
        self.thisptr = new _Id.Id()

