
cdef extern from "../basecode/Eref.h":
    cdef extern cppclass Eref:
        Eref() 
        char* data()

