# distutils: language = c++
# distutils: include_dirs = ., cymoose, ../cymoose/
# distutils: extra_compiler_args = -DCYTHON
# distutils: libraries = basecode

cimport Eref as _Eref
cimport CompartmentBase as CompartmentBase_

cdef extern from "../biophysics/Compartment.h" namespace "moose":
    cdef cppclass Compartment(CompartmentBase_.CompartmentBase):
        Compartment()
        void vSetVm( const _Eref.Eref& e, double Vm )
        double vGetVm( _Eref.Eref& e ) const
