# distutils: language = c++
# distutils: include_dirs = ., cymoose, ../cymoose/
# distutils: extra_compile_args = -DCYMOOSE

cimport Eref as _Eref
cimport CompartmentBase as CompartmentBase_

cdef extern from "../biophysics/Compartment.h" namespace "moose":

    cdef cppclass Compartment(CompartmentBase_.CompartmentBase):

         Compartment()

         void vSetVm( const _Eref.Eref& e, double Vm )
         double vGetVm( const _Eref.Eref& e ) const

         void vSetEm( const _Eref.Eref& e, double Em )
         double vGetEm( const _Eref.Eref& e ) const

         void vSetCm( const _Eref.Eref& e, double Cm )
         double vGetCm( const _Eref.Eref& e ) const

         void vSetRm( const _Eref.Eref& e, double Rm )
         double vGetRm( const _Eref.Eref& e ) const

         void vSetRa( const _Eref.Eref& e, double Ra )
         double vGetRa( const _Eref.Eref& e ) const

         double vGetIm( const _Eref.Eref& e ) const

         void vSetInject( const _Eref.Eref& e, double Inject )
         double vGetInject( const _Eref.Eref& e ) const

         void vSetInitVm( const _Eref.Eref& e, double initVm )
         double vGetInitVm( const _Eref.Eref& e ) const

