#ifndef _pymoose_IntFire_h
#define _pymoose_IntFire_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class IntFire : public Neutral
    {      public:
        static const std::string className_;
        IntFire(Id id);
        IntFire(std::string path);
        IntFire(std::string name, Id parentId);
        IntFire(std::string name, PyMooseBase& parent);
        IntFire( const IntFire& src, std::string name, PyMooseBase& parent);
        IntFire( const IntFire& src, std::string name, Id& parent);
        IntFire( const IntFire& src, std::string path);
        IntFire( const Id& src, std::string name, Id& parent);
	IntFire( const Id& src, std::string path);
        ~IntFire();
        const std::string& getType();
            double __get_Vt() const;
            void __set_Vt(double Vt);
            double __get_Vr() const;
            void __set_Vr(double Vr);
            double __get_Rm() const;
            void __set_Rm(double Rm);
            double __get_Cm() const;
            void __set_Cm(double Cm);
            double __get_Vm() const;
            void __set_Vm(double Vm);
            double __get_tau() const;
            double __get_Em() const;
            void __set_Em(double Em);
            double __get_refractT() const;
            void __set_refractT(double refractT);
            double __get_initVm() const;
            void __set_initVm(double initVm);
            double __get_inject() const;
            void __set_inject(double inject);
    };
}
#endif
