#ifndef _pymoose_IzhikevichNrn_h
#define _pymoose_IzhikevichNrn_h
#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose{
    class IzhikevichNrn : public Neutral
    {      public:
        static const std::string className_;
        IzhikevichNrn(Id id);
        IzhikevichNrn(std::string path);
        IzhikevichNrn(std::string name, Id parentId);
        IzhikevichNrn(std::string name, PyMooseBase& parent);
        IzhikevichNrn( const IzhikevichNrn& src, std::string name, PyMooseBase& parent);
        IzhikevichNrn( const IzhikevichNrn& src, std::string name, Id& parent);
        IzhikevichNrn( const IzhikevichNrn& src, std::string path);
        IzhikevichNrn( const Id& src, std::string name, Id& parent);
	IzhikevichNrn( const Id& src, std::string path);
        ~IzhikevichNrn();
        const std::string& getType();
            double __get_Vmax() const;
            void __set_Vmax(double Vmax);
            double __get_c() const;
            void __set_c(double c);
            double __get_d() const;
            void __set_d(double d);
            double __get_a() const;
            void __set_a(double a);
            double __get_b() const;
            void __set_b(double b);
            double __get_Vm() const;
            void __set_Vm(double Vm);
            double __get_u() const;
            double __get_Im() const;
            double __get_initVm() const;
            void __set_initVm(double initVm);
            double __get_initU() const;
            void __set_initU(double initU);
            double __get_alpha() const;
            void __set_alpha(double alpha);
            double __get_beta() const;
            void __set_beta(double beta);
            double __get_gamma() const;
            void __set_gamma(double gamma);
            double __get_Rm() const;
            void __set_Rm(double Rm);
    };
}
#endif
