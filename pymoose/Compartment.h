#ifndef _pymoose_Compartment_h
#define _pymoose_Compartment_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose
{
    class Compartment : public Neutral
    {
      public:
        static const std::string className_;
        Compartment(Id id);
        Compartment(std::string path);
        Compartment(std::string name, Id parentId);
        Compartment(std::string name, PyMooseBase& parent);
        Compartment(const Compartment& src,std::string name, PyMooseBase& parent);
        Compartment(const Compartment& src,std::string name, Id& parent);
        Compartment(const Id& src,std::string name, Id& parent);
        Compartment(const Compartment& src,std::string path);
        Compartment(const Id& src,std::string path);
        ~Compartment();
        const std::string& getType();
        double __get_Vm() const;
        void __set_Vm(double Vm);
        double __get_Cm() const;
        void __set_Cm(double Cm);
        double __get_Em() const;
        void __set_Em(double Em);
        double __get_Im() const;
        void __set_Im(double Im);
        double __get_inject() const;
        void __set_inject(double inject);
        double __get_initVm() const;
        void __set_initVm(double initVm);
        double __get_Rm() const;
        void __set_Rm(double Rm);
        double __get_Ra() const;
        void __set_Ra(double Ra);
        double __get_diameter() const;
        void __set_diameter(double diameter);
        double __get_length() const;
        void __set_length(double length);
        double __get_x() const;
        void __set_x(double x);
        double __get_y() const;
        void __set_y(double y);
        double __get_z() const;
        void __set_z(double z);
        double __get_x0() const;
        void __set_x0(double x);
        double __get_y0() const;
        void __set_y0(double y);
        double __get_z0() const;
        void __set_z0(double z);
    };
} // namespace

#endif
