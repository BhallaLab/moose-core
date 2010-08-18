#ifndef _pymoose_GHK_h
#define _pymoose_GHK_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class GHK : public Neutral
    {      public:
        static const std::string className_;
        GHK(Id id);
        GHK(std::string path);
        GHK(std::string name, Id parentId);
        GHK(std::string name, PyMooseBase& parent);
        GHK( const GHK& src, std::string name, PyMooseBase& parent);
        GHK( const GHK& src, std::string name, Id& parent);
        GHK( const GHK& src, std::string path);
        GHK( const Id& src, std::string name, Id& parent);
	GHK( const Id& src, std::string path);
        ~GHK();
        const std::string& getType();
            double __get_Ik() const;
            double __get_Gk() const;
            double __get_Ek() const;
            double __get_T() const;
            void __set_T(double T);
            double __get_p() const;
            void __set_p(double p);
            double __get_Vm() const;
            void __set_Vm(double Vm);
            double __get_Cin() const;
            void __set_Cin(double Cin);
            double __get_Cout() const;
            void __set_Cout(double Cout);
            double __get_valency() const;
            void __set_valency(double valency);
    };
}
#endif
