#ifndef _pymoose_Leakage_h
#define _pymoose_Leakage_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{
    class Leakage : public Neutral
    {      public:
        static const std::string className_;
        Leakage(Id id);
        Leakage(std::string path);
        Leakage(std::string name, Id parentId);
        Leakage(std::string name, PyMooseBase& parent);
        Leakage( const Leakage& src, std::string name, PyMooseBase& parent);
        Leakage( const Leakage& src, std::string name, Id& parent);
        Leakage( const Leakage& src, std::string path);
        Leakage( const Id& src, std::string name, Id& parent);
	Leakage( const Id& src, std::string path);
        ~Leakage();
        const std::string& getType();
            double __get_Ek() const;
            void __set_Ek(double Ek);
            double __get_Gk() const;
            void __set_Gk(double Gk);
            double __get_Ik() const;
            double __get_activation() const;
            void __set_activation(double activation);
    };
}
#endif
