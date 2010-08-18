#ifndef _pymoose_Nernst_h
#define _pymoose_Nernst_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class Nernst : public Neutral
    {      public:
        static const std::string className_;
        Nernst(Id id);
        Nernst(std::string path);
        Nernst(std::string name, Id parentId);
        Nernst(std::string name, PyMooseBase& parent);
        Nernst( const Nernst& src, std::string name, PyMooseBase& parent);
        Nernst( const Nernst& src, std::string name, Id& parent);
        Nernst( const Nernst& src, std::string path);
        Nernst( const Id& src, std::string name, Id& parent);
        Nernst( const Id& src, std::string path);
        ~Nernst();
        const std::string& getType();
        double __get_E() const;
        void __set_E(double E);
        double __get_Temperature() const;
        void __set_Temperature(double Temperature);
        int __get_valence() const;
        void __set_valence(int valence);
        double __get_Cin() const;
        void __set_Cin(double Cin);
        double __get_Cout() const;
        void __set_Cout(double Cout);
        double __get_scale() const;
        void __set_scale(double scale);
        double __get_ESrc() const;
        void __set_ESrc(double ESrc);
        double __get_CinMsg() const;
        void __set_CinMsg(double CinMsg);
        double __get_CoutMsg() const;
        void __set_CoutMsg(double CoutMsg);
    };
}

#endif
