#ifndef _pymoose_Reaction_h
#define _pymoose_Reaction_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Reaction : public PyMooseBase
    {    public:
        static const std::string className;
        Reaction(Id id);
        Reaction(std::string path);
        Reaction(std::string name, Id parentId);
        Reaction(std::string name, PyMooseBase& parent);
        Reaction(const Reaction& src,std::string name, PyMooseBase& parent);
        Reaction(const Reaction& src,std::string name, Id& parent);
        Reaction(const Id& src,std::string name, Id& parent);
        Reaction(const Reaction& src,std::string path);
        ~Reaction();
        const std::string& getType();
        double __get_kf() const;
        void __set_kf(double kf);
        double __get_kb() const;
        void __set_kb(double kb);
        double __get_scaleKf() const;
        void __set_scaleKf(double scaleKf);
        double __get_scaleKb() const;
        void __set_scaleKb(double scaleKb);
    };
}

#endif
