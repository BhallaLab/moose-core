#ifndef _pymoose_Reaction_h
#define _pymoose_Reaction_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class Reaction : public Neutral
    {      public:
        static const std::string className_;
        Reaction(Id id);
        Reaction(std::string path);
        Reaction(std::string name, Id parentId);
        Reaction(std::string name, PyMooseBase& parent);
        Reaction( const Reaction& src, std::string name, PyMooseBase& parent);
        Reaction( const Reaction& src, std::string name, Id& parent);
        Reaction( const Reaction& src, std::string path);
        Reaction( const Id& src, std::string name, Id& parent);
        Reaction( const Id& src, std::string path);
        ~Reaction();
        const std::string& getType();
        double __get_kf() const;
        void __set_kf(double scaleKf);
        double __get_kb() const;
        void __set_kb(double scaleKb);
        double __get_Kf() const;
        void __set_Kf(double scaleKf);
        double __get_Kb() const;
        void __set_Kb(double scaleKb);
        double __get_x();
        void __set_x(double x);
        double __get_y();
        void __set_y(double y);
        string __get_xtreeTextFg();
        void __set_xtreeTextFg(string xtreeTextFg);
        
    };
}

#endif
