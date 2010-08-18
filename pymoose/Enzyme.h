#ifndef _pymoose_Enzyme_h
#define _pymoose_Enzyme_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class Enzyme : public Neutral
    {      public:
        static const std::string className_;
        Enzyme(Id id);
        Enzyme(std::string path);
        Enzyme(std::string name, Id parentId);
        Enzyme(std::string name, PyMooseBase& parent);
        Enzyme(const Enzyme& src,std::string name, PyMooseBase& parent);
        Enzyme(const Enzyme& src,std::string name, Id& parent);
        Enzyme(const Id& src,std::string name, Id& parent);
        Enzyme(const Enzyme& src,std::string path);
        Enzyme(const Id& src,std::string path);
        ~Enzyme();
        const std::string& getType();
        double __get_k1() const;
        void __set_k1(double k1);
        double __get_k2() const;
        void __set_k2(double k2);
        double __get_k3() const;
        void __set_k3(double k3);
        double __get_Km() const;
        void __set_Km(double Km);
        double __get_kcat() const;
        void __set_kcat(double kcat);
        bool __get_mode() const;
        void __set_mode(bool mode);
        double __get_scaleKm() const;
        void __set_scaleKm(double scaleKm);
        double __get_scaleKcat() const;
        void __set_scaleKcat(double scaleKcat);
        double __get_intramol() const;
        void __set_intramol(double intramol);
        double __get_x();
        void __set_x(double x);
        double __get_y();
        void __set_y(double y);
        string __get_xtreeTextFg();
        void __set_xtreeTextFg(string xtreeTextFg);
    };
}

#endif
