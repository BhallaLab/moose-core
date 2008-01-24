#ifndef _pymoose_Molecule_h
#define _pymoose_Molecule_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Molecule : public PyMooseBase
    {    public:
        static const std::string className;
        Molecule(Id id);
        Molecule(std::string path);
        Molecule(std::string name, Id parentId);
        Molecule(std::string name, PyMooseBase& parent);
        ~Molecule();
        const std::string& getType();
        double __get_nInit() const;
        void __set_nInit(double nInit);
        double __get_volumeScale() const;
        void __set_volumeScale(double volumeScale);
        double __get_n() const;
        void __set_n(double n);
        int __get_mode() const;
        void __set_mode(int mode);
        int __get_slave_enable() const;
        void __set_slave_enable(int slave_enable);
        double __get_conc() const;
        void __set_conc(double conc);
        double __get_concInit() const;
        void __set_concInit(double concInit);
        double __get_nSrc() const;
        void __set_nSrc(double nSrc);
//        double,double __get_prd() const;
//        void __set_prd(double,double prd);
        double __get_sumTotal() const;
        void __set_sumTotal(double sumTotal);
    };
}

#endif
