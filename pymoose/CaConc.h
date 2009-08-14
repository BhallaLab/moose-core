#ifndef _pymoose_CaConc_h
#define _pymoose_CaConc_h
#include "PyMooseBase.h"
namespace pymoose
{
    
    class CaConc : public pymoose::PyMooseBase
    {
      public:
        static const std::string className_;
        CaConc(::Id id);
        CaConc(std::string path);
        CaConc(std::string name, ::Id parentId);
        CaConc(std::string name, pymoose::PyMooseBase& parent);
        CaConc(const CaConc& src,std::string name, PyMooseBase& parent);
        CaConc(const CaConc& src,std::string name, Id& parent);
        CaConc(const Id& src,std::string name, Id& parent);
        CaConc(const CaConc& src,std::string path);
        ~CaConc();
        const std::string& getType();
        double __get_Ca() const;
        void __set_Ca(double Ca);
        double __get_CaBasal() const;
        void __set_CaBasal(double CaBasal);
        double __get_Ca_base() const;
        void __set_Ca_base(double Ca_base);
        double __get_tau() const;
        void __set_tau(double tau);
        double __get_B() const;
        void __set_B(double B);
        double __get_thick() const;
        void __set_thick(double thick);
    };
} // namespace pymoose

#endif
