#ifndef _pymoose_CaConc_h
#define _pymoose_CaConc_h
#include "PyMooseBase.h"
namespace pymoose
{
    
    class CaConc : public pymoose::PyMooseBase
    {
      public:
        static const std::string className;
        CaConc(::Id id);
        CaConc(std::string path);
        CaConc(std::string name, ::Id parentId);
        CaConc(std::string name, pymoose::PyMooseBase* parent);
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
        double __get_concSrc() const;
        void __set_concSrc(double concSrc);
        double __get_current() const;
        void __set_current(double current);
        double __get_increase() const;
        void __set_increase(double increase);
        double __get_decrease() const;
        void __set_decrease(double decrease);
        double __get_basalMsg() const;
        void __set_basalMsg(double basalMsg);
    };
} // namespace pymoose

#endif
