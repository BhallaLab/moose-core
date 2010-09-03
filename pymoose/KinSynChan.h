#ifndef _pymoose_KinSynChan_h
#define _pymoose_KinSynChan_h
#include "PyMooseBase.h"
#include "SynChan.h"

namespace pymoose{
    class KinSynChan : public SynChan
    {      public:
        static const std::string className_;
        KinSynChan(Id id);
        KinSynChan(std::string path);
        KinSynChan(std::string name, Id parentId);
        KinSynChan(std::string name, PyMooseBase& parent);
        KinSynChan( const KinSynChan& src, std::string name, PyMooseBase& parent);
        KinSynChan( const KinSynChan& src, std::string name, Id& parent);
        KinSynChan( const KinSynChan& src, std::string path);
        KinSynChan( const Id& src, std::string name, Id& parent);
        KinSynChan( const Id& src, std::string path);
        ~KinSynChan();
        const std::string& getType();
            double __get_rInf() const;
            void __set_rInf(double rInf);
            double __get_tau1() const;
            void __set_tau1(double tauR);
            double __get_pulseWidth() const;
            void __set_pulseWidth(double pulseWidth);
    };
}
#endif
