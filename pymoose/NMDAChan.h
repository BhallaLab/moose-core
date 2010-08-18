#ifndef _pymoose_NMDAChan_h
#define _pymoose_NMDAChan_h
#include "PyMooseBase.h"
#include "SynChan.h"
namespace pymoose{
    class NMDAChan : public SynChan
    {      public:
        static const std::string className_;
        NMDAChan(Id id);
        NMDAChan(std::string path);
        NMDAChan(std::string name, Id parentId);
        NMDAChan(std::string name, PyMooseBase& parent);
        NMDAChan( const NMDAChan& src, std::string name, PyMooseBase& parent);
        NMDAChan( const NMDAChan& src, std::string name, Id& parent);
        NMDAChan( const NMDAChan& src, std::string path);
        NMDAChan( const Id& src, std::string name, Id& parent);
	NMDAChan( const Id& src, std::string path);
        ~NMDAChan();
        const std::string& getType();
            double getTransitionParam(const unsigned int index) const;
            void setTransitionParam(const unsigned int index,double transitionParam);
            double __get_MgConc() const;
            void __set_MgConc(double MgConc);
            double __get_unblocked() const;
            double __get_saturation() const;
            void __set_saturation(double saturation);
    };
}
#endif
