#ifndef _pymoose_STPNMDAChan_h
#define _pymoose_STPNMDAChan_h
#include "STPSynChan.h"
namespace pymoose{
    class PyMooseBase;
    class STPSynChan;
    class STPNMDAChan : public STPSynChan
    {
      public:
        static const std::string className_;
        STPNMDAChan(std::string className, std::string objectName, Id parentId);
        STPNMDAChan(std::string className, std::string path);
        STPNMDAChan(std::string className, std::string objectName, PyMooseBase& parent);
        STPNMDAChan(Id id);
        STPNMDAChan(std::string path);
        STPNMDAChan(std::string name, Id parentId);
        STPNMDAChan(std::string name, PyMooseBase& parent);
        STPNMDAChan( const STPNMDAChan& src, std::string name, PyMooseBase& parent);
        STPNMDAChan( const STPNMDAChan& src, std::string name, Id& parent);
        STPNMDAChan( const STPNMDAChan& src, std::string path);
        STPNMDAChan( const Id& src, std::string name, Id& parent);
        STPNMDAChan( const Id& src, std::string path);
        ~STPNMDAChan();
        const std::string& getType();
        double __get_MgConc() const;
        void __set_MgConc(double MgConc);
        double __get_unblocked() const;
        double __get_saturation() const;
        void __set_saturation(double saturation);
        // The following were added manually
        double getTransitionParam(const unsigned int index) const;
        void setTransitionParam( const unsigned int index,double transitionParam );
    };

}
#endif
