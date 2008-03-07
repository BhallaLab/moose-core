#ifndef _pymoose_KineticHub_h
#define _pymoose_KineticHub_h
#include "PyMooseBase.h"
namespace pymoose
{
    class KineticHub : public PyMooseBase
    {
      public:
        static const std::string className;
        KineticHub(Id id);
        KineticHub(std::string path);
        KineticHub(std::string name, Id parentId);
        KineticHub(std::string name, PyMooseBase& parent);
        KineticHub(const KineticHub& src,std::string name, PyMooseBase& parent);
        KineticHub(const KineticHub& src,std::string name, Id& parent);
        KineticHub(const Id& src,std::string name, Id& parent);
        KineticHub(const KineticHub& src,std::string path);
        ~KineticHub();
        const std::string& getType();
        unsigned int __get_nMol() const;
        void __set_nMol(unsigned int nMol);
        unsigned int __get_nReac() const;
        void __set_nReac(unsigned int nReac);
        unsigned int __get_nEnz() const;
        void __set_nEnz(unsigned int nEnz);
        void destroy();
    
        // none __get_destroy() const;
//         void __set_destroy(none destroy);
        double __get_molSum() const;
        void __set_molSum(double molSum);
    };
}

#endif
