#ifndef _pymoose_Stoich_h
#define _pymoose_Stoich_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Stoich : public PyMooseBase
    {    public:
        static const std::string className;
        Stoich(Id id);
        Stoich(std::string path);
        Stoich(std::string name, Id parentId);
        Stoich(std::string name, PyMooseBase* parent);
        ~Stoich();
        const std::string& getType();
        unsigned int __get_nMols() const;
        void __set_nMols(unsigned int nMols);
        unsigned int __get_nVarMols() const;
        void __set_nVarMols(unsigned int nVarMols);
        unsigned int __get_nSumTot() const;
        void __set_nSumTot(unsigned int nSumTot);
        unsigned int __get_nBuffered() const;
        void __set_nBuffered(unsigned int nBuffered);
        unsigned int __get_nReacs() const;
        void __set_nReacs(unsigned int nReacs);
        unsigned int __get_nEnz() const;
        void __set_nEnz(unsigned int nEnz);
        unsigned int __get_nMMenz() const;
        void __set_nMMenz(unsigned int nMMenz);
        unsigned int __get_nExternalRates() const;
        void __set_nExternalRates(unsigned int nExternalRates);
        bool __get_useOneWayReacs() const;
        void __set_useOneWayReacs(bool useOneWayReacs);
//         string __get_path() const;
//         void __set_path(string path);
        std::string path() const;
        std::string path(std::string path);
    
        unsigned int __get_rateVectorSize() const;
        void __set_rateVectorSize(unsigned int rateVectorSize);
    };
}

#endif
