#ifndef _pymoose_SymCompartment_h
#define _pymoose_SymCompartment_h
#include "PyMooseBase.h"
#include "Compartment.h"

namespace pymoose{
    class SymCompartment : public Compartment
    {      public:
        static const std::string className_;
        SymCompartment(Id id);
        SymCompartment(std::string path);
        SymCompartment(std::string name, Id parentId);
        SymCompartment(std::string name, PyMooseBase& parent);
        SymCompartment( const SymCompartment& src, std::string name, PyMooseBase& parent);
        SymCompartment( const SymCompartment& src, std::string name, Id& parent);
        SymCompartment( const SymCompartment& src, std::string path);
        SymCompartment( const Id& src, std::string name, Id& parent);
        SymCompartment( const Id& src, std::string path);
        ~SymCompartment();
        const std::string& getType();
    };
}
#endif
