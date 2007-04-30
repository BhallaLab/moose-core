#ifndef _pymoose_HHGate_h
#define _pymoose_HHGate_h
#include "PyMooseBase.h"
#include "Interpol.h"

class HHGate : public PyMooseBase
{
  public:
    static const std::string className;
    HHGate(Id id);
    HHGate(std::string path);
    HHGate(std::string name, Id parentId);
    HHGate(std::string name, PyMooseBase* parent);
    ~HHGate();
    const std::string& getType();
    // These are manually inserted
    Id getA() const;
    Id getB() const;
};
#endif
