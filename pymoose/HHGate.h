#ifndef _pymoose_HHGate_h
#define _pymoose_HHGate_h
#include "PyMooseBase.h"
#include "Table.h"

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
    Table* getA() const;
    Table* getB() const;
    void tabFill(int xdivs, int mode);
    void setupAlpha(double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size=3000, double min=-0.1, double max=0.05);
    void setupTau(double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size=3000, double min=-0.1, double max=0.05);
};
#endif
