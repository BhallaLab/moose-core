#ifndef _pymoose_TickTest_h
#define _pymoose_TickTest_h
#include "PyMooseBase.h"
namespace pymoose
{
    class TickTest : public PyMooseBase
    {    public:
        static const std::string className_;
        TickTest(Id id);
        TickTest(std::string path);
        TickTest(std::string name, Id parentId);
        TickTest(std::string name, PyMooseBase& parent);
        ~TickTest();
        const std::string& getType();
    };
}

#endif
