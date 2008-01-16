#ifndef _pymoose_Sched2_h
#define _pymoose_Sched2_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Sched2 : public PyMooseBase
    {    public:
        static const std::string className;
        Sched2(Id id);
        Sched2(std::string path);
        Sched2(std::string name, Id parentId);
        Sched2(std::string name, PyMooseBase* parent);
        ~Sched2();
        const std::string& getType();
    };
}

#endif
