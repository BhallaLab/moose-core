#ifndef _pymoose_Sched1_h
#define _pymoose_Sched1_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Sched1 : public PyMooseBase
    {    public:
        static const std::string className;
        Sched1(Id id);
        Sched1(std::string path);
        Sched1(std::string name, Id parentId);
        Sched1(std::string name, PyMooseBase& parent);
        Sched1(const Sched1& src,std::string name, PyMooseBase& parent);
        Sched1(const Sched1& src,std::string name, Id& parent);
        Sched1(const Id& src,std::string name, Id& parent);
        Sched1(const Sched1& src,std::string path);
        ~Sched1();
        const std::string& getType();
    };
}

#endif
