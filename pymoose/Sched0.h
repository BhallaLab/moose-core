#ifndef _pymoose_Sched0_h
#define _pymoose_Sched0_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Sched0 : public PyMooseBase
    {    public:
        static const std::string className_;
        Sched0(Id id);
        Sched0(std::string path);
        Sched0(std::string name, Id parentId);
        Sched0(std::string name, PyMooseBase& parent);
        Sched0(const Sched0& src,std::string name, PyMooseBase& parent);
        Sched0(const Sched0& src,std::string name, Id& parent);
        Sched0(const Id& src,std::string name, Id& parent);
        Sched0(const Sched0& src,std::string path);
        ~Sched0();
        const std::string& getType();
    };
}

#endif
