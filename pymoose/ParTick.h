#ifndef _pymoose_ParTick_h
#define _pymoose_ParTick_h
#include "PyMooseBase.h"
namespace pymoose
{
    class ParTick : public PyMooseBase
    {    public:
        static const std::string className;
        ParTick(Id id);
        ParTick(std::string path);
        ParTick(std::string name, Id parentId);
        ParTick(std::string name, PyMooseBase& parent);
        ParTick(const ParTick& src,std::string name, PyMooseBase& parent);
        ParTick(const ParTick& src,std::string name, Id& parent);
        ParTick(const Id& src,std::string name, Id& parent);
        ParTick(const ParTick& src,std::string path);
        ~ParTick();
        const std::string& getType();
    };
}
#endif
