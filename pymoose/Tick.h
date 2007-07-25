#ifndef _pymoose_Tick_h
#define _pymoose_Tick_h
#include "PyMooseBase.h"
class ClockTick : public PyMooseBase
{    public:
        static const std::string className;
        ClockTick(Id id);
        ClockTick(std::string path);
        ClockTick(std::string name, Id parentId);
        ClockTick(std::string name, PyMooseBase* parent);
        ~ClockTick();
        const std::string& getType();
        double __get_dt() const;
        void __set_dt(double dt);
        int __get_stage() const;
        void __set_stage(int stage);
        int __get_ordinal() const;
        void __set_ordinal(int ordinal);
        double __get_nextTime() const;
        void __set_nextTime(double nextTime);
        string __get_path() const;
        void __set_path(string path);
        double __get_updateDtSrc() const;
        void __set_updateDtSrc(double updateDtSrc);
};
#endif
