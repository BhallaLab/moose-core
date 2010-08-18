#ifndef _pymoose_Tick_h
#define _pymoose_Tick_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class Tick : public Neutral
    {    public:
        static const std::string className_;
        Tick(Id id);
        Tick(std::string path);
        Tick(std::string name, Id parentId);
        Tick(std::string name, PyMooseBase& parent);
        Tick(const Tick& src,std::string name, PyMooseBase& parent);
        Tick(const Tick& src,std::string name, Id& parent);
        Tick(const Id& src,std::string name, Id& parent);
        Tick(const Tick& src,std::string path);
        Tick(const Id& src,std::string path);
        ~Tick();
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
}

#endif
