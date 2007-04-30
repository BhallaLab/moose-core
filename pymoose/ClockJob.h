#ifndef _pymoose_ClockJob_h
#define _pymoose_ClockJob_h
#include "PyMooseBase.h"
class ClockJob : public PyMooseBase
{    public:
        static const std::string className;
        ClockJob(Id id);
        ClockJob(std::string path);
        ClockJob(std::string name, Id parentId);
        ClockJob(std::string name, PyMooseBase* parent);
        ~ClockJob();
        const std::string& getType();
        double __get_runTime() const;
        void __set_runTime(double runTime);
        double __get_currentTime() const;
        void __set_currentTime(double currentTime);
        int __get_nsteps() const;
        void __set_nsteps(int nsteps);
        int __get_currentStep() const;
        void __set_currentStep(int currentStep);
        double __get_start() const;
        void __set_start(double start);
        int __get_step() const;
        void __set_step(int step);
};
#endif
