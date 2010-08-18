#ifndef _pymoose_ClockJob_h
#define _pymoose_ClockJob_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class ClockJob : public Neutral
    {      public:
        static const std::string className_;
        ClockJob(Id id);
        ClockJob(std::string path);
        ClockJob(std::string name, Id parentId);
        ClockJob(std::string name, PyMooseBase& parent);
        ClockJob(const ClockJob& src,std::string name, PyMooseBase& parent);
        ClockJob(const ClockJob& src,std::string name, Id& parent);
        ClockJob(const Id& src,std::string name, Id& parent);
        ClockJob(const ClockJob& src,std::string path);
        ClockJob(const Id& src,std::string path);
        ~ClockJob();
        const std::string& getType();
        double __get_runTime() const;
        void __set_runTime(double runTime);
        double __get_currentTime() const;
        // void __set_currentTime(double currentTime);
        int __get_nsteps() const;
        void __set_nsteps(int nsteps);
        int __get_currentStep() const;
        int __get_autoschedule() const;
        void __set_autoschedule(int value);
        // void __set_currentStep(int currentStep);
        // double __get_start() const;
        // void __set_start(double start);
        // int __get_step() const;
        // void __set_step(int step);
        // void resched();
        // void reinit();
        // void stop();
        static vector <double> & getClocks(); // check with Upi - if this
        // clock is the set of
        // individual clocks or
        // different ticks only
    
    };
}

#endif
