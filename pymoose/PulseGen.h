#ifndef _pymoose_PulseGen_h
#define _pymoose_PulseGen_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class PulseGen : public Neutral
    {      public:
        static const std::string className_;
        PulseGen(Id id);
        PulseGen(std::string path);
        PulseGen(std::string name, Id parentId);
        PulseGen(std::string name, PyMooseBase& parent);
        PulseGen( const PulseGen& src, std::string name, PyMooseBase& parent);
        PulseGen( const PulseGen& src, std::string name, Id& parent);
        PulseGen( const PulseGen& src, std::string path);
        PulseGen( const Id& src, std::string name, Id& parent);
        PulseGen( const Id& src, std::string path);
        ~PulseGen();
        const std::string& getType();
        double __get_firstLevel() const;
        void __set_firstLevel(double firstLevel);
        double __get_firstWidth() const;
        void __set_firstWidth(double firstWidth);
        double __get_firstDelay() const;
        void __set_firstDelay(double firstDelay);
        double __get_secondLevel() const;
        void __set_secondLevel(double secondLevel);
        double __get_secondWidth() const;
        void __set_secondWidth(double secondWidth);
        double __get_secondDelay() const;
        void __set_secondDelay(double secondDelay);
        double __get_baseLevel() const;
        void __set_baseLevel(double baseLevel);
        double __get_output() const;
        double __get_trigTime() const;
        void __set_trigTime(double trigTime);        
        int __get_trigMode() const;
        void __set_trigMode(int trigMode);
        int __get_prevInput() const;
        void setCount(int count);
        int getCount() const;
        
        void setDelay(int index, double value);
        double getDelay(int index);
        void setLevel(int index, double value);
        double getLevel(int index);
        void setWidth(int index, double value);
        double getWidth(int index);        
    };
}

#endif
