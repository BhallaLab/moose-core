#ifndef _pymoose_SpikeGen_h
#define _pymoose_SpikeGen_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class SpikeGen : public Neutral
    {      public:
        static const std::string className_;
        SpikeGen(string className, string objectName, Id parentId);
        SpikeGen(string className, string path);
        SpikeGen(std::string className, std::string objectName, PyMooseBase& parent);
        SpikeGen(Id id);
        SpikeGen(std::string path);
        SpikeGen(std::string name, Id parentId);
        SpikeGen(std::string name, PyMooseBase& parent);
        SpikeGen( const SpikeGen& src, std::string name, PyMooseBase& parent);
        SpikeGen( const SpikeGen& src, std::string name, Id& parent);
        SpikeGen( const SpikeGen& src, std::string path);
        SpikeGen( const Id& src, std::string name, Id& parent);
        SpikeGen( const Id& src, std::string path);
        ~SpikeGen();
        const std::string& getType();
        double __get_threshold() const;
        void __set_threshold(double threshold);
        double __get_refractT() const;
        void __set_refractT(double refractT);
        double __get_absRefractT() const;
        void __set_absRefractT(double abs_refract);
        double __get_amplitude() const;
        void __set_amplitude(double amplitude);
        double __get_state() const;
        void __set_state(double state);
        int __get_edgeTriggered() const;
        void __set_edgeTriggered(int edgeTriggered);
        
    };
}

#endif
