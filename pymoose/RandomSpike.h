#ifndef _pymoose_RandomSpike_h
#define _pymoose_RandomSpike_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class RandomSpike : public Neutral
    {      public:
        static const std::string className_;
        RandomSpike(Id id);
        RandomSpike(std::string path);
        RandomSpike(std::string name, Id parentId);
        RandomSpike(std::string name, PyMooseBase& parent);
        RandomSpike( const RandomSpike& src, std::string name, PyMooseBase& parent);
        RandomSpike( const RandomSpike& src, std::string name, Id& parent);
        RandomSpike( const RandomSpike& src, std::string path);
        RandomSpike( const Id& src, std::string name, Id& parent);
        RandomSpike( const Id& src, std::string path);
        ~RandomSpike();
        const std::string& getType();
        double __get_minAmp() const;
        void __set_minAmp(double minAmp);
        double __get_maxAmp() const;
        void __set_maxAmp(double maxAmp);
        double __get_rate() const;
        void __set_rate(double rate);
        double __get_resetValue() const;
        void __set_resetValue(double resetValue);
        double __get_state() const;
        void __set_state(double state);
        double __get_absRefract() const;
        void __set_absRefract(double absRefract);
        double __get_lastEvent() const;
        int __get_reset() const;
        void __set_reset(int reset);
    };
}

#endif
