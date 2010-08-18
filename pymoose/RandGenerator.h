#ifndef _pymoose_RandGenerator_h
#define _pymoose_RandGenerator_h
#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose
{
    class RandGenerator : public Neutral
    {    public:
        static const std::string className_;
        RandGenerator(Id id);
        RandGenerator(string className, std::string path);
        RandGenerator(string className, std::string name, Id parentId);
        RandGenerator(const RandGenerator& src,std::string name, PyMooseBase& parent);
        RandGenerator(const RandGenerator& src, std::string name, Id& parent);
        RandGenerator(const Id& src,std::string name, Id& parent);
        RandGenerator(const Id& src,std::string path);
        RandGenerator(const RandGenerator& src,std::string path);
        RandGenerator(string className, std::string name, PyMooseBase& parent);
//        ~RandGenerator();
        const std::string& getType();
        double __get_sample() const;
        void __set_sample(double sample);
        double __get_mean() const;
        void __set_mean(double mean);
        double __get_variance() const;
        void __set_variance(double variance);
        double __get_output() const;
        void __set_output(double output);
    };
}

#endif
