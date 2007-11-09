#ifndef _pymoose_NormalRng_h
#define _pymoose_NormalRng_h
#include "RandGenerator.h"
class NormalRng : public RandGenerator
{    public:
        static const std::string className;
        NormalRng(Id id);
        NormalRng(std::string path);
        NormalRng(std::string name, Id parentId);
        NormalRng(std::string name, PyMooseBase* parent);
        ~NormalRng();
        const std::string& getType();
        double __get_mean() const;
        void __set_mean(double mean);
        double __get_variance() const;
        void __set_variance(double variance);
        int __get_method() const;
        void __set_method(int method);
};
#endif
