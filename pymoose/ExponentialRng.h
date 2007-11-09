#ifndef _pymoose_ExponentialRng_h
#define _pymoose_ExponentialRng_h
#include "RandGenerator.h"
class ExponentialRng : public RandGenerator
{    public:
        static const std::string className;
        ExponentialRng(Id id);
        ExponentialRng(std::string path);
        ExponentialRng(std::string name, Id parentId);
        ExponentialRng(std::string name, PyMooseBase* parent);
        ~ExponentialRng();
        const std::string& getType();
        double __get_mean() const;
        void __set_mean(double mean);
        int __get_method() const;
        void __set_method(int method);
};
#endif
