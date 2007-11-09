#ifndef _pymoose_BinomialRng_h
#define _pymoose_BinomialRng_h
#include "RandGenerator.h"
class BinomialRng : public RandGenerator
{    public:
        static const std::string className;
        BinomialRng(Id id);
        BinomialRng(std::string path);
        BinomialRng(std::string name, Id parentId);
        BinomialRng(std::string name, PyMooseBase* parent);
        ~BinomialRng();
        const std::string& getType();
        int __get_n() const;
        void __set_n(int n);
        double __get_p() const;
        void __set_p(double p);
};
#endif
