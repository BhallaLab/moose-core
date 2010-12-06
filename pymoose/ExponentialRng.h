#ifndef _pymoose_ExponentialRng_h
#define _pymoose_ExponentialRng_h
#include "RandGenerator.h"
namespace pymoose
{
    class ExponentialRng : public RandGenerator
    {
      public:
        static const std::string className_;
        ExponentialRng(Id id);
        ExponentialRng(std::string path);
        ExponentialRng(std::string name, Id parentId);
        ExponentialRng(std::string name, PyMooseBase& parent);
        ExponentialRng(const ExponentialRng& src,std::string name, PyMooseBase& parent);
        ExponentialRng(const ExponentialRng& src,std::string name, Id& parent);
        ExponentialRng(const Id& src,std::string name, Id& parent);
        ExponentialRng(const ExponentialRng& src,std::string path);
        ExponentialRng(const Id& src,std::string path);
        ~ExponentialRng();
        const std::string& getType();
        int __get_method() const;
        void __set_method(int method);
    };
} // namepsace pymoose
#endif
