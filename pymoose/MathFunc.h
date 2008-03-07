#ifndef _pymoose_MathFunc_h
#define _pymoose_MathFunc_h
#include "PyMooseBase.h"
namespace pymoose
{
    class MathFunc : public PyMooseBase
    {
      public:
        static const std::string className;
        MathFunc(Id id);
        MathFunc(std::string path);
        MathFunc(std::string name, Id parentId);
        MathFunc(std::string name, PyMooseBase& parent);
        MathFunc(const MathFunc& src,std::string name, PyMooseBase& parent);
        MathFunc(const MathFunc& src,std::string name, Id& parent);
        MathFunc(const Id& src,std::string name, Id& parent);
        MathFunc(const MathFunc& src,std::string path);
        ~MathFunc();
        const std::string& getType();
        string __get_mathML() const;
        void __set_mathML(string mathML);
        string __get_function() const;
        void __set_function(string function);
        double __get_result() const;
        void __set_result(double result);
        double __get_output() const;
        void __set_output(double output);
        double __get_args() const;
        void __set_args(double args);
        double __get_arg1() const;
        void __set_arg1(double arg1);
        double __get_arg2() const;
        void __set_arg2(double arg2);
        double __get_arg3() const;
        void __set_arg3(double arg3);
        double __get_arg4() const;
        void __set_arg4(double arg4);
    };
}

#endif
