#ifndef _pymoose_MathFunc_h
#define _pymoose_MathFunc_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class MathFunc : public Neutral
    {
      public:
        static const std::string className_;
        MathFunc(Id id);
        MathFunc(std::string path);
        MathFunc(std::string name, Id parentId);
        MathFunc(std::string name, PyMooseBase& parent);
        MathFunc(const MathFunc& src,std::string name, PyMooseBase& parent);
        MathFunc(const MathFunc& src,std::string name, Id& parent);
        MathFunc( const MathFunc& src, std::string path);
        MathFunc( const Id& src, std::string name, Id& parent);
	MathFunc( const Id& src, std::string path);
        ~MathFunc();
        const std::string& getType();
        string __get_mathML() const;
        void __set_mathML(string mathML);
        string __get_function() const;
        void __set_function(string function);
        double __get_result() const;
        void __set_result(double result);
    };
}
#endif
