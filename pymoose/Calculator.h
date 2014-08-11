#ifndef _pymoose_Calculator_h
#define _pymoose_Calculator_h
#include "Neutral.h"
namespace pymoose{
    class PyMooseBase;
    class Neutral;
    class Calculator : public Neutral
    {
      public:
        static const std::string className_;
        Calculator(std::string className, std::string objectName, Id parentId);
        Calculator(std::string className, std::string path);
        Calculator(std::string className, std::string objectName, PyMooseBase& parent);
        Calculator(Id id);
        Calculator(std::string path);
        Calculator(std::string name, Id parentId);
        Calculator(std::string name, PyMooseBase& parent);
        Calculator( const Calculator& src, std::string name, PyMooseBase& parent);
        Calculator( const Calculator& src, std::string name, Id& parent);
        Calculator( const Calculator& src, std::string path);
        Calculator( const Id& src, std::string name, Id& parent);
        Calculator( const Id& src, std::string path);
        ~Calculator();
        const std::string& getType();
            double __get_initValue() const;
            void __set_initValue(double initValue);
            double __get_value() const;
    };

}
#endif
