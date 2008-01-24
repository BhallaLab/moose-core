#ifndef _pymoose_Kintegrator_h
#define _pymoose_Kintegrator_h
#include "PyMooseBase.h"
namespace pymoose
{
    class Kintegrator : public PyMooseBase
    {
      public:
        static const std::string className;
        Kintegrator(Id id);
        Kintegrator(std::string path);
        Kintegrator(std::string name, Id parentId);
        Kintegrator(std::string name, PyMooseBase& parent);
        ~Kintegrator();
        const std::string& getType();
        bool __get_isInitiatilized() const;
        void __set_isInitiatilized(bool isInitiatilized);
//         string __get_method() const;
//         void __set_method(string method);
        std::string imethod() const;
        std::string imethod(std::string);
    
    };
}

#endif
