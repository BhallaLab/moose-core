#ifndef _pymoose_Kintegrator_h
#define _pymoose_Kintegrator_h
#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose
{
    class Kintegrator : public Neutral
    {
      public:
        static const std::string className_;
        Kintegrator(Id id);
        Kintegrator(std::string path);
        Kintegrator(std::string name, Id parentId);
        Kintegrator(std::string name, PyMooseBase& parent);
        Kintegrator( const Kintegrator& src, std::string name, PyMooseBase& parent);
        Kintegrator( const Kintegrator& src, std::string name, Id& parent);
        Kintegrator( const Kintegrator& src, std::string path);
        Kintegrator( const Id& src, std::string name, Id& parent);
        Kintegrator( const Id& src, std::string path);
        ~Kintegrator();
        const std::string& getType();
            bool __get_isInitiatilized() const;
            string  __get_method() const;
            void __set_method(string method);
    };
}

#endif
