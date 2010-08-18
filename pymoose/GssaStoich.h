#ifndef _pymoose_GssaStoich_h
#define _pymoose_GssaStoich_h
#include "PyMooseBase.h"
#include "Stoich.h"
namespace pymoose{
    class GssaStoich : public Stoich
    {      public:
        static const std::string className_;
        GssaStoich(Id id);
        GssaStoich(std::string path);
        GssaStoich(std::string name, Id parentId);
        GssaStoich(std::string name, PyMooseBase& parent);
        GssaStoich( const GssaStoich& src, std::string name, PyMooseBase& parent);
        GssaStoich( const GssaStoich& src, std::string name, Id& parent);
        GssaStoich( const GssaStoich& src, std::string path);
        GssaStoich( const Id& src, std::string name, Id& parent);
	GssaStoich( const Id& src, std::string path);
        ~GssaStoich();
        const std::string& getType();
            string __get_method() const;
            void __set_method(string method);
            string __get_path() const;
            void __set_path(string path);
    };
}
#endif
