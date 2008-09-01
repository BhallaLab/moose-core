#ifndef _pymoose_KinCompt_h
#define _pymoose_KinCompt_h
#include "PyMooseBase.h"
namespace pymoose{
    class KinCompt : public PyMooseBase
    {      public:
        static const std::string className;
        KinCompt(Id id);
        KinCompt(std::string path);
        KinCompt(std::string name, Id parentId);
        KinCompt(std::string name, PyMooseBase& parent);
        KinCompt( const KinCompt& src, std::string name, PyMooseBase& parent);
        KinCompt( const KinCompt& src, std::string name, Id& parent);
        KinCompt( const KinCompt& src, std::string path);
        KinCompt( const Id& src, std::string name, Id& parent);
        ~KinCompt();
        const std::string& getType();
            double __get_volume() const;
            void __set_volume(double volume);
            double __get_area() const;
            void __set_area(double area);
            double __get_perimeter() const;
            void __set_perimeter(double perimeter);
            double __get_size() const;
            void __set_size(double size);
            unsigned int __get_numDimensions() const;
            void __set_numDimensions(unsigned int numDimensions);
    };
}
#endif
