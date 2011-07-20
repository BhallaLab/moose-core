#ifndef _pymoose_Efield_h
#define _pymoose_Efield_h
#include "Neutral.h"
namespace pymoose{
    class PyMooseBase;
    class Neutral;
    class Efield : public Neutral
    {
      public:
        static const std::string className_;
        Efield(std::string className, std::string objectName, Id parentId);
        Efield(std::string className, std::string path);
        Efield(std::string className, std::string objectName, PyMooseBase& parent);
        Efield(Id id);
        Efield(std::string path);
        Efield(std::string name, Id parentId);
        Efield(std::string name, PyMooseBase& parent);
        Efield( const Efield& src, std::string name, PyMooseBase& parent);
        Efield( const Efield& src, std::string name, Id& parent);
        Efield( const Efield& src, std::string path);
        Efield( const Id& src, std::string name, Id& parent);
        Efield( const Id& src, std::string path);
        ~Efield();
        const std::string& getType();
            double __get_x() const;
            void __set_x(double x);
            double __get_y() const;
            void __set_y(double y);
            double __get_z() const;
            void __set_z(double z);
            double __get_scale() const;
            void __set_scale(double scale);
            double __get_potential() const;
    };

}
#endif
