#ifndef _pymoose_Interpol2D_h
#define _pymoose_Interpol2D_h
#include "PyMooseBase.h"
namespace pymoose{
    class Interpol2D : public PyMooseBase
    {      public:
        static const std::string className_;
        Interpol2D(Id id);
        Interpol2D(std::string path);
        Interpol2D(std::string name, Id parentId);
        Interpol2D(std::string name, PyMooseBase& parent);
        Interpol2D( const Interpol2D& src, std::string name, PyMooseBase& parent);
        Interpol2D( const Interpol2D& src, std::string name, Id& parent);
        Interpol2D( const Interpol2D& src, std::string path);
        Interpol2D( const Id& src, std::string name, Id& parent);
	Interpol2D( const Id& src, std::string path);
        ~Interpol2D();
        const std::string& getType();
            double __get_ymin() const;
            void __set_ymin(double ymin);
            double __get_ymax() const;
            void __set_ymax(double ymax);
            int __get_ydivs() const;
            void __set_ydivs(int ydivs);
            double __get_dy() const;
            void __set_dy(double dy);
            none __get_tableVector2D() const;
            void __set_tableVector2D(none tableVector2D);
    };
}
#endif
