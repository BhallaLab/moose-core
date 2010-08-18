#ifndef _pymoose_DifShell_h
#define _pymoose_DifShell_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class DifShell : public Neutral
    {      public:
        static const std::string className_;
        DifShell(Id id);
        DifShell(std::string path);
        DifShell(std::string name, Id parentId);
        DifShell(std::string name, PyMooseBase& parent);
        DifShell( const DifShell& src, std::string name, PyMooseBase& parent);
        DifShell( const DifShell& src, std::string name, Id& parent);
        DifShell( const DifShell& src, std::string path);
        DifShell( const Id& src, std::string name, Id& parent);
	DifShell( const Id& src, std::string path);
        ~DifShell();
        const std::string& getType();
        double __get_C() const;
        double __get_Ceq() const;
        void __set_Ceq(double Ceq);
        double __get_D() const;
        void __set_D(double D);
        double __get_valence() const;
        void __set_valence(double valence);
        double __get_leak() const;
        void __set_leak(double leak);
        unsigned int __get_shapeMode() const;
        void __set_shapeMode(unsigned int shapeMode);
        double __get_length() const;
        void __set_length(double length);
        double __get_diameter() const;
        void __set_diameter(double diameter);
        double __get_thickness() const;
        void __set_thickness(double thickness);
        double __get_volume() const;
        void __set_volume(double volume);
        double __get_outerArea() const;
        void __set_outerArea(double outerArea);
        double __get_innerArea() const;
        void __set_innerArea(double innerArea);
    };
}
#endif
