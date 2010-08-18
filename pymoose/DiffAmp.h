#ifndef _pymoose_DiffAmp_h
#define _pymoose_DiffAmp_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class DiffAmp : public Neutral
    {      public:
        static const std::string className_;
        DiffAmp(Id id);
        DiffAmp(std::string path);
        DiffAmp(std::string name, Id parentId);
        DiffAmp(std::string name, PyMooseBase& parent);
        DiffAmp( const DiffAmp& src, std::string name, PyMooseBase& parent);
        DiffAmp( const DiffAmp& src, std::string name, Id& parent);
        DiffAmp( const DiffAmp& src, std::string path);
        DiffAmp( const Id& src, std::string name, Id& parent);
	DiffAmp( const Id& src, std::string path);
        ~DiffAmp();
        const std::string& getType();
            double __get_gain() const;
            void __set_gain(double gain);
            double __get_saturation() const;
            void __set_saturation(double saturation);
            double __get_plus() const;
            double __get_minus() const;
            double __get_output() const;
    };
}
#endif
