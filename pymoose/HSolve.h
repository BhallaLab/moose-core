#ifndef _pymoose_HSolve_h
#define _pymoose_HSolve_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class HSolve : public Neutral
    {      public:
        static const std::string className_;
        HSolve(Id id);
        HSolve(std::string path);
        HSolve(std::string name, Id parentId);
        HSolve(std::string name, PyMooseBase& parent);
        HSolve( const HSolve& src, std::string name, PyMooseBase& parent);
        HSolve( const HSolve& src, std::string name, Id& parent);
        HSolve( const HSolve& src, std::string path);
        HSolve( const Id& src, std::string name, Id& parent);
        HSolve( const Id& src, std::string path);
        ~HSolve();
        const std::string& getType();
    
        const string __get_seed_path() const;
        void __set_seed_path(string path) const;
        int __get_NDiv() const;
        void __set_NDiv(int NDiv);
        double __get_VLo() const;
        void __set_VLo(double VLo);
        double __get_VHi() const;
        void __set_VHi(double VHi);
    };
}

#endif
