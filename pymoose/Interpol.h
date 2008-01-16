#ifndef _pymoose_Interpol_h
#define _pymoose_Interpol_h
#include "PyMooseBase.h"
namespace pymoose
{

    class TableIterator;

    class InterpolationTable : public PyMooseBase
    {
      public:    
        static const std::string className;
        InterpolationTable(Id id);
        InterpolationTable(std::string path);
        InterpolationTable(std::string name, Id parentId);
        InterpolationTable(std::string name, PyMooseBase* parent);
        ~InterpolationTable();
        const std::string& getType();
        double __get_xmin() const;
        void __set_xmin(double xmin);
        double __get_xmax() const;
        void __set_xmax(double xmax);
        int __get_xdivs() const;
        void __set_xdivs(int xdivs);
        int __get_mode() const;
        void __set_mode(int mode);
        int __get_calc_mode() const;
        void __set_calc_mode(int calc_mode);
        double __get_dx() const;
        void __set_dx(double dx);
        double __get_sy() const;
        void __set_sy(double sy);
        // Manually edited the following two
        double __getitem__(unsigned int index) const;
        void __setitem__( unsigned int index, double value );
        TableIterator* __iter__();
        int __len__();
        void tabFill(int xdivs, int mode);
    
        double __get_lookupSrc() const;
        void __set_lookupSrc(double lookupSrc);
        double __get_lookup() const;
        void __set_lookup(double lookup);
        string dumpFile() const;
        void dumpFile(string fileName);
    
    
      protected:
        // This constructor is for allowing derived type (Table) to
        // have constructors exactly as if it was directly derived from PyMooseBase.
    
        InterpolationTable(std::string className, std::string objectName, Id parentId);
        InterpolationTable(std::string className, std::string path);    
        InterpolationTable(std::string className, std::string objectName, PyMooseBase* parent);
    };
}

#endif
