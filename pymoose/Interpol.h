#ifndef _pymoose_Interpol_h
#define _pymoose_Interpol_h
#include "PyMooseBase.h"

namespace pymoose
{

    class TableIterator;

    class InterpolationTable : public PyMooseBase
    {
      public:    
        static const std::string className_;
        InterpolationTable(Id id);
        InterpolationTable(std::string path);
        InterpolationTable(std::string name, Id parentId);
        InterpolationTable(std::string name, PyMooseBase& parent);
        InterpolationTable(const InterpolationTable& src,std::string name, PyMooseBase& parent);
        InterpolationTable(const InterpolationTable& src,std::string name, Id& parent);
        InterpolationTable(const Id& src,std::string name, Id& parent);
        InterpolationTable(const InterpolationTable& src,std::string path);
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
        double __get_dx() const;
        void __set_dx(double dx);
        double __get_sy() const;
        void __set_sy(double sy);
        // Manually edited the following two
        double __getitem__(unsigned int index) const;
        void __setitem__( unsigned int index, double value );
        TableIterator* __iter__();
        int __len__();
        int __get_calcMode() const;
        void __set_calcMode(int calc_mode);

        string dumpFile() const;
        void dumpFile(string fileName, bool append = false);
        void tabFill(int xdivs, int mode);

      protected:
        // This constructor is for allowing derived type (Table) to
        // have constructors exactly as if it was directly derived from PyMooseBase.
    
        InterpolationTable(std::string className, std::string objectName, Id parentId);
        InterpolationTable(std::string className, std::string path);    
        InterpolationTable(std::string className, std::string objectName, PyMooseBase& parent);
    };
}

#endif
