#ifndef _pymoose_Interpol_h
#define _pymoose_Interpol_h
#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose
{

    class TableIterator;
    class Neutral;

    class Interpol : public Neutral
    {
      public:    
        static const std::string className_;
        Interpol(Id id);
        Interpol(std::string path);
        Interpol(std::string name, Id parentId);
        Interpol(std::string name, PyMooseBase& parent);
        Interpol(const Interpol& src,std::string name, PyMooseBase& parent);
        Interpol(const Interpol& src,std::string name, Id& parent);
        Interpol(const Id& src,std::string name, Id& parent);
        Interpol(const Interpol& src,std::string path);
        Interpol(const Id& src,std::string path);
        ~Interpol();
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
        const vector <double>& __get_table();
        string dumpFile() const;
        void dumpFile(string fileName, bool append = false);
        void tabFill(int xdivs, int mode);
        void load(string fileName, unsigned int skiplines);
        // double lookup(double index);
        // void appendTableVector(vector<double> vec);
        void clear();
        void push(double);
        void pop();

      protected:
        // This constructor is for allowing derived type (Table) to
        // have constructors exactly as if it was directly derived from PyMooseBase.
    
        Interpol(std::string className, std::string objectName, Id parentId);
        Interpol(std::string className, std::string path);    
        Interpol(std::string className, std::string objectName, PyMooseBase& parent);
    };
}

#endif
