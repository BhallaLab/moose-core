#ifndef _pymoose_TimeTable_h
#define _pymoose_TimeTable_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class TimeTable : public Neutral
    {      public:
        static const std::string className_;
        TimeTable(Id id);
        TimeTable(std::string path);
        TimeTable(std::string name, Id parentId);
        TimeTable(std::string name, PyMooseBase& parent);
        TimeTable( const TimeTable& src, std::string name, PyMooseBase& parent);
        TimeTable( const TimeTable& src, std::string name, Id& parent);
        TimeTable( const TimeTable& src, std::string path);
        TimeTable( const Id& src, std::string name, Id& parent);
        TimeTable( const Id& src, std::string path);
        ~TimeTable();
        const std::string& getType();
        double __get_maxTime() const;
        void __set_maxTime(double maxTime);
        vector < double > __get_tableVector() const;
        void __set_tableVector(const vector < double >& tableVector);
        unsigned int __get_tableSize() const;
        
        double __getitem__(const unsigned int index) const;
        
        void __setitem__(const unsigned int index, double value);
        double __get_state();
        
        int __get_method();
        
        void __set_method(const int method);

        const std::string __get_filename();
        
        void __set_filename(const std::string& filename);
    };
}
#endif
