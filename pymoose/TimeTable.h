#ifndef _pymoose_TimeTable_h
#define _pymoose_TimeTable_h
#include "PyMooseBase.h"
namespace pymoose{
    class TimeTable : public PyMooseBase
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
        ~TimeTable();
        const std::string& getType();
            double __get_maxTime() const;
            void __set_maxTime(double maxTime);
            vector < double > __get_tableVector() const;
            void __set_tableVector(vector < double > tableVector);
            unsigned int __get_tableSize() const;
    };
}
#endif
