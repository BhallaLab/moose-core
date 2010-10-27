#ifndef _pymoose_Table_h
#define _pymoose_Table_h
#include "PyMooseBase.h"
#include "Interpol.h"
namespace pymoose
{
enum {TAB_IO, TAB_LOOP, TAB_ONCE, TAB_BUF, TAB_SPIKE, TAB_FIELDS, TAB_DELAY };
class Table : public Interpol
{
  public:
    
    static const std::string className_;
    Table(Id id);
    Table(std::string path);
    Table(std::string name, Id parentId);
    Table(std::string name, PyMooseBase& parent);
    Table(const Table& src,std::string name, PyMooseBase& parent);
    Table(const Table& src,std::string name, Id& parent);
    Table(const Id& src,std::string name, Id& parent);
    Table(const Table& src,std::string path);
    Table(const Id& src,std::string path);
    ~Table();
    const std::string& getType();
    double __get_input() const;
    void __set_input(double input);
    double __get_output() const;
    void __set_output(double output);
    int __get_stepMode() const;
    void __set_stepMode(int stepMode);
    double __get_stepSize() const;
    void __set_stepSize(double stepSize);
    double __get_threshold() const;
    void __set_threshold(double threshold);
    // todo: tackle these two functiosn properly
    //        double __get_tableLookup() const;
    //        void __set_tableLookup(double tableLookup);
    //         double __get_outputSrc() const;
    //         void __set_outputSrc(double outputSrc);
    //         double __get_msgInput() const;
    //         void __set_msgInput(double msgInput);
    //         double __get_sum() const;
    //         void __set_sum(double sum);
    //         double __get_prd() const;
    //         void __set_prd(double prd);

    void createTable(int xdiv, double xmin, double xmax );
    
};
}

#endif
