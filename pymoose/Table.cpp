#ifndef _pymoose_Table_cpp
#define _pymoose_Table_cpp
#include "Table.h"
using namespace pymoose;
const std::string Table::className = "Table";
Table::Table(Id id):InterpolationTable(id){}
Table::Table(std::string path):InterpolationTable(className, path){}
Table::Table(std::string name, Id parentId):InterpolationTable(className, name, parentId){}
Table::Table(std::string name, PyMooseBase& parent):InterpolationTable(className, name, parent){}
Table::~Table(){}
const std::string& Table::getType(){ return className; }
double Table::__get_input() const
{
    double input;
    get < double > (id_(), "input",input);
    return input;
}
void Table::__set_input( double input )
{
    set < double > (id_(), "input", input);
}
double Table::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
void Table::__set_output( double output )
{
    set < double > (id_(), "output", output);
}
int Table::__get_step_mode() const
{
    int step_mode;
    get < int > (id_(), "step_mode",step_mode);
    return step_mode;
}
void Table::__set_step_mode( int step_mode )
{
    set < int > (id_(), "step_mode", step_mode);
}
int Table::__get_stepmode() const
{
    int stepmode;
    get < int > (id_(), "stepmode",stepmode);
    return stepmode;
}
void Table::__set_stepmode( int stepmode )
{
    set < int > (id_(), "stepmode", stepmode);
}
double Table::__get_stepsize() const
{
    double stepsize;
    get < double > (id_(), "stepsize",stepsize);
    return stepsize;
}
void Table::__set_stepsize( double stepsize )
{
    set < double > (id_(), "stepsize", stepsize);
}
double Table::__get_threshold() const
{
    double threshold;
    get < double > (id_(), "threshold",threshold);
    return threshold;
}
void Table::__set_threshold( double threshold )
{
    set < double > (id_(), "threshold", threshold);
}
// todo: tackle the following two later
// double Table::__get_tableLookup(unsigned int index) const
// {
//     double entry;
//     get < double , unsigned int> (id_(), "tableLookup", entry, index);
//     return entry;
// }
// void Table::__set_tableLookup( double tableLookup, unsigned int index )
// {
//     set < double > (id_(), "tableLookup", tableLookup, index);
// }
double Table::__get_outputSrc() const
{
    double outputSrc;
    get < double > (id_(), "outputSrc",outputSrc);
    return outputSrc;
}
void Table::__set_outputSrc( double outputSrc )
{
    set < double > (id_(), "outputSrc", outputSrc);
}
double Table::__get_msgInput() const
{
    double msgInput;
    get < double > (id_(), "msgInput",msgInput);
    return msgInput;
}
void Table::__set_msgInput( double msgInput )
{
    set < double > (id_(), "msgInput", msgInput);
}
double Table::__get_sum() const
{
    double sum;
    get < double > (id_(), "sum",sum);
    return sum;
}
void Table::__set_sum( double sum )
{
    set < double > (id_(), "sum", sum);
}
double Table::__get_prd() const
{
    double prd;
    get < double > (id_(), "prd",prd);
    return prd;
}
void Table::__set_prd( double prd )
{
    set < double > (id_(), "prd", prd);
}
void Table::createTable( int xdivs, double xmin, double xmax)
{
    __set_xmin(xmin);
    __set_xmax(xmax);
    __set_xdivs(xdivs);    
}

#endif
