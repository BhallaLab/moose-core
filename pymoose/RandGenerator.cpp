#ifndef _pymoose_RandGenerator_cpp
#define _pymoose_RandGenerator_cpp
#include "RandGenerator.h"
using namespace pymoose;
const std::string RandGenerator::className = "RandGenerator";
RandGenerator::RandGenerator(Id id):PyMooseBase(id){}
RandGenerator::RandGenerator(string className, std::string path):PyMooseBase(className, path){}
RandGenerator::RandGenerator(string className, std::string name, Id parentId):PyMooseBase(className, name, parentId){}
RandGenerator::RandGenerator(string className, std::string name, PyMooseBase& parent):PyMooseBase(className, name, parent){}
RandGenerator::RandGenerator(const RandGenerator& src, std::string objectName,  PyMooseBase& parent):PyMooseBase(src, objectName, parent){}

RandGenerator::RandGenerator(const RandGenerator& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
RandGenerator::RandGenerator(const RandGenerator& src, std::string path):PyMooseBase(src, path)
{
}

RandGenerator::RandGenerator(const Id& src, string name, Id& parent):PyMooseBase(src, name, parent)
{
}
//RandGenerator::~RandGenerator(){}
const std::string& RandGenerator::getType(){ return className; }
double RandGenerator::__get_sample() const
{
    double sample;
    get < double > (id_(), "sample",sample);
    return sample;
}
void RandGenerator::__set_sample( double sample )
{
    set < double > (id_(), "sample", sample);
}
double RandGenerator::__get_mean() const
{
    double mean;
    get < double > (id_(), "mean",mean);
    return mean;
}
void RandGenerator::__set_mean( double mean )
{
    set < double > (id_(), "mean", mean);
}
double RandGenerator::__get_variance() const
{
    double variance;
    get < double > (id_(), "variance",variance);
    return variance;
}
void RandGenerator::__set_variance( double variance )
{
    set < double > (id_(), "variance", variance);
}
double RandGenerator::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
void RandGenerator::__set_output( double output )
{
    set < double > (id_(), "output", output);
}
#endif
