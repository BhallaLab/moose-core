#ifndef _pymoose_RandGenerator_cpp
#define _pymoose_RandGenerator_cpp
#include "RandGenerator.h"
using namespace pymoose;
const std::string RandGenerator::className_ = "RandGenerator";
RandGenerator::RandGenerator(Id id):Neutral(id){}
RandGenerator::RandGenerator(std::string className, std::string path):Neutral(className, path){}
RandGenerator::RandGenerator(std::string className, std::string name, Id parentId):Neutral(className, name, parentId){}
RandGenerator::RandGenerator(std::string className, std::string name, PyMooseBase& parent):Neutral(className, name, parent){}
RandGenerator::RandGenerator(const RandGenerator& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
RandGenerator::RandGenerator(const RandGenerator& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
RandGenerator::RandGenerator(const RandGenerator& src, std::string path):Neutral(src, path){}
RandGenerator::RandGenerator(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
RandGenerator::RandGenerator(const Id& src, std::string path):Neutral(src, path){}
// RandGenerator::~RandGenerator(){}
const std::string& RandGenerator::getType(){ return className_; }
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
