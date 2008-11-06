#ifndef _pymoose_Adaptor_cpp
#define _pymoose_Adaptor_cpp
#include "Adaptor.h"
using namespace pymoose;
const std::string Adaptor::className = "Adaptor";
Adaptor::Adaptor(Id id):PyMooseBase(id){}
Adaptor::Adaptor(std::string path):PyMooseBase(className, path){}
Adaptor::Adaptor(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Adaptor::Adaptor(std::string name, PyMooseBase& parent):PyMooseBase(className, name, parent){}
Adaptor::Adaptor(const Adaptor& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
Adaptor::Adaptor(const Adaptor& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
Adaptor::Adaptor(const Adaptor& src, std::string path):PyMooseBase(src, path){}
Adaptor::Adaptor(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
Adaptor::~Adaptor(){}
const std::string& Adaptor::getType(){ return className; }
double Adaptor::__get_inputOffset() const
{
    double inputOffset;
    get < double > (id_(), "inputOffset",inputOffset);
    return inputOffset;
}
void Adaptor::__set_inputOffset( double inputOffset )
{
    set < double > (id_(), "inputOffset", inputOffset);
}
double Adaptor::__get_outputOffset() const
{
    double outputOffset;
    get < double > (id_(), "outputOffset",outputOffset);
    return outputOffset;
}
void Adaptor::__set_outputOffset( double outputOffset )
{
    set < double > (id_(), "outputOffset", outputOffset);
}
double Adaptor::__get_scale() const
{
    double scale;
    get < double > (id_(), "scale",scale);
    return scale;
}
void Adaptor::__set_scale( double scale )
{
    set < double > (id_(), "scale", scale);
}
double Adaptor::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
#endif
