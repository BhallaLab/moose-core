#ifndef _pymoose_Adaptor_cpp
#define _pymoose_Adaptor_cpp
#include "Adaptor.h"
using namespace pymoose;
const std::string Adaptor::className_ = "Adaptor";
Adaptor::Adaptor(Id id):Neutral(id){}
Adaptor::Adaptor(std::string path):Neutral(className_, path){}
Adaptor::Adaptor(std::string name, Id parentId):Neutral(className_, name, parentId){}
Adaptor::Adaptor(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Adaptor::Adaptor(const Adaptor& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Adaptor::Adaptor(const Adaptor& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Adaptor::Adaptor(const Adaptor& src, std::string path):Neutral(src, path){}
Adaptor::Adaptor(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Adaptor::Adaptor(const Id& src, std::string path):Neutral(src, path){}
Adaptor::~Adaptor(){}
const std::string& Adaptor::getType(){ return className_; }
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
