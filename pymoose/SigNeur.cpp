#ifndef _pymoose_SigNeur_cpp
#define _pymoose_SigNeur_cpp
#include "SigNeur.h"
using namespace pymoose;
const std::string SigNeur::className_ = "SigNeur";
SigNeur::SigNeur(Id id):Neutral(id){}
SigNeur::SigNeur(std::string path):Neutral(className_, path){}
SigNeur::SigNeur(std::string name, Id parentId):Neutral(className_, name, parentId){}
SigNeur::SigNeur(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
SigNeur::SigNeur(const SigNeur& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
SigNeur::SigNeur(const SigNeur& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
SigNeur::SigNeur(const SigNeur& src, std::string path):Neutral(src, path){}
SigNeur::SigNeur(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
SigNeur::SigNeur(const Id& src, std::string path):Neutral(src, path){}
SigNeur::~SigNeur(){}
const std::string& SigNeur::getType(){ return className_; }

string SigNeur::__get_cellProto() const
{
    Id cellProto;
    get < Id > (id_(), "cellProto", cellProto);
    return cellProto.path();
}
void SigNeur::__set_cellProto( string cellProto )
{
    set < Id > (id_(), "cellProto", Id(cellProto));
}
string SigNeur::__get_spineProto() const
{
    Id spineProto;
    get < Id > (id_(), "spineProto",spineProto);
    return spineProto.path();
}
void SigNeur::__set_spineProto( string spineProto )
{
    set < Id > (id_(), "spineProto", Id(spineProto));
}
string SigNeur::__get_dendProto() const
{
    Id dendProto;
    get < Id > (id_(), "dendProto", dendProto);
    return dendProto.path();
}
void SigNeur::__set_dendProto( string dendProto )
{
    set < Id > (id_(), "dendProto", Id(dendProto));
}
string SigNeur::__get_somaProto() const
{
    Id somaProto;
    get < Id > (id_(), "somaProto",somaProto);
    return somaProto.path();
}
void SigNeur::__set_somaProto( string somaProto )
{
    Id id = Id(somaProto);
    set < Id > (id_(), "somaProto", id);
}
string SigNeur::__get_cell() const
{
    Id cell;
    get < Id > (id_(), "cell",cell);
    return cell.path();
}
string SigNeur::__get_spine() const
{
    Id spine;
    get < Id > (id_(), "spine",spine);
    return spine.path();
}
string SigNeur::__get_dend() const
{
    Id dend;
    get < Id > (id_(), "dend",dend);
    return dend.path();
}
string SigNeur::__get_soma() const
{
    Id soma;
    get < Id > (id_(), "soma",soma);
    return soma.path();
}
string SigNeur::__get_cellMethod() const
{
    string cellMethod;
    get < string > (id_(), "cellMethod",cellMethod);
    return cellMethod;
}
void SigNeur::__set_cellMethod( string cellMethod )
{
    set < string > (id_(), "cellMethod", cellMethod);
}
string SigNeur::__get_spineMethod() const
{
    string spineMethod;
    get < string > (id_(), "spineMethod",spineMethod);
    return spineMethod;
}
void SigNeur::__set_spineMethod( string spineMethod )
{
    set < string > (id_(), "spineMethod", spineMethod);
}
string SigNeur::__get_dendMethod() const
{
    string dendMethod;
    get < string > (id_(), "dendMethod",dendMethod);
    return dendMethod;
}
void SigNeur::__set_dendMethod( string dendMethod )
{
    set < string > (id_(), "dendMethod", dendMethod);
}
string SigNeur::__get_somaMethod() const
{
    string somaMethod;
    get < string > (id_(), "somaMethod",somaMethod);
    return somaMethod;
}
void SigNeur::__set_somaMethod( string somaMethod )
{
    set < string > (id_(), "somaMethod", somaMethod);
}
double SigNeur::__get_sigDt() const
{
    double sigDt;
    get < double > (id_(), "sigDt",sigDt);
    return sigDt;
}
void SigNeur::__set_sigDt( double sigDt )
{
    set < double > (id_(), "sigDt", sigDt);
}
double SigNeur::__get_cellDt() const
{
    double cellDt;
    get < double > (id_(), "cellDt",cellDt);
    return cellDt;
}
void SigNeur::__set_cellDt( double cellDt )
{
    set < double > (id_(), "cellDt", cellDt);
}
double SigNeur::__get_Dscale() const
{
    double Dscale;
    get < double > (id_(), "Dscale",Dscale);
    return Dscale;
}
void SigNeur::__set_Dscale( double Dscale )
{
    set < double > (id_(), "Dscale", Dscale);
}
double SigNeur::__get_lambda() const
{
    double lambda;
    get < double > (id_(), "lambda",lambda);
    return lambda;
}
void SigNeur::__set_lambda( double lambda )
{
    set < double > (id_(), "lambda", lambda);
}
int SigNeur::__get_parallelMode() const
{
    int parallelMode;
    get < int > (id_(), "parallelMode",parallelMode);
    return parallelMode;
}
void SigNeur::__set_parallelMode( int parallelMode )
{
    set < int > (id_(), "parallelMode", parallelMode);
}
double SigNeur::__get_updateStep() const
{
    double updateStep;
    get < double > (id_(), "updateStep",updateStep);
    return updateStep;
}
void SigNeur::__set_updateStep( double updateStep )
{
    set < double > (id_(), "updateStep", updateStep);
}
double SigNeur::__get_calciumScale() const
{
    double calciumScale;
    get < double > (id_(), "calciumScale",calciumScale);
    return calciumScale;
}
void SigNeur::__set_calciumScale( double calciumScale )
{
    set < double > (id_(), "calciumScale", calciumScale);
}
string SigNeur::__get_dendInclude() const
{
    string dendInclude;
    get < string > (id_(), "dendInclude",dendInclude);
    return dendInclude;
}
void SigNeur::__set_dendInclude( string dendInclude )
{
    set < string > (id_(), "dendInclude", dendInclude);
}
string SigNeur::__get_dendExclude() const
{
    string dendExclude;
    get < string > (id_(), "dendExclude",dendExclude);
    return dendExclude;
}
void SigNeur::__set_dendExclude( string dendExclude )
{
    set < string > (id_(), "dendExclude", dendExclude);
}

void SigNeur::build()
{
    set(id_(), "build");
}
#endif
