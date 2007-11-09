#ifndef _pymoose_BinomialRng_cpp
#define _pymoose_BinomialRng_cpp
#include "BinomialRng.h"
const std::string BinomialRng::className = "BinomialRng";
BinomialRng::BinomialRng(Id id):RandGenerator(id){}
BinomialRng::BinomialRng(std::string path):RandGenerator(className, path){}
BinomialRng::BinomialRng(std::string name, Id parentId):RandGenerator(className, name, parentId){}
BinomialRng::BinomialRng(std::string name, PyMooseBase* parent):RandGenerator(className, name, parent){}
BinomialRng::~BinomialRng(){}
const std::string& BinomialRng::getType(){ return className; }
int BinomialRng::__get_n() const
{
    int n;
    get < int > (id_(), "n",n);
    return n;
}
void BinomialRng::__set_n( int n )
{
    set < int > (id_(), "n", n);
}
double BinomialRng::__get_p() const
{
    double p;
    get < double > (id_(), "p",p);
    return p;
}
void BinomialRng::__set_p( double p )
{
    set < double > (id_(), "p", p);
}
#endif
