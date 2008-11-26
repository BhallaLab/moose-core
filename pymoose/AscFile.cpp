#ifndef _pymoose_AscFile_cpp
#define _pymoose_AscFile_cpp
#include "AscFile.h"
using namespace pymoose;
const std::string AscFile::className_ = "AscFile";
AscFile::AscFile(Id id):PyMooseBase(id){}
AscFile::AscFile(std::string path):PyMooseBase(className_, path){}
AscFile::AscFile(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
AscFile::AscFile(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
AscFile::AscFile(const AscFile& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
AscFile::AscFile(const AscFile& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
AscFile::AscFile(const AscFile& src, std::string path):PyMooseBase(src, path){}
AscFile::AscFile(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
AscFile::~AscFile(){}
const std::string& AscFile::getType(){ return className_; }
string AscFile::__get_fileName() const
{
    string fileName;
    get < string > (id_(), "fileName",fileName);
    return fileName;
}
void AscFile::__set_fileName( string fileName )
{
    set < string > (id_(), "fileName", fileName);
}
int AscFile::__get_appendFlag() const
{
    int appendFlag;
    get < int > (id_(), "appendFlag",appendFlag);
    return appendFlag;
}
void AscFile::__set_appendFlag( int appendFlag )
{
    set < int > (id_(), "appendFlag", appendFlag);
}
#endif
