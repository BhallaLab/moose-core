#ifndef _pymoose_AscFile_cpp
#define _pymoose_AscFile_cpp
#include "AscFile.h"
using namespace pymoose;
const std::string AscFile::className_ = "AscFile";
AscFile::AscFile(Id id):Neutral(id){}
AscFile::AscFile(std::string path):Neutral(className_, path){}
AscFile::AscFile(std::string name, Id parentId):Neutral(className_, name, parentId){}
AscFile::AscFile(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
AscFile::AscFile(const AscFile& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
AscFile::AscFile(const AscFile& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
AscFile::AscFile(const AscFile& src, std::string path):Neutral(src, path){}
AscFile::AscFile(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
AscFile::AscFile(const Id& src, std::string path):Neutral(src, path){}
AscFile::~AscFile(){}
const std::string& AscFile::getType(){ return className_; }
const string&  AscFile::__get_filename() const
{
return this->getField("filename");
}
void AscFile::__set_filename( string filename )
{
    set < string > (id_(), "filename", filename);
}
int AscFile::__get_append() const
{
    int append;
    get < int > (id_(), "append",append);
    return append;
}
void AscFile::__set_append( int append )
{
    set < int > (id_(), "append", append);
}
int AscFile::__get_time() const
{
    int time;
    get < int > (id_(), "time",time);
    return time;
}
void AscFile::__set_time( int time )
{
    set < int > (id_(), "time", time);
}
int AscFile::__get_header() const
{
    int header;
    get < int > (id_(), "header",header);
    return header;
}
void AscFile::__set_header( int header )
{
    set < int > (id_(), "header", header);
}
const string&  AscFile::__get_comment() const
{
return this->getField("comment");
}
void AscFile::__set_comment( string comment )
{
    set < string > (id_(), "comment", comment);
}
const string&  AscFile::__get_delimiter() const
{
return this->getField("delimiter");
}
void AscFile::__set_delimiter( string delimiter )
{
    set < string > (id_(), "delimiter", delimiter);
}
#endif
