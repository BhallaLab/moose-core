#ifndef _pymoose_GLview_cpp
#define _pymoose_GLview_cpp

#include "GLview.h"
using namespace pymoose;

const std::string GLview::className_ = "GLview";
GLview::GLview( Id id ) :Neutral( id ) {}
GLview::GLview( std::string path ) :Neutral( className_, path ) {}
GLview::GLview( std::string name, Id parentId ) :Neutral( className_, name, parentId ) {}
GLview::GLview( std::string name, PyMooseBase& parent ) :Neutral( className_, name, parent ) {}
GLview::GLview( const GLview& src, std::string name, PyMooseBase& parent ) :Neutral( src, name, parent ) {}
GLview::GLview( const GLview& src, std::string name, Id& parent ) :Neutral( src, name, parent ) {}
GLview::GLview( const GLview& src, std::string path ) :Neutral( src, path ) {}
GLview::GLview( const Id& src, std::string path ) :Neutral( src, path ) {}
GLview::GLview( const Id& src, std::string name, Id& parent ) :Neutral( src, name, parent ) {}
GLview::~GLview() {}
const std::string& GLview::getType() { return className_; }

std::string GLview::__get_vizpath() const
{
    return getField(  "vizpath");
}

void GLview::__set_vizpath( std::string vizpath )
{
	set< std::string >( id_(), "vizpath", vizpath );
}

std::string GLview::__get_clientHost() const
{
    return getField(  "host");
}

void GLview::__set_clientHost( std::string clientHost )
{
	set< std::string >( id_(), "host", clientHost );
}

std::string GLview::__get_clientPort() const
{
	return getField(  "port");
}

void GLview::__set_clientPort( std::string clientPort )
{
	set< std::string >( id_(), "port", clientPort );
}

std::string GLview::__get_relPath() const
{
    return this->getField(  "relpath");
}

void GLview::__set_relPath( std::string relPath )
{
	set< std::string >( id_(), "relpath", relPath );
}

std::string GLview::__get_value1Field() const
{
    return this->getField( "value1");
}

void GLview::__set_value1Field( std::string value1Field )
{
	set< std::string >( id_(), "value1", value1Field );
}

double GLview::__get_value1Min() const
{
	double value1Min;
	get< double >( id_(), "value1min", value1Min );
	return value1Min;
}

void GLview::__set_value1Min( double value1Min )
{
	set< double >( id_(), "value1min", value1Min );
}

double GLview::__get_value1Max() const
{
	double value1Max;
	get< double >( id_(), "value1max", value1Max );
	return value1Max;
}

void GLview::__set_value1Max( double value1Max )
{
	set< double >( id_(), "value1max", value1Max );
}

std::string GLview::__get_value2Field() const
{
    return this->getField(  "value2");
}

void GLview::__set_value2Field( std::string value2Field )
{
	set< std::string >( id_(), "value2", value2Field );
}

double GLview::__get_value2Min() const
{
	double value2Min;
	get< double >( id_(), "value2min", value2Min );
	return value2Min;
}

void GLview::__set_value2Min( double value2Min )
{
	set< double >( id_(), "value2min", value2Min );
}

double GLview::__get_value2Max() const
{
	double value2Max;
	get< double >( id_(), "value2max", value2Max );
	return value2Max;
}

void GLview::__set_value2Max( double value2Max )
{
	set< double >( id_(), "value2max", value2Max );
}

std::string GLview::__get_value3Field() const
{
    return this->getField(  "value3");
}

void GLview::__set_value3Field( std::string value3Field )
{
	set< std::string >( id_(), "value3", value3Field );
}

double GLview::__get_value3Min() const
{
	double value3Min;
	get< double >( id_(), "value3min", value3Min );
	return value3Min;
}

void GLview::__set_value3Min( double value3Min )
{
	set< double >( id_(), "value3min", value3Min );
}

double GLview::__get_value3Max() const
{
	double value3Max;
	get< double >( id_(), "value3max", value3Max );
	return value3Max;
}

void GLview::__set_value3Max( double value3Max )
{
	set< double >( id_(), "value3max", value3Max );
}

std::string GLview::__get_value4Field() const
{
	return this->getField("value4"  );
}

void GLview::__set_value4Field( std::string value4Field )
{
	set< std::string >( id_(), "value4", value4Field );
}

double GLview::__get_value4Min() const
{
	double value4Min;
	get< double >( id_(), "value4min", value4Min );
	return value4Min;
}

void GLview::__set_value4Min( double value4Min )
{
	set< double >( id_(), "value4min", value4Min );
}

double GLview::__get_value4Max() const
{
	double value4Max;
	get< double >( id_(), "value4max", value4Max );
	return value4Max;
}

void GLview::__set_value4Max( double value4Max )
{
set< double >( id_(), "value4max", value4Max );
}

std::string GLview::__get_value5Field() const
{
    return this->getField(  "value5");
}

void GLview::__set_value5Field( std::string value5Field )
{
	set< std::string >( id_(), "value5", value5Field );
}

double GLview::__get_value5Min() const
{
	double value5Min;
	get< double >( id_(), "value5min", value5Min );
	return value5Min;
}

void GLview::__set_value5Min( double value5Min )
{
	set< double >( id_(), "value5min", value5Min );
}

double GLview::__get_value5Max() const
{
	double value5Max;
	get< double >( id_(), "value5max", value5Max );
	return value5Max;
}

void GLview::__set_value5Max( double value5Max )
{
	set< double >( id_(), "value5max", value5Max );
}

std::string GLview::__get_bgColor() const
{
    return this->getField(  "bgcolor");
}

void GLview::__set_bgColor( std::string bgColor )
{
	set< std::string >( id_(), "bgcolor", bgColor );
}

std::string GLview::__get_syncMode() const
{
    return this->getField(  "sync");
}

void GLview::__set_syncMode( std::string syncMode )
{
	set< std::string >( id_(), "sync", syncMode );
}

std::string GLview::__get_gridMode() const
{
	return this->getField(  "grid");
}

void GLview::__set_gridMode( std::string gridMode )
{
	set< std::string >( id_(), "grid", gridMode );
}

unsigned int GLview::__get_colorVal() const
{
	unsigned int colorVal;
	get< unsigned int >( id_(), "color_val", colorVal );
	return colorVal;
}

void GLview::__set_colorVal( unsigned int colorVal )
{
	set< unsigned int >( id_(), "color_val", colorVal );
}

unsigned int GLview::__get_morphVal() const
{
	unsigned int morphVal;
	get< unsigned int >( id_(), "morph_val", morphVal );
	return morphVal;
}

void GLview::__set_morphVal( unsigned int morphVal )
{
	set< unsigned int >( id_(), "morph_val", morphVal );
}

unsigned int GLview::__get_xoffsetVal() const
{
	unsigned int xoffsetVal;
	get< unsigned int >( id_(), "xoffset_val", xoffsetVal );
	return xoffsetVal;
}

void GLview::__set_xoffsetVal( unsigned int xoffsetVal )
{
	set< unsigned int >( id_(), "xoffset_val", xoffsetVal );
}

unsigned int GLview::__get_yoffsetVal() const
{
	unsigned int yoffsetVal;
	get< unsigned int >( id_(), "yoffset_val", yoffsetVal );
	return yoffsetVal;
}

void GLview::__set_yoffsetVal( unsigned int yoffsetVal )
{
	set< unsigned int >( id_(), "yoffset_val", yoffsetVal );
}

unsigned int GLview::__get_zoffsetVal() const
{
	unsigned int zoffsetVal;
	get< unsigned int >( id_(), "zoffset_val", zoffsetVal );
	return zoffsetVal;
}

void GLview::__set_zoffsetVal( unsigned int zoffsetVal )
{
	set< unsigned int >( id_(), "zoffset_val", zoffsetVal );
}

#endif
