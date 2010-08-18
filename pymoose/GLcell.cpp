#ifndef _pymoose_GLcell_cpp
#define _pymoose_GLcell_cpp

#include "GLcell.h"
using namespace pymoose;

const std::string GLcell::className_ = "GLcell";
GLcell::GLcell( Id id ) :Neutral( id ) {}
GLcell::GLcell( std::string path ) :Neutral( className_, path ) {}
GLcell::GLcell( std::string name, Id parentId ) :Neutral( className_, name, parentId ) {}
GLcell::GLcell( std::string name, PyMooseBase& parent ) :Neutral( className_, name, parent ) {}
GLcell::GLcell( const GLcell& src, std::string name, PyMooseBase& parent ) :Neutral( src, name, parent ) {}
GLcell::GLcell( const GLcell& src, std::string name, Id& parent ) :Neutral( src, name, parent ) {}
GLcell::GLcell( const GLcell& src, std::string path ) :Neutral( src, path ) {}
GLcell::GLcell( const Id& src, std::string path ) :Neutral( src, path ) {}
GLcell::GLcell( const Id& src, std::string name, Id& parent ) :Neutral( src, name, parent ) {}
GLcell::~GLcell() {}
const std::string& GLcell::getType() { return className_; }

const std::string GLcell::__get_vizpath() const
{
    return this->getField("vizpath");
}

void GLcell::__set_vizpath( const std::string vizpath )
{
	set< std::string >( id_(), "vizpath", vizpath );
}

const std::string& GLcell::__get_clientHost() const
{
    return getField("host");
}

void GLcell::__set_clientHost( const std::string strClientHost )
{
	set< std::string >( id_(), "host", strClientHost ); 
}

const std::string& GLcell::__get_clientPort() const
{
    return this->getField("port");
}

void GLcell::__set_clientPort( const std::string strClientPort )
{
	set< std::string >( id_(), "port", strClientPort ); 
}

const std::string& GLcell::__get_attributeName() const
{
    return getField("attribute");
}

void GLcell::__set_attributeName( const std::string strAttributeName )
{
	set< std::string >( id_(), "attribute", strAttributeName ); 
}

double GLcell::__get_changeThreshold() const
{
	double changeThreshold;
	get< double >( id_(), "threshold", changeThreshold );
	return changeThreshold;
}

void GLcell::__set_changeThreshold( double changeThreshold )
{
	set< double >( id_(), "threshold", changeThreshold );
}

double GLcell::__get_VScale() const
{
	double vScale;
	get< double >( id_(), "vscale", vScale );
	return vScale;
}

void GLcell::__set_VScale( double vScale )
{
	set< double >( id_(), "vscale", vScale );
}

const std::string& GLcell::__get_syncMode() const
{
    return this->getField("sync");	
}

void GLcell::__set_syncMode( const std::string& strSyncMode )
{
	set< std::string >( id_(), "sync", strSyncMode );
}

const std::string& GLcell::__get_bgColor() const
{
	return this->getField("bgcolor" );
}

void GLcell::__set_bgColor( const std::string strBgColor )
{
	set< std::string >( id_(), "bgcolor", strBgColor );
}

double GLcell::__get_highValue() const
{
	double highValue;
	get< double >( id_(), "highvalue", highValue );
	return highValue;
}

void GLcell::__set_highValue( double highValue )
{
	set< double >( id_(), "highvalue", highValue );
}

double GLcell::__get_lowValue() const
{
	double lowValue;
	get< double >( id_(), "lowvalue", lowValue );
	return lowValue;
}

void GLcell::__set_lowValue( double lowValue )
{
	set< double >( id_(), "lowvalue", lowValue );
}

#endif
