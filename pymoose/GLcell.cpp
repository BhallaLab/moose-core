#ifndef _pymoose_GLcell_cpp
#define _pymoose_GLcell_cpp

#include "GLcell.h"
using namespace pymoose;

const std::string GLcell::className_ = "GLcell";
GLcell::GLcell( Id id ) : PyMooseBase( id ) {}
GLcell::GLcell( std::string path ) : PyMooseBase( className_, path ) {}
GLcell::GLcell( std::string name, Id parentId ) : PyMooseBase( className_, name, parentId ) {}
GLcell::GLcell( std::string name, PyMooseBase& parent ) : PyMooseBase( className_, name, parent ) {}
GLcell::GLcell( const GLcell& src, std::string name, PyMooseBase& parent ) : PyMooseBase( src, name, parent ) {}
GLcell::GLcell( const GLcell& src, std::string name, Id& parent ) : PyMooseBase( src, name, parent ) {}
GLcell::GLcell( const GLcell& src, std::string path ) : PyMooseBase( src, path ) {}
GLcell::GLcell( const Id& src, std::string name, Id& parent ) : PyMooseBase( src, name, parent ) {}
GLcell::~GLcell() {}
const std::string& GLcell::getType() { return className_; }

std::string GLcell::__get_path() const
{
	std::string path;
	get< std::string >( id_(), "path", path );
	return path;
}

void GLcell::__set_path( std::string path )
{
	set< std::string >( id_(), "path", path );
}

std::string GLcell::__get_clientHost() const
{
	std::string clientHost;
	get< std::string >( id_(), "clientHost", clientHost );
	return clientHost;
}

void GLcell::__set_clientHost( std::string strClientHost )
{
	set< std::string >( id_(), "clientHost", strClientHost ); 
}

std::string GLcell::__get_clientPort() const
{
	std::string clientPort;
	get< std::string >( id_(), "clientPort", clientPort );
	return clientPort;	
}

void GLcell::__set_clientPort( std::string strClientPort )
{
	set< std::string >( id_(), "clientPort", strClientPort ); 
}

std::string GLcell::__get_attributeName() const
{
	std::string attributeName;
	get< std::string >( id_(), "attributeName", attributeName );
	return attributeName;
}

void GLcell::__set_attributeName( std::string strAttributeName )
{
	set< std::string >( id_(), "attributeName", strAttributeName ); 
}

double GLcell::__get_changeThreshold() const
{
	double changeThreshold;
	get< double >( id_(), "changeThreshold", changeThreshold );
	return changeThreshold;
}

void GLcell::__set_changeThreshold( double changeThreshold )
{
	set< double >( id_(), "changeThreshold", changeThreshold );
}

double GLcell::__get_VScale() const
{
	double vScale;
	get< double >( id_(), "vScale", vScale );
	return vScale;
}

void GLcell::__set_VScale( double vScale )
{
	set< double >( id_(), "vScale", vScale );
}

std::string GLcell::__get_syncMode() const
{
	std::string syncMode;
	get< std::string >( id_(), "syncMode", syncMode );
	return syncMode;	
}

void GLcell::__set_syncMode( std::string strSyncMode )
{
	set< std::string >( id_(), "syncMode", strSyncMode );
}

std::string GLcell::__get_bgColor() const
{
	std::string bgColor;
	get< std::string >( id_(), "bgColor", bgColor );
	return bgColor;
}

void GLcell::__set_bgColor( std::string strBgColor )
{
	set< std::string >( id_(), "bgColor", strBgColor );
}

double GLcell::__get_highValue() const
{
	double highValue;
	get< double >( id_(), "highValue", highValue );
	return highValue;
}

void GLcell::__set_highValue( double highValue )
{
	set< double >( id_(), "highValue", highValue );
}

double GLcell::__get_lowValue() const
{
	double lowValue;
	get< double >( id_(), "lowValue", lowValue );
	return lowValue;
}

void GLcell::__set_lowValue( double lowValue )
{
	set< double >( id_(), "lowValue", lowValue );
}

#endif
