#include <string.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/xmlreader.h>
#include <iostream>
#include <cassert>
#include "Segment.h"
using namespace std;

void Segment::unsetId( )
{
	
	id.erase();
}
void Segment::setId( const std::string& value )
{
	
	id = value;
}

const std::string& Segment::getId() const
{
	return id;
}
void Segment::setName( const std::string& value )
{
	
	name = value;
}
void Segment::unsetName( )
{
	
	name.erase();
}
const std::string& Segment::getName() const
{
	return name;
}
bool Segment::isSetId () const
{
   return (id.empty() == false);
}
bool Segment::isSetName () const
{
   return (name.empty() == false);
}
void Segment::setCable( const std::string& value )
{
	cable = value;
}
const std::string& Segment::getCable() const
{
	return cable;
}
void Segment::setParent( const std::string& value )
{
	parent = value;
	setparent = true;
}
const std::string& Segment::getParent() const
{
	return parent;
}
bool Segment::isSetParent() const
{
	return setparent;
}
bool Segment::isSetProximal() const
{
	return setproximal;
}
void Point::setX( double value )
{
	x = value;
}
double Point::getX( ) const 
{
	return x;
}
void Point::setY( double value )
{
	y = value;
}
double Point::getY( ) const 
{
	return y;
}
void Point::setZ( double value )
{
	z = value;
}
double Point::getZ( ) const 
{
	return z;
}
void Point::setDiameter( double value )
{
	diameter = value;
}
double Point::getDiameter( ) const 
{
	return diameter;
}


