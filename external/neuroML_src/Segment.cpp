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
/**
 * Unsets the value of the "id" attribute of this Segment. 
 */
void Segment::unsetId( )
{
	
	id.erase();
}
/**
 * Sets the value of the "id" attribute of this Segment. 
 * 
 */
void Segment::setId( const std::string& value )
{
	
	id = value;
}
/**
 * Returns the value of the "id" attribute of this Segment.
 * 
 */
const std::string& Segment::getId() const
{
	return id;
}
/**
 * Sets the value of the "name" attribute of this Segment. 
 * 
 */
void Segment::setName( const std::string& value )
{
	
	name = value;
}
/**
 * Unsets the value of the "name" attribute of this Segment. 
 */
void Segment::unsetName( )
{
	
	name.erase();
}
/**
 * Returns the value of the "name" attribute of this Segment.
 * 
 */
const std::string& Segment::getName() const
{
	return name;
}
/**
 * Predicate returning true or false depending on whether this Segment's 
 * "id" attribute has been set. 
 * 
 */
bool Segment::isSetId () const
{
   return (id.empty() == false);
}
/**
 * Predicate returning true or false depending on whether this Segment's 
 * "name" attribute has been set. 
 * 
 */
bool Segment::isSetName () const
{
   return (name.empty() == false);
}
/**
 * Sets the value of the "cable" attribute of this Segment. 
 * 
 */
void Segment::setCable( const std::string& value )
{
	cable = value;
}
/**
 * Returns the value of the "cable" attribute of this Segment.
 * 
 */
const std::string& Segment::getCable() const
{
	return cable;
}
/**
 * Sets the value of the "parent" attribute of this Segment. 
 * 
 */
void Segment::setParent( const std::string& value )
{
	parent = value;
	setparent = true;
}
/**
 * Returns the value of the "parent" attribute of this Segment.
 * 
 */
const std::string& Segment::getParent() const
{
	return parent;
}
/**
 * Predicate returning true or false depending on whether this Segment's 
 * "parent" attribute has been set. 
 * 
 */
bool Segment::isSetParent() const
{
	return setparent;
}
/**
 * Predicate returning true or false depending on whether this Segment's 
 * "proximal" attribute has been set. 
 * 
 */
bool Segment::isSetProximal() const
{
	return setproximal;
}
/**
 * Sets the value of the "x" attribute of this Segment. 
 * 
 */
void NPoint::setX( double value )
{
	x = value;
}
/**
 * Returns the value of the "x" attribute of this Segment.
 * 
 */
double NPoint::getX( ) const 
{
	return x;
}
/**
 * Sets the value of the "y" attribute of this Segment. 
 * 
 */
void NPoint::setY( double value )
{
	y = value;
}
/**
 * Returns the value of the "y" attribute of this Segment.
 * 
 */
double NPoint::getY( ) const 
{
	return y;
}
/**
 * Sets the value of the "z" attribute of this Segment. 
 * 
 */
void NPoint::setZ( double value )
{
	z = value;
}
/**
 * Returns the value of the "z" attribute of this Segment.
 * 
 */
double NPoint::getZ( ) const 
{
	return z;
}
/**
 * Sets the value of the "diameter" attribute of this Segment. 
 * 
 */
void NPoint::setDiameter( double value )
{
	diameter = value;
}
/**
 * Returns the value of the "diameter" attribute of this Segment.
 * 
 */
double NPoint::getDiameter( ) const 
{
	return diameter;
}


