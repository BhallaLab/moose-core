#include <vector>
#include <string>
#include <iostream>
#include "NCable.h"
using namespace std;
/**
 * Unsets the value of the "id" attribute of this NCable. 
 */
void NCable::unsetId( )
{
	id.erase();
}
/**
 * Sets the value of the "id" attribute of this NCable. 
 * 
 */
void NCable::setId( const std::string& value )
{
	id = value;
}
/**
 * Returns the value of the "id" attribute of this NCable. 
 * 
 */
const std::string& NCable::getId() const
{       
	return id;
}
/**
 * Sets the value of the "name" attribute of this NCable. 
 * 
 */
void NCable::setName( const std::string& value )
{
	
	name = value;
}
/**
 * Unsets the value of the "name" attribute of this NCable. 
 */
void NCable::unsetName( )
{
	
	name.erase();
}
/**
 * Returns the value of the "name" attribute of this NCable. 
 * 
 */
const std::string& NCable::getName() const
{
	return name;
}
/**
 * Predicate returning true or false depending on whether this NCable's 
 * "id" attribute has been set. 
 * 
 */
bool NCable::isSetId () const
{
   return (id.empty() == false);
}
/**
 * Predicate returning true or false depending on whether this NCable's 
 * "name" attribute has been set. 
 * 
 */
bool NCable::isSetName () const
{
   return (name.empty() == false);
}
/**
 * Returns the value of the "groups" attribute of this NCable. 
 * 
 */
vector < string > NCable::getGroups()const
{
	return groups_;	
}
/**
 * Sets the value of the "groups" attribute of this NCable. 
 * 
 */
void NCable::setGroups( string group )
{
	groups_.push_back( group );
        
}
/**
 * Unsets the value of the "groups" attribute of this NCable. 
 */
void NCable::unsetGroups()
{
	groups_.clear();
}

