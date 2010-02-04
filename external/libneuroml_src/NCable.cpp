#include <vector>
#include <string>
#include <iostream>
#include "NCable.h"
using namespace std;
void NCable::unsetId( )
{
	id.erase();
}
void NCable::setId( const std::string& value )
{
	id = value;
}

const std::string& NCable::getId() const
{       
	return id;
}
void NCable::setName( const std::string& value )
{
	
	name = value;
}
void NCable::unsetName( )
{
	
	name.erase();
}
const std::string& NCable::getName() const
{
	return name;
}
bool NCable::isSetId () const
{
   return (id.empty() == false);
}
bool NCable::isSetName () const
{
   return (name.empty() == false);
}

vector < string > NCable::getGroups()const
{
	return groups_;	
}

void NCable::setGroups( string group )
{
	groups_.push_back( group );
        
}
void NCable::unsetGroups()
{
	groups_.clear();
}

