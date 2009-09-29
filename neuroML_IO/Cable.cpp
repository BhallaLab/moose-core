#include "moose.h"
#include <vector>
#include <string>
#include <iostream>
#include "Cable.h"
using namespace std;
const Cinfo* initCableCinfo()
{
	static Finfo* cableFinfos[] = 
	{
		/*new ValueFinfo( "name", ValueFtype1< string >::global(),
			reinterpret_cast< GetFunc >( &Cable::getName ),
			reinterpret_cast< RecvFunc >( &Cable::setName )
		),*/
		new ValueFinfo( "groups", ValueFtype1< vector< string > >::global(),
			reinterpret_cast< GetFunc >( &Cable::getGroups ),
			reinterpret_cast< RecvFunc >( &Cable::setGroups )
		),
		new SrcFinfo( "compartment", Ftype0::global() ),
	};
	static string doc[] =
	{
		"Name", "Cable",
		"Author", "Siji P George",
		"Description", "Cable object.",
	};
	static Cinfo cableCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),
				initNeutralCinfo(),
				cableFinfos,
				sizeof( cableFinfos ) / sizeof( Finfo* ),
				ValueFtype1< Cable >::global()
	);

	return &cableCinfo;
}
/*void Cable::setName( const Conn* c, std::string value )
{
	
	static_cast< Cable* >( c->data() )->name_ = value;
}
std::string Cable::getName( Eref e ) 
{
	return static_cast< Cable* >( e.data() )->name_;
}*/
vector < string > Cable::getGroups( Eref e )
{
	return static_cast< Cable* >( e.data() )->groups_;	
}

void Cable::setGroups( const Conn* c,vector< string > group )
{
	static_cast< Cable* >( c->data() )->groups_ =  group;
        
}


