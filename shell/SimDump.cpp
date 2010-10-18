/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include "Shell.h"
#include "SimDump.h"

//////////////////////////////////////////////////////////////////
// SimDumpInfo functions
//////////////////////////////////////////////////////////////////

SimDumpInfo::SimDumpInfo(
	const string& oldObject, const string& newObject,
			const string& oldFields, const string& newFields)
			: oldObject_( oldObject ), newObject_( newObject )
{
	vector< string > oldList;
	vector< string > newList;

	separateString( oldFields, oldList, " " );
	separateString( newFields, newList, " " );

	if ( oldList.size() != newList.size() ) {
		cout << "Error: SimDumpInfo::SimDumpInfo: field list length diffs:\n" << oldFields << "\n" << newFields << "\n";
		return;
	}
	for ( unsigned int i = 0; i < oldList.size(); i++ )
		fields_[ oldList[ i ] ] = newList[ i ];
}

// Takes info from simobjdump
void SimDumpInfo::setFieldSequence( vector< string >& argv )
{
	string blank = "";
	fieldSequence_.resize( 0 );
	assert( argv.size() > 2 );
	vector< string >::iterator i;
	for ( i = argv.begin() + 2; i != argv.end(); i++ )
	{
		map< string, string >::iterator j = fields_.find( *i );
		if ( j != fields_.end() )
			fieldSequence_.push_back( j->second );
		else 
			fieldSequence_.push_back( blank );
	}
}

bool SimDumpInfo::setFields( Element* e, vector< string >::iterator begin,
	vector< string >::iterator end)
{
	if ( end == begin )
		return 0;
	assert( end >= begin );
	unsigned long size = static_cast< unsigned long >( end - begin );
	if ( size != fieldSequence_.size() ) {
		cout << "Error: SimDumpInfo::setFields('" << e->name() << 
			"'):: Number of argument mismatch\n";
		return 0;
	}
	vector< string >::iterator j;
	unsigned int i = 0;
	for ( j = begin; j != end; j++ ) {
		if ( fieldSequence_[ i ].length() > 0 )
		{
			const Finfo* f = e->findFinfo( fieldSequence_[ i ] );
			if ( !f ) {
				cout << "Error: SimDumpInfo::setFields:: Failed to find field '" << fieldSequence_[i] << "'\n";
				return 0;
			}
			
			bool ret = f->strSet( e, *j );
			if ( ret == 0 )
			{
				cout << "Error: SimDumpInfo::setFields:: Failed to set '";
				cout << e->name() << "/" << 
					fieldSequence_[ i ] << " to " << *j << "'\n";
				return 0;
			}
		}
		i++;
	}
	return 1;
}

//////////////////////////////////////////////////////////////////
// SimDump functions
//////////////////////////////////////////////////////////////////

SimDump::SimDump()
{
	string className = "molecule";
	vector< SimDumpInfo *> sid;
	// Here we initialize some simdump conversions. Several things here
	// are for backward compatibility. Need to think about how to
	// eventually evolve out of these. Perhaps SBML.
	sid.push_back( new SimDumpInfo(
		"kpool", "Molecule", 
		"DiffConst n nInit vol slave_enable x y xtree_textfg_req xtree_fg_req", 
		"D n nInit volumeScale slave_enable x y xtree_textfg_req xtree_fg_req") );
	sid.push_back( new SimDumpInfo(
		"kreac", "Reaction", "kf kb x y xtree_textfg_req xtree_fg_req", "kf kb x y xtree_textfg_req xtree_fg_req") );
	sid.push_back( new SimDumpInfo( "kenz", "Enzyme",
		"k1 k2 k3 usecomplex nComplexInit x y xtree_textfg_req xtree_fg_req",
		"k1 k2 k3 mode nInitComplex x y xtree_textfg_req xtree_fg_req") );
	sid.push_back( new SimDumpInfo( "xtab", "Table",
	"input output step_mode stepsize",
	"input output stepmode stepsize" ) );

	sid.push_back( new SimDumpInfo( "group", "KinCompt", "x y", "x y" ) );
	sid.push_back( new SimDumpInfo( "xgraph", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xplot", "Table", "", "" ) );

	sid.push_back( new SimDumpInfo( "geometry", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xcoredraw", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xtree", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "xtext", "Neutral", "", "" ) );
	sid.push_back( new SimDumpInfo( "text", "Neutral", "", "" ) );

	sid.push_back( new SimDumpInfo( "kchan", "ConcChan",
		"perm Vm",
		"permeability Vm" ) );

	for (unsigned int i = 0 ; i < sid.size(); i++ ) {
		dumpConverter_[ sid[ i ]->oldObject() ] = sid[ i ];
	}
}

/**
 * Stores a list of fields to be used when dumping or undumping an object.
 * First argument is the function call.
 */
void SimDump::simObjDump( const string& args )
{
	vector< string > argv;
	separateString( args, argv, " " );

	if ( argv.size() < 3 )
		return;
	string name = argv[ 1 ];
	map< string, SimDumpInfo* >::iterator i =
		dumpConverter_.find( name );
	if ( i != dumpConverter_.end() ) {
		i->second->setFieldSequence( argv );
	}
}

void SimDump::simUndump( const string& args )
{
	vector< string > argv;
	separateString( args, argv, " " );
	


	// use a map to associate class with sequence of fields, 
	// as set up in default and also with simobjdump
	if (argv.size() < 4 ) {
		cout << string("usage: ") + argv[ 0 ] +
			" class path clock [fields...]\n";
		return;
	}
	string oldClassName = argv[ 1 ];
	string path = argv[ 2 ];
	map< string, SimDumpInfo*  >::iterator i;

	i = dumpConverter_.find( oldClassName );
	if ( i == dumpConverter_.end() ) {
		cout << string("simundumpFunc: old class name '") + 
			oldClassName + "' not entered into simobjdump\n";
		return;
	}

	// Case 1: The element already exists.
	Id id( path, "/" );
	Element* e = 0;
	if ( !id.bad() ) {
		e = id();
		assert( e != 0 );
		i->second->setFields( e, argv.begin() + 4, argv.end() );
		return;
	}

	// Case 2: Element does not exist but parent element does.
	string ppath = Shell::head( path, "/" );
	string epath = Shell::tail( path, "/" );
	id = Id( ppath, "/" );
	if ( id.bad() ) {
		cout << "simundumpFunc: bad parent path: " << epath << endl;
		return;
	}
	Element* parent = id();
	assert( parent != 0 );

	string newClassName = i->second->newObject();

	e = Neutral::create( newClassName, epath, id, Id::scratchId() );
	if ( !e ) {
		cout << "Error: simundumpFunc: Failed to create " << newClassName <<
			" " << epath << endl;
			return;
	}
	i->second->setFields( e, argv.begin() + 4, argv.end() );
}

/**
 * This should give a standalone, parser-independent function for
 * reading in kkit simdump files
 */
bool SimDump::read( const string& filename )
{
	cout << " in SimDump::read( const string& filename )\n";
	return 0;
}

/**
 * This should give a standalone, parser-independent function for
 * writing out kkit simdump files
 */
bool SimDump::write( const string& filename, const string& path )
{
	cout << "in SimDump::write( const string& filename, const string& path )\n";
	return 0;
}


