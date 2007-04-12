/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include <fstream>
#include "moose.h"
#include "Shell.h"
#include "ReadCell.h"
#include "../element/Neutral.h"

ReadCell::ReadCell()
		: RM( 0.0 ), CM( 0.0 ), RA( 0.0 ), EREST_ACT( 0.0 ),
		dendrDiam( 0.0 ), aveLength( 0.0 ),
		spineSurf( 0.0 ), spineDens( 0.0 ),
		spineFreq( 0.0 ), membFactor( 0.0 ),
		numCompartments_( 0 ), numChannels_( 0 ), numOthers_( 0 ),
		cell_( 0 ), lastCompt_( 0 ),
		polarFlag_( 0 )
{
		;
}

/**
 * The readcell function implements the old GENESIS cellreader
 * functionality. Although it is really a parser operation, I
 * put it here in biophysics because the cell format is indpendent
 * of parser and is likely to remain a legacy for a while.
 */

Element* ReadCell::start( const string& cellpath )
{
	// Warning: here is a parser dependence.
	unsigned int cellId = Shell::path2eid( cellpath, "/" );
	// There should not be an existing object of this name.
	// In the old GENESIS it deleted it. Here we will complain
	
	if ( cellId != BAD_ID ) {
		cout << "Warning: cell '" << cellpath << "' already exists.\n";
		return 0;
	}

	string::size_type pos = cellpath.find_last_of( "/" );
	Element* cellpa;
	if ( pos == string::npos ) {
		cellpa = Element::root(); // actually should be cwe
	} else if ( pos == 0 ) {
		cellpa = Element::root();
	} else {
		cellId = Shell::path2eid( cellpath.substr( 0, pos - 1 ), "/" );
		if ( cellId == BAD_ID ) {
			cout << "Warning: cell path '" << cellpath <<
					"' not found.\n";
			return 0;
		}
		cellpa = Element::element( cellId );
	}
	
	return Neutral::create( "Neutral", "cell", cellpa );
}

void ReadCell::read( const string& filename, const string& cellpath )
{
	ifstream fin( filename.c_str() );
	
	cell_ = start( cellpath );
	if ( !cell_ ) return;

	string line;
	unsigned int lineNum = 0;
	string::size_type pos;
	ParseStage parseMode = DATA;
	while ( getline( fin, line ) ) {
		lineNum++;
		if ( line.length() == 0 )
				continue;
		pos = line.find_first_not_of( "\t " );
		if ( pos == string::npos )
				continue;
		else
			line = line.substr( pos );
		if ( line == "//" )
				continue;
		if ( line == "/*" ) {
				parseMode = COMMENT;
		} else if ( line == "*/" ) {
				parseMode = DATA;
				continue;
		} else if ( line[0] == '*' ) {
				parseMode = SCRIPT;
		}

		if ( parseMode == COMMENT ) {
			pos = line.find( "*/" );
			if ( pos != string::npos ) {
				parseMode = DATA;
				if ( line.length() > pos + 2 )
					line = line.substr( pos + 2 );
			}
		}
		if ( parseMode == DATA )
				readData( line, lineNum );
		else if ( parseMode == SCRIPT ) {
				readScript( line, lineNum );
				parseMode = DATA;
		}
	}
	cout << filename << " read: " <<
			numCompartments_ << " compartments, " << 
			numChannels_ << " channels, " << 
			numOthers_ << " others\n";
}

void ReadCell::readData( const string& line, unsigned int lineNum )
{
	vector< string > argv;
	parseString( line, argv, "\t " ); 
	if ( argv.size() < 6 ) {
			cout << "Readfile: Error on line " << lineNum << endl;
			cout << "Too few arguments in line: " << argv.size() <<
					", should be > 6\n";
			return;
	}
	string name = argv[0];
	string parent = argv[1];
	double x = 1.0e-6 * atof( argv[2].c_str() );
	double y = atof( argv[3].c_str() );
	double z = atof( argv[4].c_str() );
	double d = 1.0e-6 * atof( argv[5].c_str() );
	if ( polarFlag_ ) {
		double r = x;
		double theta = y * PI / 180.0;
		double phi = z * PI / 180.0;
		x = r * sin( phi ) * cos ( theta );
		y = r * sin( phi ) * sin ( theta );
		z = r * cos( phi );
	} else {
		y *= 1.0e-6;
		z *= 1.0e-6;
	}

	buildCompartment( name, parent, x, y, z, d, argv );
}

void ReadCell::buildCompartment( 
				const string& name, const string& parent,
				double x, double y, double z, double d,
				vector< string >& argv )
{
	static const Finfo* axial = 
			Cinfo::find( "Compartment" )->findFinfo( "axial" );
	static const Finfo* raxial = 
			Cinfo::find( "Compartment" )->findFinfo( "raxial" );
	Element* pa;
	if ( parent == "." ) { // Shorthand: use the previous compartment.
			pa = lastCompt_;
	} else if ( parent == "none" || parent == "nil" ) {
			pa = Element::root();
	} else {
		unsigned int paId;
		bool ret = lookupGet< unsigned int, string >(
					cell_, "lookupChild", paId, parent );
		assert( ret );
		if ( paId == BAD_ID ) {
			cout << "Error: ReadCell: could not find parent compt '" <<
					parent << "' for child '" << name << "'\n";
			return;
		}
		pa = Element::element( paId );
	}
	if ( pa == 0 )
		return;
	unsigned int childId;
	bool ret = lookupGet< unsigned int, string >(
				cell_, "lookupChild", childId, name );
	assert( ret );
	if ( childId != BAD_ID ) {
		cout << "Error: ReadCell: duplicate child on parent compt '" <<
				parent << "' for child '" << name << "'\n";
		return;
	}

	Element* compt = Neutral::create( "Compartment", name, cell_ );
	++numCompartments_;
	lastCompt_ = compt;

	if ( pa != Element::root() )
		axial->add( pa, compt, raxial );
}

void ReadCell::readScript( const string& line, unsigned int lineNum )
{
}
