/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>

using namespace std;
typedef int Id; // dummy
#include "ReadKkit.h"


unsigned int chopLine( string line, vector< string >& ret )
{
	ret.resize( 0 );
	stringstream ss( line );
	string arg;
	while ( ss >> arg ) {
		ret.push_back( arg );
	}
	return ret.size();
}

int main( int argc, const char* argv[] )
{
	string fname = "dend_v26.g";
	if ( argc == 2 )
		fname = argv[1];
	ReadKkit rk;
	rk.read( fname, "cell", 0 );
}

////////////////////////////////////////////////////////////////////////

ReadKkit::ReadKkit()
	:
	maxtime_( 1.0 ),
	simdt_( 0.01 ),
	fastdt_( 0.001 ),
	numCompartments_( 0 ),
	numMols_( 0 ),
	numReacs_( 0 ),
	numEnz_( 0 ),
	numPlot_( 0 ),
	numOthers_( 0 )
{
	;
}

void ReadKkit::innerRead( ifstream& fin )
{
	string line;
	lineNum_ = 0;
	string::size_type pos;
	ParseMode parseMode = INIT;
	while ( getline( fin, line ) ) {
		lineNum_++;
		if ( line.length() == 0 )
				continue;
		pos = line.find_first_not_of( "\t " );
		if ( pos == string::npos )
				continue;
		else
			line = line.substr( pos );
		if ( line.substr( 0, 2 ) == "//" )
				continue;
		if ( (pos = line.find("//")) != string::npos ) 
			line = line.substr( 0, pos );
		if ( line.substr( 0, 2 ) == "/*" ) {
				parseMode = COMMENT;
		} else if ( line.find( "*/" ) != string::npos ) {
				parseMode = DATA;
				continue;
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
				readData( line );
		else if ( parseMode == INIT ) {
				parseMode = readInit( line );
		}
	}
	
	cout << " innerRead: " <<
			lineNum_ << " lines read, " << 
			numCompartments_ << " compartments, " << 
			numMols_ << " molecules, " << 
			numReacs_ << " reacs, " << 
			numOthers_ << " others," <<
			" PlotDt = " << plotdt_ <<
			endl;
}

ReadKkit::ParseMode ReadKkit::readInit( const string& line )
{
	vector< string > argv;
	chopLine( line, argv ); 
	if ( argv.size() < 3 )
		return INIT;

	if ( argv[0] == "FASTDT" ) {
		fastdt_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "SIMDT" ) {
		simdt_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "CONTROLDT" ) {
		controldt_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "PLOTDT" ) {
		plotdt_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "MAXTIME" ) {
		maxtime_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "TRANSIENT_TIME" ) {
		transientTime_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "VARIABLE_DT_FLAG" ) {
		useVariableDt_ = atoi( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "DEFAULT_VOL" ) {
		defaultVol_ = atof( argv[2].c_str() );
		return INIT;
	}
	if ( argv[0] == "VERSION" ) {
		version_ = atoi( argv[2].c_str() );
		return INIT;
	}

	if ( argv[0] == "initdump" ) {
		initdumpVersion_ = atoi( argv[2].c_str() );
		return DATA;
	}

	return INIT;
}

/**
 * The readcell function implements the old GENESIS cellreader
 * functionality. Although it is really a parser operation, I
 * put it here in Shell because the cell format is independent
 * of parser and is likely to remain a legacy for a while.
 */
void ReadKkit::read(
	const string& filename, 
	const string& cellname,
	Id pa )
{
	ifstream fin( filename.c_str() );
	if (!fin){
		cerr << "ReadKkit::read: could not open file " << filename << endl;
		return;
    }

	vector< unsigned int > dimensions( 1, 1 );
	// Id model = s->doCreate( "Neutral", pa, cellname, dimensions );
	innerRead( fin );
}

void ReadKkit::readData( const string& line )
{
	vector< string > argv;
	chopLine( line, argv ); 
	
	if ( argv[0] == "simundump" )
		undump( argv );
	else if ( argv[0] == "addmsg" )
		addmsg( argv );
}

void ReadKkit::undump( const vector< string >& args)
{
	if ( args[1] == "kpool" )
		buildMol();
	else if ( args[1] == "kreac" )
		buildReac();
	else if ( args[1] == "kenz" )
		buildEnz();
	else if ( args[1] == "text" )
		buildText();
	else if ( args[1] == "xplot" )
		buildPlot();
	else if ( args[1] == "group" )
		buildGroup();
	else
		cout << "ReadKkit::undump: Do not know how to build '" << args[1] <<
		"'\n";
}

Id ReadKkit::buildCompartment()
{
	Id compt;
	numCompartments_++;
	return compt;
}

Id ReadKkit::buildReac()
{
	Id reac;
	numReacs_++;
	return reac;
}

Id ReadKkit::buildEnz()
{
	Id enz;
	numEnz_++;
	return enz;
}

Id ReadKkit::buildText()
{
	Id text;
	numOthers_++;
	return text;
}

Id ReadKkit::buildGroup()
{
	Id group;
	numOthers_++;
	return group;
}

Id ReadKkit::buildMol()
{
	Id mol;
	numMols_++;
	return mol;
}

Id ReadKkit::buildPlot()
{
	Id plot;
	numPlot_++;
	return plot;
}

Id ReadKkit::buildTab()
{
	Id tab;
	numOthers_++;
	return tab;
}

unsigned int ReadKkit::loadTab()
{
	return 0;
}

void ReadKkit::addmsg( const vector< string >& args)
{
	;
}
