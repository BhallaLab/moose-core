/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include <iomanip>
#include <fstream>
/*
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cassert>
#include <stdlib.h>

using namespace std;
typedef int Id; // dummy
*/
#include "header.h"
#include "ReadKkit.h"

unsigned int chopLine( const string& line, vector< string >& ret )
{
	ret.resize( 0 );
	stringstream ss( line );
	string arg;
	while ( ss >> arg ) {
		ret.push_back( arg );
	}
	return ret.size();
}

/*
int main( int argc, const char* argv[] )
{
	string fname = "dend_v26.g";
	if ( argc == 2 )
		fname = argv[1];
	ReadKkit rk;
	rk.read( fname, "cell", 0 );
}
*/

////////////////////////////////////////////////////////////////////////

ReadKkit::ReadKkit()
	:
	fastdt_( 0.001 ),
	simdt_( 0.01 ),
	controldt_( 0.1 ),
	plotdt_( 1 ),
	maxtime_( 1.0 ),
	transientTime_( 1.0 ),
	useVariableDt_( 0 ),
	defaultVol_( 1 ),
	version_( 11 ),
	initdumpVersion_( 3 ),
	numCompartments_( 0 ),
	numMols_( 0 ),
	numReacs_( 0 ),
	numEnz_( 0 ),
	numPlot_( 0 ),
	numOthers_( 0 ),
	lineNum_( 0 ),
	shell_( reinterpret_cast< Shell* >( Id().eref().data() ) )
{
	;
}

void ReadKkit::innerRead( ifstream& fin )
{
	string line;
	string temp;
	lineNum_ = 0;
	string::size_type pos;
	bool clearLine = 1;
	ParseMode parseMode = INIT;

	while ( getline( fin, temp ) ) {
		lineNum_++;
		if ( clearLine )
			line = "";

		if ( temp.length() == 0 )
				continue;
		pos = temp.find_last_not_of( "\t " );
		if ( pos == string::npos ) { 
			// Nothing new in line, go with what was left earlier, 
			// and clear out line for the next cycle.
			temp = "";
			clearLine = 1;
		} else {
			if ( temp[pos] == '\\' ) {
				temp[pos] = ' ';
				line.append( temp );
				clearLine = 0;
				continue;
			} else {
				line.append( temp );
				clearLine = 1;
			}
		}
			
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
			numEnz_ << " enzs, " << 
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
		Id kinetics = Neutral::child( Id().eref(), "kinetics" );
		if ( kinetics == Id() ) {
			vector< unsigned int > dims( 1, 1 );
			shell_->doCreate( "Neutral", Id(), "kinetics", dims );
		}
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
	else if ( argv[0] == "call" )
		call( argv );
	else if ( argv[0] == "simobjdump" )
		objdump( argv );
	else if ( argv[0] == "xtextload" )
		textload( argv );
	else if ( argv[0] == "loadtab" )
		loadtab( argv );
}

string pathTail( const string& path, string& head )
{
	string::size_type pos = path.find_last_of( "/" );
	assert( pos != string::npos );

	head = path.substr( 0, pos ); 
	return path.substr( pos + 1 );
}

Id ReadKkit::findParent( const string& path ) const
{
	return 1;
}

void assignArgs( map< string, int >& argConv, const vector< string >& args )
{
	for ( unsigned int i = 2; i != args.size(); ++i )
		argConv[ args[i] ] = i + 2;
}

void ReadKkit::objdump( const vector< string >& args)
{
	if ( args[1] == "kpool" )
		assignArgs( molMap_, args );
	else if ( args[1] == "kreac" )
		assignArgs( reacMap_, args );
	else if ( args[1] == "kenz" )
		assignArgs( enzMap_, args );
	else if ( args[1] == "group" )
		assignArgs( groupMap_, args );
	else if ( args[1] == "table" )
		assignArgs( tableMap_, args );
}

void ReadKkit::call( const vector< string >& args)
{
}

void ReadKkit::textload( const vector< string >& args)
{
}

void ReadKkit::loadtab( const vector< string >& args)
{
}

void ReadKkit::undump( const vector< string >& args)
{
	if ( args[1] == "kpool" )
		buildMol( args );
	else if ( args[1] == "kreac" )
		buildReac( args );
	else if ( args[1] == "kenz" )
		buildEnz( args );
	else if ( args[1] == "text" )
		buildText( args );
	else if ( args[1] == "xplot" )
		buildPlot( args );
	else if ( args[1] == "group" )
		buildGroup( args );
	else
		cout << "ReadKkit::undump: Do not know how to build '" << args[1] <<
		"'\n";
}

Id ReadKkit::buildCompartment( const vector< string >& args )
{
	Id compt;
	numCompartments_++;
	return compt;
}

Id ReadKkit::buildReac( const vector< string >& args )
{
	static vector< unsigned int > dim( 1, 1 );

	string head;
	string tail = pathTail( args[2], head );
	Id pa = shell_->doFind( head );
	assert( pa != Id() );

	double kf = atof( args[ reacMap_[ "kf" ] ].c_str() );
	double kb = atof( args[ reacMap_[ "kb" ] ].c_str() );

	Id reac = shell_->doCreate( "Reac", pa, tail, dim );
	reacIds_[ args[2].substr( 10 ) ] = reac; 

	Field< double >::set( reac.eref(), "kf", kf );
	Field< double >::set( reac.eref(), "kb", kb );

	Id info = buildInfo( reac, reacMap_, args );

	numReacs_++;
	return reac;
}

Id ReadKkit::buildEnz( const vector< string >& args )
{
	static vector< unsigned int > dim( 1, 1 );
	string head;
	string tail = pathTail( args[2], head );
	Id pa = findParent( head );
	assert ( pa != Id() );

	double k1 = atof( args[ reacMap_[ "k1" ] ].c_str() );
	double k2 = atof( args[ reacMap_[ "k2" ] ].c_str() );
	double k3 = atof( args[ reacMap_[ "k3" ] ].c_str() );
	double nComplexInit = atof( args[ reacMap_[ "nComplexInit" ] ].c_str());
	double vol = atof( args[ reacMap_[ "vol" ] ].c_str());
	bool isMM = atoi( args[ reacMap_[ "usecomplex" ] ].c_str());

	if ( !isMM ) {
		Id enz = shell_->doCreate( "Enz", pa, tail, dim );
		string enzPath = args[2].substr( 10 );
		enzIds_[ enzPath ] = enz; 

		Field< double >::set( enz.eref(), "k1", k1 );
		Field< double >::set( enz.eref(), "k2", k2 );
		Field< double >::set( enz.eref(), "k3", k3 );

		string cplxName = tail + "_cplx";
		string cplxPath = enzPath + "/" + cplxName;
		Id cplx = shell_->doCreate( "Mol", enz, cplxName, dim );
		molIds_[ cplxPath ] = enz; 
		Field< double >::set( cplx.eref(), "nInit", nComplexInit );
		Id info = buildInfo( enz, enzMap_, args );
		numEnz_++;
		return enz;
	} else {
		return Id();
	}
}

Id ReadKkit::buildText( const vector< string >& args )
{
	Id text;
	numOthers_++;
	return text;
}

Id ReadKkit::buildInfo( Id parent, 
	map< string, int >& m, const vector< string >& args )
{
	static vector< unsigned int > dim( 1, 1 );
	Id info = shell_->doCreate( "Neutral", parent, "info", dim );

	/*
	Id info = shell_->doCreate( "KkitInfo", parent, "info", dim );
	double x = atof( args[ m[ "x" ] ].c_str() );
	double y = atof( args[ m[ "y" ] ].c_str() );
	int color = atoi( args[ m[ "color" ] ].c_str() );

	Field< double >::set( info.eref(), "x", x );
	Field< double >::set( info.eref(), "y", y );
	Field< int >::set( info.eref(), "color", color );
	*/

	return info;
}

Id ReadKkit::buildGroup( const vector< string >& args )
{
	static vector< unsigned int > dim( 1, 1 );

	string head;
	string tail = pathTail( args[2], head );

	Id pa = shell_->doFind( head );
	assert( pa != Id() );
	Id group = shell_->doCreate( "Neutral", pa, tail, dim );
	Id info = buildInfo( group, groupMap_, args );

	numOthers_++;
	return group;
}

Id ReadKkit::buildMol( const vector< string >& args )
{
	static vector< unsigned int > dim( 1, 1 );

	string head;
	string tail = pathTail( args[2], head );
	Id pa = shell_->doFind( head );
	assert( pa != Id() );

	double nInit = atof( args[ molMap_[ "nInit" ] ].c_str() );
	double vol = atof( args[ molMap_[ "vol" ] ].c_str() );
	int slaveEnable = atoi( args[ molMap_[ "slave_enable" ] ].c_str() );
	double diffConst = atof( args[ molMap_[ "DiffConst" ] ].c_str() );

	Id mol = shell_->doCreate( "Mol", pa, tail, dim );
	// skip the 10 chars of "/kinetics/"
	molIds_[ args[2].substr( 10 ) ] = mol; 

	Field< double >::set( mol.eref(), "nInit", nInit );

	Id info = buildInfo( mol, molMap_, args );

	/*
	cout << setw( 20 ) << head << setw( 15 ) << tail << "	" << 
		setw( 12 ) << nInit << "	" << 
		vol << "	" << diffConst << "	" <<
		slaveEnable << endl;
		*/
	numMols_++;
	return mol;
}

Id ReadKkit::buildPlot( const vector< string >& args )
{
	Id plot;
	numPlot_++;
	return plot;
}

Id ReadKkit::buildTab( const vector< string >& args )
{
	Id tab;
	numOthers_++;
	return tab;
}

unsigned int ReadKkit::loadTab( const vector< string >& args )
{
	return 0;
}

void ReadKkit::addmsg( const vector< string >& args)
{
	string src = args[1].substr( 10 );
	string dest = args[2].substr( 10 );
	MsgId ret;
	
	if ( args[3] == "REAC" ) {
		if ( args[4] == "A" && args[5] == "B" ) {
			map< string, Id >::iterator i = reacIds_.find( src );
			assert( i != reacIds_.end() );
			Id srcId = i->second;

			i = molIds_.find( dest );
			assert( i != molIds_.end() );
			Id destId = i->second;

			// dest mol is substrate of src reac
			ret = shell_->doAddMsg( "single", 
				FullId( srcId, 0 ), "sub", 
				FullId( destId, 0 ), "reac" ); 
			assert( ret != Msg::badMsg );
		} 
		else if ( args[4] == "B" && args[5] == "A" ) {
			// dest mol is product of src reac
			map< string, Id >::iterator i = reacIds_.find( src );
			assert( i != reacIds_.end() );
			Id srcId = i->second;

			i = molIds_.find( dest );
			assert( i != molIds_.end() );
			Id destId = i->second;

			// dest mol is substrate of src reac
			ret = shell_->doAddMsg( "single", 
				FullId( srcId, 0 ), "prd", 
				FullId( destId, 0 ), "reac" ); 
			assert( ret != Msg::badMsg );
		}
	}
}
