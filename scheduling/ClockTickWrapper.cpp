/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ClockTick.h"
#include "ClockTickWrapper.h"


Finfo* ClockTickWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< int >(
		"stage", &ClockTickWrapper::getStage, 
		&ClockTickWrapper::setStage, "int" ),
	new ReadOnlyValueFinfo< int >(
		"ordinal", &ClockTickWrapper::getOrdinal, "int" ),
	new ValueFinfo< double >(
		"nextt", &ClockTickWrapper::getNextt, 
		&ClockTickWrapper::setNextt, "double" ),
	new ValueFinfo< double >(
		"epsnextt", &ClockTickWrapper::getEpsnextt, 
		&ClockTickWrapper::setEpsnextt, "double" ),
	new ValueFinfo< double >(
		"max_clocks", &ClockTickWrapper::getMax_clocks, 
		&ClockTickWrapper::setMax_clocks, "double" ),
	new ValueFinfo< double >(
		"nclocks", &ClockTickWrapper::getNclocks, 
		&ClockTickWrapper::setNclocks, "double" ),
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< string >(
		"path", &ClockTickWrapper::getPath, 
		&ClockTickWrapper::setPath, "string" ),
	new ValueFinfo< double >(
		"dt", &ClockTickWrapper::getDt, 
		&ClockTickWrapper::setDt, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &ClockTickWrapper::getProcessSrc, 
		"processIn", 1 ),
	new NSrc0Finfo(
		"reinitOut", &ClockTickWrapper::getReinitSrc, 
		"reinitIn", 1 ),
	new NSrc1Finfo< double >(
		"passStepOut", &ClockTickWrapper::getPassStepSrc, 
		"checkStepIn", 1 ),
	new SingleSrc2Finfo< double, Conn* >(
		"dtOut", &ClockTickWrapper::getDtSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"checkStepIn", &ClockTickWrapper::checkStepFunc,
		&ClockTickWrapper::getSolverStepConn, "passStepOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ClockTickWrapper::processFunc,
		&ClockTickWrapper::getClockConn, "processOut", 1 ),
	new Dest0Finfo(
		"reinitIn", &ClockTickWrapper::reinitFunc,
		&ClockTickWrapper::getClockConn, "reinitOut", 1 ),
	new Dest0Finfo(
		"reschedIn", &ClockTickWrapper::reschedFunc,
		&ClockTickWrapper::getClockConn, "", 1 ),
	new Dest1Finfo< Element* >(
		"schedNewObjectIn", &ClockTickWrapper::schedNewObjectFunc,
		&ClockTickWrapper::getClockConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"clock", &ClockTickWrapper::getClockConn,
		"processIn, reinitIn, reschedIn, schedNewObjectIn, dtOut" ),
	new SharedFinfo(
		"process", &ClockTickWrapper::getProcessConn,
		"processOut, reinitOut" ),
	new SharedFinfo(
		"solverStep", &ClockTickWrapper::getSolverStepConn,
		"passStepOut, checkStepIn" ),
};

const Cinfo ClockTickWrapper::cinfo_(
	"ClockTick",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"ClockTick: ClockTick class. Controls execution of objects on a given dt.",
	"Neutral",
	ClockTickWrapper::fieldArray_,
	sizeof(ClockTickWrapper::fieldArray_)/sizeof(Finfo *),
	&ClockTickWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// EvalField function definitions
///////////////////////////////////////////////////

string ClockTickWrapper::localGetPath() const
{
			return path_;
}
void ClockTickWrapper::localSetPath( string value ) {
			innerSetPath( value );
}
double ClockTickWrapper::localGetDt() const
{
			return dt_;
}
void ClockTickWrapper::localSetDt( double value ) {
			dt_ = value;
			dtSrc_.send( dt_, dtSrc_.conn() );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ClockTickWrapper::reinitFuncLocal(  )
{
			nextt_ = 0.0;
			epsnextt_ = 0.0;
			reinitSrc_.send( );
}
void ClockTickWrapper::schedNewObjectFuncLocal( Element* e )
{
			unsigned int i;
			for (i = 0; i < managedCinfo_.size(); i++ ) {
				const Cinfo* c = managedCinfo_[i];
				if ( c == 0 || e->cinfo()->isA( c ) ) {
					string p = e->path();
					const string& q = managedPath_[i];
					if ( p.substr( 0, q.length() ) == q ) {
						Field src = field( "process" );
						Field dest = e->field( "process" );
						src.add( dest );
						return;
					}
				}
			}
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* clockConnClockTickLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockTickWrapper, clockConn_ );
	return reinterpret_cast< ClockTickWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
int ClockTick::ordinalCounter_ = 0;
void ClockTickWrapper::innerSetPath( const string& path )
{
	path_ = path;
	size_t pos = path.find_last_of("/");
	if ( pos == string::npos || pos == path.length()) {
		cerr << "Error:ClockTickWrapper::innerSetPath: no finfo name on tick:" << name() << " in path " << path << "\n"; 
		return;
	}
	string finfoName = path.substr( pos + 1 );
	string pathHead = path.substr( 0, pos );
	vector< Element* > ret;
	vector< Element* >::iterator i;
	Element::startFind( pathHead, ret );
	processSrc_.dropAll();
	reinitSrc_.dropAll();
	Field src = field( "process" );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		if ( !( *i )->isSolved() ) {
			Field dest = ( *i )->field( finfoName );
			src.add( dest );
		}
	}
	separatePathOnCommas();
}
void ClockTickWrapper::separatePathOnCommas()
{
	string::size_type pos = 0;
	string temp = path_;
	pos = temp.find( "," );
	fillManagementInfo( temp.substr( 0, pos ) );
	while ( pos != string::npos ) {
		temp = temp.substr( pos + 1 );
		pos = temp.find( "," );
		fillManagementInfo( temp.substr( 0, pos ) );
	}
}
void ClockTickWrapper::fillManagementInfo( const string& s )
{
	string::size_type pos = s.find_first_of( "#" );
	managedPath_.push_back( s.substr( 0, pos ) );
	pos = s.find( "=" ); 
	if ( pos == string::npos ) {
		managedCinfo_.push_back( 0 );
	} else {
		string tname = s.substr( pos + 1 );
		pos = tname.find( "]" );
		const Cinfo* c = Cinfo::find( tname.substr( 0, pos ) );
		managedCinfo_.push_back( c );
	}
}
Element* ClockTickWrapper::create(
	const string& name, Element* pa, const Element* proto )
{
	if ( pa->cinfo()->isA( Cinfo::find( "ClockJob" ) ) ) {
		Field clock = pa->field( "clock" );
		ClockTickWrapper* ret = new ClockTickWrapper( name );
		ret->assignOrdinal();
		Field tick = ret->field( "clock" );
		clock.add( tick );
		return ret;
	};
	return 0;
}
