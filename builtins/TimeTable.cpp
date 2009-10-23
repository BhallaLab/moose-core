/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/


#include <fstream>
#include <math.h>
#include "moose.h"
#include "TimeTable.h"

const Cinfo* initTimeTableCinfo()
{
  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &TimeTable::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &TimeTable::reinitFunc ) ),
    };
  static Finfo* process = new SharedFinfo( "process", processShared,
                                           sizeof( processShared ) / sizeof( Finfo* ) );

  static Finfo* timeTableFinfos[] =
    {
      ///////////////////////////////////////////////////////
      // Field definitions
      ///////////////////////////////////////////////////////
      new ValueFinfo( "filename", 
                      ValueFtype1< string >::global(),
                      GFCAST( &TimeTable::getFilename ),
                      RFCAST( &TimeTable::setFilename )
                      ),
      new ValueFinfo( "method", 
                      ValueFtype1< int >::global(),
                      GFCAST( &TimeTable::getMethod ),
                      RFCAST( &TimeTable::setMethod )
                      ),
      new ValueFinfo( "tableVector", 
                      ValueFtype1< vector< double > >::global(),
                      GFCAST( &TimeTable::getTableVector ),
                      RFCAST( &TimeTable::setTableVector )
                      ),
      new ValueFinfo( "tableSize",
                      ValueFtype1< unsigned int >::global(),
                      GFCAST( &TimeTable::getTableSize ),
                      RFCAST( &TimeTable::setTableSize )
                      ),
      new LookupFinfo( "table",
                       LookupFtype< double, unsigned int >::global(),
                       GFCAST( &TimeTable::getTable ),
                       RFCAST( &TimeTable::setTable )
                       ),
      new ValueFinfo( "state",
                      ValueFtype1< double >::global(),
                      GFCAST( &TimeTable::getState ),
                      &dummyFunc
                      ),
      ///////////////////////////////////////////////////////
      // MsgSrc definitions
      ///////////////////////////////////////////////////////

      //~ // Continous variable switching between 0 and 1
      //~ new SrcFinfo( "state", Ftype1< double >::global() ),

      // Event triggered by a spike
      new SrcFinfo( "event",
                    Ftype1< double >::global(),
                    "Sends spike events. No intermediate SpikeGen object "
                    "required as in GENESIS"
                    ),

      ///////////////////////////////////////////////////////
      // MsgDest definitions
      ///////////////////////////////////////////////////////

      ///////////////////////////////////////////////////////
      // Shared definitions
      ///////////////////////////////////////////////////////
      process,
    };

  // Schedule molecules for the slower clock, stage 0.
  static SchedInfo schedInfo[] = { { process, 0, 0 } };

  static string doc[] =
	{
		"Name", "TimeTable",
		"Author", "Johannes Hjorth, 2008, KTH, Stockholm",
		"Description", "TimeTable: Read in spike times from file.",
	};
  static Cinfo timeTableCinfo(
                            doc,
                            sizeof( doc ) / sizeof( string ),
                            initNeutralCinfo(),
                            timeTableFinfos,
                            sizeof( timeTableFinfos )/sizeof(Finfo *),
                            ValueFtype1< TimeTable >::global(),
                            schedInfo, 1
                            );

  return &timeTableCinfo;
}

static const Cinfo* timeTableCinfo = initTimeTableCinfo();

//~ static const Slot stateSlot = initTimeTableCinfo()->getSlot( "state" );
static const Slot eventSlot =  initTimeTableCinfo()->getSlot( "event" );


///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

TimeTable::TimeTable()
  :
  state_( 0.0 ),
  curPos_( 0 ),
  method_( 4 )
{ ; }

TimeTable::~TimeTable()
{ ; }

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/* Filename */
void TimeTable::setFilename( const Conn* c, string filename )
{
  static_cast< TimeTable* >( c->data() )->localSetFilename( filename );
}

string TimeTable::getFilename( Eref e )
{
  return static_cast< TimeTable* >( e.data() )->filename_;
}

void TimeTable::localSetFilename( string filename )
{
  filename_ = filename;
  
  std::ifstream fin( filename_.c_str() );
  string line;
  
  if( !fin.good()) {
    cout << "Error: TimeTable::innerload: Unable to open file" 
         << filename_ << endl;
  }
  
  //~ If lines need to be skipped:
  //~ for(unsigned int i = 0; (i < skipLines) & fin.good() ; i++)
    //~ getline( fin, line );
  
  timeTable_.clear();
  
  double dataPoint, dataPointOld = -1000;
  while( fin >> dataPoint ) {
    timeTable_.push_back(dataPoint);

    if(dataPoint < dataPointOld) {
      cerr << "TimeTable: Warning: Spike times in file " << filename_
           << " are not in increasing order."
           << endl;
    }

    dataPointOld = dataPoint;
  }
}

/* Method */
void TimeTable::setMethod( const Conn* c, int method )
{
  if ( method != 4 ) {
    cerr <<
      "Error: TimeTable::setMethod: "
      "Currently only method 4 (loading from file) supported.\n";
    return;
  }
  
  static_cast< TimeTable* >( c->data() )->method_ = method;
}

int TimeTable::getMethod( Eref e )
{
  return static_cast< TimeTable* >( e.data() )->method_;
}

/* TableVector */
void TimeTable::setTableVector( const Conn* c, vector< double > table )
{
  static_cast< TimeTable* >( c->data() )->timeTable_ = table;
}

vector< double > TimeTable::getTableVector( Eref e )
{
  return static_cast< TimeTable* >( e.data() )->timeTable_;
}

/* TableSize */
void TimeTable::setTableSize( const Conn* c, unsigned int size )
{
  static_cast< TimeTable* >( c->data() )->
      timeTable_.resize( size, 0.0 );
}

unsigned int TimeTable::getTableSize( Eref e )
{
  return static_cast< TimeTable* >( e.data() )->timeTable_.size();
}

/* Table value */
void TimeTable::setTable(const Conn* c, double val, const unsigned int& i )
{
  static_cast< TimeTable* >( c->data() )->localSetTable( val, i );
}

double TimeTable::getTable( Eref e, const unsigned int& i )
{
  return static_cast< TimeTable* >( e.data() )->localGetTable( i );
}

void TimeTable::localSetTable( double value, unsigned int index ) {
  if ( index < timeTable_.size() )
    timeTable_[ index ] = value;
}

double TimeTable::localGetTable( unsigned int index ) const {
  if ( index < timeTable_.size() )
    return timeTable_[ index ];
  
  return 0.0; // Thou shall not call outside the defined vector
}

/* state */
double TimeTable::getState( Eref e )
{
  return static_cast< TimeTable* >( e.data() )->state_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void TimeTable::reinitFunc( const Conn* c, ProcInfo info )
{
  static_cast< TimeTable* >( c->data() )->reinitFuncLocal( );
}

void TimeTable::reinitFuncLocal( )
{
  curPos_ = 0;
  state_ = 0;
}

void TimeTable::processFunc( const Conn* c, ProcInfo info )
{
  static_cast< TimeTable* >( c->data() )->processFuncLocal( c->target(), info );
}

void TimeTable::processFuncLocal( Eref e, ProcInfo info )
{

  // Two ways of telling the world about the spike events, both
  // happening in parallel
  //
  // state is a continous variable, switching from 0 to 1 when spiking
  // event is an event, that happens at the time of a spike
  //

  state_ = 0;

  if ( curPos_ < timeTable_.size() &&
       info->currTime_ >= timeTable_[curPos_] ) {
    send1< double >( e, eventSlot, timeTable_[curPos_]);
    curPos_++;
    state_ = 1;
  }

  //~ send1< double >( e, stateSlot, state_ );
}



#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"

void testTimeTable()
{
  static double check[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2 };


  cout << "\nTesting TimeTable" << flush;

  Eref n = Neutral::create( "Neutral", "n", Element::root()->id(),
                            Id::scratchId() );
  Element* tt = Neutral::create( "TimeTable", "tt", n->id(),
                                 Id::scratchId() );
  ASSERT( tt != 0, "creating TimeTable" );

  Element* sg = Neutral::create( "SpikeGen", "s", n->id(),
                                Id::scratchId() );
  ASSERT( sg != 0, "creating SpikeGen" );

  bool ret;

  // Adding message
  ret = Eref(tt).add("state", sg, "Vm");
  ASSERT( ret, "adding msg");


  ProcInfoBase p;
  SetConn cm0( tt, 0 );
  SetConn cm1( sg, 0 );

  // Loading file
  //~ TimeTable::load( &cm0, fileName, (unsigned int) 0);
  //~ TimeTable::setFilename( &cm0, fileName );
  TimeTable::setTableSize( &cm0, 2 );
  TimeTable::setTable( &cm0, 0.3, 0 );
  TimeTable::setTable( &cm0, 0.6, 1 );

  p.dt_ = 0.1;

  set< double >( sg, "threshold", 0.5 );
  set< double >( sg, "abs_refract", 0.0 );
  set< double >( sg, "amplitude", 1.0 );

  TimeTable::reinitFunc( &cm0, &p );
  TimeTable::reinitFunc( &cm1, &p );

  unsigned int i = 0;
  double numSpikesSent = 0;
  //double numSpikesReceived = 0;

  for ( p.currTime_ = 0.0; p.currTime_ < 1; p.currTime_ += p.dt_ )
    {
      TimeTable::processFunc( &cm0, &p );
      TimeTable::processFunc( &cm1, &p );

      numSpikesSent += TimeTable::getState( Eref( tt ) );

      ASSERT( numSpikesSent != check[i], "testing reading");
      i++;
    }

  ASSERT( numSpikesSent != 3, "Testing reading spikes")

  // Get rid of all the compartments.
  set( n, "destroy" );


}
#endif
