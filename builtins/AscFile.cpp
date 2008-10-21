#include <fstream>
#include <math.h>
#include "moose.h"
#include "AscFile.h"

const Cinfo* initAscFileCinfo()
{
  
  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &AscFile::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &AscFile::reinitFunc ) ),
    };
      
  static Finfo* process = 
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );


  static Finfo* inputRequestShared[] =
    {
      // Sends out the request. Issued from the process call.
      new SrcFinfo( "requestInput", Ftype0::global() ),
      // Handle the returned value.
      new DestFinfo( "handleInput", Ftype1< double >::global(),
                     RFCAST( &AscFile::input ) ),
    };


  static Finfo* ascFileFinfos[] =
    {
      ///////////////////////////////////////////////////////
      // Field definitions
      ///////////////////////////////////////////////////////
      new ValueFinfo( "fileName",
                      ValueFtype1< string >::global(),
                      GFCAST( &AscFile::getFileName ),
                      RFCAST( &AscFile::setFileName )
                      ),
      new ValueFinfo( "appendFlag",
                      ValueFtype1< int >::global(),
                      GFCAST( &AscFile::getAppendFlag ),
                      RFCAST( &AscFile::setAppendFlag )
                      ),
      ///////////////////////////////////////////////////////
      // MsgSrc definitions
      ///////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////
      // MsgDest definitions
      ///////////////////////////////////////////////////////
      // Replaced with SharedFinfo, to get hsolver to work.
      //~ new DestFinfo( "save",
                     //~ Ftype1< double >::global(),
                     //~ RFCAST( &AscFile::input )
                     //~ ),
      ///////////////////////////////////////////////////////
      // Synapse definitions
      ///////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////
      // Shared definitions
      ///////////////////////////////////////////////////////
      new SharedFinfo( "save", inputRequestShared, 
                       sizeof( inputRequestShared ) / sizeof( Finfo* ) ),

      process,
    };

  static SchedInfo schedInfo[] = { { process, 0, 0 } };

  static Cinfo ascFileCinfo(
                "AscFile",
                "Johannes Hjorth, 2008, KTH, Stockholm",
                "AscFile: Multi-column output to file.",
                initNeutralCinfo(),
                ascFileFinfos,
                sizeof( ascFileFinfos )/sizeof(Finfo *),
                ValueFtype1< AscFile >::global(),
                schedInfo, 1
                );

  return &ascFileCinfo;

}

static const Cinfo* ascFileCinfo = initAscFileCinfo();

static const Slot inputRequestSlot = 
	initAscFileCinfo()->getSlot( "save.requestInput" );


AscFile::AscFile()
        :
        fileName_( "utdata.txt" ),
        nCols_( 0 ),
        appendFlag_( 0 )
{
  fileOut_ = new std::ofstream();
}

AscFile::~AscFile() {
  cerr << "DESTROYING AscFile!!" << endl;
}

///////////////////////////////////////////////////



void AscFile::processFunc( const Conn* c, ProcInfo p )
{
  // The data arriving by message belongs to the previous timestep.
  static_cast< AscFile* >( c->data() )->processFuncLocal(c->target(), p->currTime_ - p->dt_);
}

void AscFile::processFuncLocal(Eref e, double time)
{
  // Write data to file

  //~ cerr << "processFuncLocal" << endl;

  vector< double >::iterator i;

  send0(e, inputRequestSlot );

// !!! Flushing into stream here (slowww)

  if(fileOut_->good()) {

    if(time >= 0) {
      // First column should contain the time
      *fileOut_ << time << " " << flush;

      for( i = columnData_.begin(); i != columnData_.end(); i++) {
        *fileOut_ << *i << " " << flush;
      }

      *fileOut_ << "\n" << flush;
    }

    // Empty the old data
    // columnData_.clear();
  }
  else {
    cerr << "AscFile::processFuncLocal, unable to write to file" << endl;
  }

}

void AscFile::reinitFunc( const Conn* c, ProcInfo info) 
{
  static_cast< AscFile* >( c->data() )->reinitFuncLocal(c);
}

void AscFile::reinitFuncLocal( const Conn* c ) 
{
  cerr << "AscFile::reinitFuncLocal called." << endl;


  nCols_ = c->target().e->numTargets( "save" );
  columnData_.resize(nCols_);
  cout << "Saving " <<  nCols_ << " column(s) of data" << endl;



  // Close the file if it is open
  if(fileOut_->is_open()) {
    fileOut_->close();
  }

  // Open file for writing

  if(appendFlag_) {
    fileOut_->open( fileName_.c_str(), ios_base::app );
  } else {
    fileOut_->open( fileName_.c_str(), ios_base::trunc );
  }

  if(!fileOut_->good()) {
    cerr << "AscFile::reinitFuncLocal: Unable to open " << fileName_ << endl;
  }

  //bup = fileOut_;
  //bup2 = fileOut_;

}



///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void AscFile::setFileName( const Conn* c, string s )
{
   static_cast< AscFile* >( c->data() )->fileName_ = s;
}

string AscFile::getFileName( Eref e )
{
  return static_cast< AscFile* >( e.data() )->fileName_;
}

void AscFile::setAppendFlag( const Conn* c, int f )
{
  static_cast< AscFile* >( c->data() )->appendFlag_ = f;
}

int AscFile::getAppendFlag( Eref e )
{
  return static_cast< AscFile* >( e.data() )->appendFlag_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


void AscFile::input( const Conn* c, double value )
{
  static_cast< AscFile* >( c->data() )->inputLocal( c->targetIndex(), value );
}

void AscFile::inputLocal( unsigned int columnId, double value )
{
  // cerr << "columnId = " << columnId << ", size = " << columnData_.size() << endl;

  // assert(columnId < columnData_.size()); //  Temp. commented


  // This is temporary check, since the reinit function fails
  // to get the right number of incoming messages sometimes
  if(columnId >= columnData_.size()) {
    nCols_ = columnId + 1;
    columnData_.resize(nCols_);
  }

  columnData_[columnId] = value;
}






















