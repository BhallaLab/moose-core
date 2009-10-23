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
      
      new SrcFinfo( "requestInput", Ftype0::global(),
					"Sends out the request. Issued from the process call." ),
      new DestFinfo( "handleInput", Ftype1< double >::global(),
                     RFCAST( &AscFile::input ),
                     "Handle the returned value." ),
    };


  static Finfo* ascFileFinfos[] =
    {
      ///////////////////////////////////////////////////////
      // Field definitions
      ///////////////////////////////////////////////////////
      new ValueFinfo( "filename",
                      ValueFtype1< string >::global(),
                      GFCAST( &AscFile::getFileName ),
                      RFCAST( &AscFile::setFileName ),
                      "Data will be written into this file."
                      ),
      new ValueFinfo( "append",
                      ValueFtype1< int >::global(),
                      GFCAST( &AscFile::getAppend ),
                      RFCAST( &AscFile::setAppend ),
                      "Data will be appended to file only if non-zero. "
                      "On by default."
                      ),
      new ValueFinfo( "time",
                      ValueFtype1< int >::global(),
                      GFCAST( &AscFile::getTime ),
                      RFCAST( &AscFile::setTime ),
                      "Simulation time will be written to first column "
                      "only if this value is non-zero. On by default."
                      ),
      new ValueFinfo( "header",
                      ValueFtype1< int >::global(),
                      GFCAST( &AscFile::getHeader ),
                      RFCAST( &AscFile::setHeader ),
                      "A header containing names of objects being recorded. "
                      "Currently no way to write the objects' field names into"
                      "the header. Will be written only if this value is "
                      "non-zero. On by default."
                      ),
      new ValueFinfo( "comment",
                      ValueFtype1< string >::global(),
                      GFCAST( &AscFile::getComment ),
                      RFCAST( &AscFile::setComment ),
                      "This string will be inserted at the beginning of the "
                      "header line. Can be used to comment out the header "
                      "line. Default value: \"#\"."
                      ),
      new ValueFinfo( "delimiter",
                      ValueFtype1< string >::global(),
                      GFCAST( &AscFile::getDelimiter ),
                      RFCAST( &AscFile::setDelimiter ),
                      "This string will be used to separate entries in a row. "
                      "Default value: \"\\t\"."
                      ),
      ///////////////////////////////////////////////////////
      // MsgSrc definitions
      ///////////////////////////////////////////////////////
/**
 * \todo At present this call must be made explicitly at the end of a simulation
 * to flush the file. Currently elements do not get destroyed at exit, which is
 * why the destructor also never gets called.
 */
      new DestFinfo( "close", Ftype0::global(),
                     RFCAST( &AscFile::closeFunc ),
                     "Explicit call to close the AscFile. Useful if the user "
                     "wishes to switch to a new file between resets."
                   ),
      ///////////////////////////////////////////////////////
      // MsgDest definitions
      ///////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////
      // Shared definitions
      ///////////////////////////////////////////////////////
      new SharedFinfo( "save", inputRequestShared, 
                       sizeof( inputRequestShared ) / sizeof( Finfo* ) ),

      process,
    };

  static SchedInfo schedInfo[] = { { process, 0, 0 } };

  static string doc[] =
	{
		"Name", "AscFile",
		"Author", "Johannes Hjorth, 2008, KTH, Stockholm",
		"Description", "AscFile: Multi-column output to file.",
	};

  static Cinfo ascFileCinfo(
               	doc,
               	sizeof( doc ) / sizeof( string ),
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
        append_( 1 ),
        time_( 1 ),
        header_( 1 ),
        delimiter_( "\t" ),
        comment_( "#" ),
        fileOut_( new std::ofstream() )
{ ; }

/**
 * At present the destructor is never called at exit, because elements do not
 * get destroy at exit.
 */
AscFile::~AscFile()
{
  fileOut_->close();
  
  /*
   * If fileOut_ is used as an object, instead of a pointer, then this file
   * does not compile. This is due to a call to the following function during
   * Cinfo initialization:
   *    ValueFtype1< AscFile >::global()
   * Not really sure why, but it has something to do with a copy constructor.
   * Should be able to find a cleaner way.
   */
  delete fileOut_;
  fileOut_ = 0;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void AscFile::setFileName( const Conn* c, string s )
{
   static_cast< AscFile* >( c->data() )->filename_ = s;
}

string AscFile::getFileName( Eref e )
{
  return static_cast< AscFile* >( e.data() )->filename_;
}

void AscFile::setAppend( const Conn* c, int v )
{
  static_cast< AscFile* >( c->data() )->append_ = v;
}

int AscFile::getAppend( Eref e )
{
  return static_cast< AscFile* >( e.data() )->append_;
}

void AscFile::setTime( const Conn* c, int v )
{
  static_cast< AscFile* >( c->data() )->time_ = v;
}

int AscFile::getTime( Eref e )
{
  return static_cast< AscFile* >( e.data() )->time_;
}

void AscFile::setHeader( const Conn* c, int v )
{
  static_cast< AscFile* >( c->data() )->header_ = v;
}

int AscFile::getHeader( Eref e )
{
  return static_cast< AscFile* >( e.data() )->header_;
}

void AscFile::setDelimiter( const Conn* c, string v )
{
  static_cast< AscFile* >( c->data() )->delimiter_ = v;
}

string AscFile::getDelimiter( Eref e )
{
  return static_cast< AscFile* >( e.data() )->delimiter_;
}

void AscFile::setComment( const Conn* c, string v )
{
  static_cast< AscFile* >( c->data() )->comment_ = v;
}

string AscFile::getComment( Eref e )
{
  return static_cast< AscFile* >( e.data() )->comment_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void AscFile::reinitFunc( const Conn* c, ProcInfo info ) 
{
  static_cast< AscFile* >( c->data() )->reinitFuncLocal(c->target());
}

void AscFile::reinitFuncLocal( Eref e ) 
{
  unsigned int nCols = e->numTargets( "save" );
  columnData_.resize(nCols, 0.0);

  // Close the file if it is open
  if(fileOut_->is_open()) {
    fileOut_->close();
  }

  // Open file for writing

  if(append_) {
    fileOut_->open( filename_.c_str(), ios_base::app );
  } else {
    fileOut_->open( filename_.c_str(), ios_base::trunc );
  }
  
  if(!fileOut_->good()) {
    cerr << "AscFile::reinitFuncLocal: Unable to open " << filename_ << endl;
  } else if ( header_ ) {
    // Write header
    string header = comment_;
    if ( time_ )
    	header += "Time" + delimiter_;
    
    Conn* i = e->targets( "save", 0 );
    for ( ; i->good(); i->increment() )
        header += i->target()->name() + delimiter_;
    delete i;
    
    *fileOut_ << header << "\n";
  }
}

void AscFile::processFunc( const Conn* c, ProcInfo p )
{
  // The data arriving by message belongs to the previous timestep.
  static_cast< AscFile* >( c->data() )->
    processFuncLocal(c->target(), p->currTime_ - p->dt_);
}

void AscFile::processFuncLocal(Eref e, double time)
{
  if(! fileOut_->good()) {
    cerr << "Error: AscFile::processFuncLocal, unable to write to file" << endl;
    return;
  }
  
  // Receive data
  send0(e, inputRequestSlot );
  
  // Write row of data
  vector< double >::iterator i;
  if(time >= 0) {
    if ( time_ )
      *fileOut_ << time << delimiter_;
    
    for( i = columnData_.begin(); i != columnData_.end(); i++)
      *fileOut_ << *i << delimiter_;
    
    *fileOut_ << "\n";
  }
}

void AscFile::input( const Conn* c, double value )
{
  static_cast< AscFile* >( c->data() )->
    inputLocal( c->targetIndex(), value );
}

void AscFile::inputLocal( unsigned int columnId, double value )
{
  // cerr << "columnId = " << columnId << ", size = " << columnData_.size() << endl;

  // assert(columnId < columnData_.size()); //  Temp. commented

  // This is temporary check, since the reinit function fails
  // to get the right number of incoming messages sometimes
  if(columnId >= columnData_.size())
    columnData_.resize(columnId + 1);

  columnData_[columnId] = value;
}

void AscFile::closeFunc( const Conn* c ) 
{
  static_cast< AscFile* >( c->data() )->closeFuncLocal( );
}

void AscFile::closeFuncLocal( ) 
{
  fileOut_->close();
  delete fileOut_;
  // Just in case the object is used again after calling "close".
  fileOut_ = new std::ofstream();
}
