/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <boost/log/trivial.hpp>

#include "header.h"
#include "global.h"
#include <fstream>

#include "TableBase.h"
#include "Table.h"
#include "Clock.h"


static SrcFinfo1< vector< double >* > *requestOut()
{
    static SrcFinfo1< vector< double >* > requestOut(
        "requestOut",
        "Sends request for a field to target object"
    );
    return &requestOut;
}

static DestFinfo *handleInput()
{
    static DestFinfo input(
        "input",
        "Fills data into table. Also handles data sent back following request",
        new OpFunc1< Table, double >( &Table::input )
    );
    return &input;
}

const Cinfo* Table::initCinfo()
{
    //////////////////////////////////////////////////////////////
    // Field Definitions
    //////////////////////////////////////////////////////////////
    static ValueFinfo< Table, double > threshold(
        "threshold"
        , "threshold used when Table acts as a buffer for spikes"
        , &Table::setThreshold
        , &Table::getThreshold
    );

    static ValueFinfo< Table, bool > useStreamer(
        "useStreamer"
        , "When set to true, write to a file instead writing in memory."
        " If `outfile` is not set, streamer writes to default path."
        , &Table::setUseStreamer
        , &Table::getUseStreamer
    );

    static ValueFinfo< Table, string > outfile(
        "outfile"
        , "Set the name of file to which data is written to. If set, "
        " streaming support is automatically enabled."
        , &Table::setOutfile
        , &Table::getOutfile
    );

    static ValueFinfo< Table, string > format(
        "format"
        , "Data format for table: default csv"
        , &Table::setFormat
        , &Table::getFormat
    );

    //////////////////////////////////////////////////////////////
    // MsgDest Definitions
    //////////////////////////////////////////////////////////////

    static DestFinfo spike(
        "spike",
        "Fills spike timings into the Table. Signal has to exceed thresh",
        new OpFunc1< Table, double >( &Table::spike )
    );

    static DestFinfo process(
        "process",
        "Handles process call, updates internal time stamp.",
        new ProcOpFunc< Table >( &Table::process )
    );

    static DestFinfo reinit(
        "reinit",
        "Handles reinit call.",
        new ProcOpFunc< Table >( &Table::reinit )
    );

    //////////////////////////////////////////////////////////////
    // SharedMsg Definitions
    //////////////////////////////////////////////////////////////
    static Finfo* procShared[] =
    {
        &process, &reinit
    };

    static SharedFinfo proc(
        "proc"
        , "Shared message for process and reinit"
        , procShared, sizeof( procShared ) / sizeof( const Finfo* )
    );

    //////////////////////////////////////////////////////////////
    // Field Element for the vector data
    // Use a limit of 2^20 entries for the tables, about 1 million.
    //////////////////////////////////////////////////////////////

    static Finfo* tableFinfos[] =
    {
        &threshold,		// Value
        &format,                // Value
        &outfile,               // Value 
        &useStreamer,           // Value
        handleInput(),		// DestFinfo
        &spike,			// DestFinfo
        requestOut(),		// SrcFinfo
        &proc,			// SharedFinfo
    };

    static string doc[] =
    {
        "Name", "Table",
        "Author", "Upi Bhalla",
        "Description",
        "Table for accumulating data values, or spike timings. "
        "Can either receive incoming doubles, or can explicitly "
        "request values from fields provided they are doubles. "
        "The latter mode of use is preferable if you wish to have "
        "independent control of how often you sample from the output "
        "variable. \n"
        "Typically used for storing simulation output into memory, or to file"
        " when stream is set to True \n"
        "There are two functionally identical variants of the Table "
        "class: Table and Table2. Their only difference is that the "
        "default scheduling of the Table (Clock Tick 8, dt = 0.1 ms ) "
        "makes it suitable for "
        "tracking electrical compartmental models of neurons and "
        "networks. \n"
        "Table2 (Clock Tick 18, dt = 1.0 s) is good for tracking "
        "biochemical signaling pathway outputs. \n"
        "These are just the default values and Tables can be assigned"
        " to any Clock Tick and timestep in the usual manner.",
    };

    static Dinfo< Table > dinfo;

    static Cinfo tableCinfo (
        "Table",
        TableBase::initCinfo(),
        tableFinfos,
        sizeof( tableFinfos ) / sizeof ( Finfo* ),
        &dinfo,
        doc,
        sizeof( doc ) / sizeof( string )
    );

    static string doc2[] =
    {
        doc[0], "Table2", doc[2], doc[3], doc[4], doc[5]
    };

    doc2[1] = "Table2";

    static Cinfo table2Cinfo (
        "Table2",
        TableBase::initCinfo(),
        tableFinfos,
        sizeof( tableFinfos ) / sizeof ( Finfo* ),
        &dinfo,
        doc2,
        sizeof( doc2 ) / sizeof( string )
    );

    return &tableCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* tableCinfo = Table::initCinfo();

Table::Table() : threshold_( 0.0 ) , lastTime_( 0.0 ) , input_( 0.0 ) 
{
    // Initialize the directory to which each table should stream.
    rootdir_ /= "_tables";

    // If this directory does not exists, craete it. Following takes care of it.
    boost::filesystem::create_directories( rootdir_ );
}

Table::~Table( )
{
    of_.close();
}

Table& Table::operator=( const Table& tab )
{
    return *this;
}

void Table::writeToOutfile( )
{
    // Just to be safe.
    if( ! useStreamer_ )
        return;

    for( auto v : vec() ) 
    {
        text_ += moose::global::toString( dt_ * numLines ) + delimiter_
             + moose::global::toString( v ) + '\n';
        numLines += 1;
    }
    of_ << text_; text_ = "";
    if( getVecSize() > 0) clearVec();
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Table::process( const Eref& e, ProcPtr p )
{
    lastTime_ = p->currTime;

    // Copy incoming data to ret and insert into vector.
    vector< double > ret;
    requestOut()->send( e, &ret );
    vec().insert( vec().end(), ret.begin(), ret.end() );

    /*  If we are streaming to a file, let's write to a file. And clean the
     *  vector.  
     */
    if( useStreamer_ )
    {
        writeToOutfile( );
        clearVec();
    }
}

void Table::reinit( const Eref& e, ProcPtr p )
{
    unsigned int numTick = e.element()->getTick();
    Clock* clk = reinterpret_cast<Clock*>(Id(1).eref().data());
    dt_ = clk->getTickDt( numTick );

    /** Create the default filepath for this table.  */
    if( useStreamer_ )
    {
        // If useStreamer is set then we need to construct the table path, if
        // not set by user.
        if( ! outfileIsSet )
            setOutfile( 
                    moose::global::createPosixPath( 
                        rootdir_.string() + e.id().path() + "." + format_ 
                        )
                    );

        // Create its root directory.
        BOOST_LOG_TRIVIAL( debug ) << "Creating directory " 
            << outfile_.parent_path();

        moose::global::createDirs( outfile_.parent_path() );

        // Open the stream to write to file.
        of_.open( outfile_.string(), ios::out );
        of_ << "time,value\n";

    }

    input_ = 0.0;
    vec().resize( 0 );
    lastTime_ = 0;

    vector< double > ret;
    requestOut()->send( e, &ret );
    vec().insert( vec().end(), ret.begin(), ret.end() );

    if( useStreamer_ )
        writeToOutfile( );
}

//////////////////////////////////////////////////////////////
// Used to handle direct messages into the table, or
// returned plot data from queried objects.
//////////////////////////////////////////////////////////////
void Table::input( double v )
{
    vec().push_back( v );
}

void Table::spike( double v )
{
    if ( v > threshold_ )
        vec().push_back( lastTime_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Table::setThreshold( double v )
{
    threshold_ = v;
}

double Table::getThreshold() const
{
    return threshold_;
}

// Set the format of table to which its data should be written.
void Table::setFormat( string format )
{
    format_ = format;
}

// Get the format of table to which it has to be written.
string Table::getFormat( void ) const
{
    return format_;
}

/* Enable/disable streamer support. */
void Table::setUseStreamer( bool useStreamer )
{
    useStreamer_ = useStreamer;
}

bool Table::getUseStreamer( void ) const
{
    return useStreamer_;
}

/*  set/get outfile_ */
void Table::setOutfile( string outpath )
{
    outfile_ /= moose::global::createPosixPath( outpath );
    outfile_.make_preferred();
    outfileIsSet = true;
    setUseStreamer( true );
}

string Table::getOutfile( void ) const
{
    return outfile_.string();
}


