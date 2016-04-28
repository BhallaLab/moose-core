/***
 *       Filename:  Streamer.cpp
 *
 *    Description:  Stream table data.
 *
 *        Version:  0.0.1
 *        Created:  2016-04-26

 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */


#include "global.h"
#include "header.h"
#include "Streamer.h"
#include "../scheduling/Clock.h"

#include <algorithm>
#include <sstream>


const Cinfo* Streamer::initCinfo()
{
    /*-----------------------------------------------------------------------------
     * Finfos
     *-----------------------------------------------------------------------------*/
    static ValueFinfo< Streamer, string > outfile(
        "outfile"
        , "File/stream to write table data to. Default is 'stdout'."
        , &Streamer::setOutFilename
        , &Streamer::getOutFilename
    );

    static ValueFinfo< Streamer, string > format(
        "format"
        , "Format of output file, default is csv"
        , &Streamer::setFormat
        , &Streamer::getFormat
    );

    static ReadOnlyValueFinfo< Streamer, size_t > numTables (
        "numTables"
        , "Number of Tables handled by Streamer "
        , &Streamer::getNumTables
    );

    /*-----------------------------------------------------------------------------
     *
     *-----------------------------------------------------------------------------*/
    static DestFinfo process(
        "process"
        , "Handle process call"
        , new ProcOpFunc< Streamer >( &Streamer::process )
    );

    static DestFinfo reinit(
        "reinit"
        , "Handles reinit call"
        , new ProcOpFunc< Streamer > ( &Streamer::reinit )
    );


    static DestFinfo addTable(
        "addTable"
        , "Add a table to Streamer"
        , new OpFunc1<Streamer, Id>( &Streamer::addTable )
    );

    static DestFinfo addTables(
        "addTables"
        , "Add many tables to Streamer"
        , new OpFunc1<Streamer, vector<Id> >( &Streamer::addTables )
    );

    static DestFinfo removeTable(
        "removeTable"
        , "Remove a table from Streamer"
        , new OpFunc1<Streamer, Id>( &Streamer::removeTable )
    );

    static DestFinfo removeTables(
        "removeTables"
        , "Remove tables -- if found -- from Streamer"
        , new OpFunc1<Streamer, vector<Id> >( &Streamer::removeTables )
    );

    /*-----------------------------------------------------------------------------
     *  ShareMsg definitions.
     *-----------------------------------------------------------------------------*/
    static Finfo* procShared[] =
    {
        &process , &reinit , &addTable, &addTables, &removeTable, &removeTables
    };

    static SharedFinfo proc(
        "proc",
        "Shared message for process and reinit",
        procShared, sizeof( procShared ) / sizeof( const Finfo* )
    );

    static Finfo * tableStreamFinfos[] =
    {
        &outfile, &format, &proc, &numTables
    };

    static string doc[] =
    {
        "Name", "Streamer",
        "Author", "Dilawar Singh, 2016, NCBS, Bangalore.",
        "Description", "Streamer: Stream moose.Table data to out-streams\n"
    };

    static Dinfo< Streamer > dinfo;

    static Cinfo tableStreamCinfo(
        "Streamer",
        TableBase::initCinfo(),
        tableStreamFinfos,
        sizeof( tableStreamFinfos )/sizeof(Finfo *),
        &dinfo,
        doc,
        sizeof(doc) / sizeof(string)
    );

    return &tableStreamCinfo;
}

static const Cinfo* tableStreamCinfo = Streamer::initCinfo();

// Class function definitions

Streamer::Streamer() : outfile_(""), format_( "csv" )
{
    ss_.precision( STRINGSTREAM_DOUBLE_PRECISION );
}

Streamer& Streamer::operator=( const Streamer& st )
{
    this->outfile_ = st.outfile_;
    return *this;
}


Streamer::~Streamer()
{
    // Before closing the stream, write the left-over of stringstream to ss_
    of_ << ss_.str();
    ss_.str( "" );
    of_.close( );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string Streamer::getOutFilename() const
{
    return outfile_;
}

void Streamer::setOutFilename( string filename )
{
    outfile_ = filename;
}


string Streamer::getFormat( ) const
{
    return format_;
}

void Streamer::setFormat( string format )
{
    cout << "Lazy developer is yet is to support it. Currently only csv is supported"
        << endl;
    format_ = "csv";
}

/**
 * @brief Add a table to streamer.
 *
 * @param table Id of table.
 */
void Streamer::addTable( Id table )
{
    // If this table is not already in the vector, add it.
    for( auto t : tables_ )
        if( table.path() == t.first.path() )
            return;                             /* Already added. */

    TableBase* t = reinterpret_cast<TableBase*>(table.eref().data());
    tables_[ table ] = t;
}

/**
 * @brief Add multiple tables to Streamer.
 *
 * @param tables
 */
void Streamer::addTables( vector<Id> tables )
{
    for( auto t : tables ) addTable( t );
}


/**
 * @brief Remove a table from Streamer.
 *
 * @param table. Id of table.
 */
void Streamer::removeTable( Id table )
{
    auto it = tables_.find( table );
    if( it != tables_.end() )
        tables_.erase( it );
}

/**
 * @brief Remove multiple tables -- if found -- from Streamer.
 *
 * @param tables
 */
void Streamer::removeTables( vector<Id> tables )
{
    for( auto t : tables ) removeTable( t );
}

/**
 * @brief Get the number of tables handled by Streamer.
 *
 * @return  Number of tables.
 */
size_t Streamer::getNumTables( void ) const
{
    return tables_.size();
}

/**
 * @brief Write  text to of_. Clean the text.
 * FIXME: Currently only csv is supported.
 */
void Streamer::write( string& text )
{
    of_ << text;
    text = "";
}

/**
 * @brief Reinit.
 *
 * @param e
 * @param p
 */
void Streamer::reinit(const Eref& e, ProcPtr p)
{
    // If it is not stdout, then open a file and write standard header to it.
    if( outfile_.size() < 1 )
        outfile_ = "tables.dat";

    of_.open( outfile_, ios::out );

    if( ! of_.is_open() )
        std::cerr << "Warn: Could not open file " << outfile_
                  << ". I am going to write to 'tables.dat'. "
                  << endl;

    // Now write header to this file. First column is always time
    text_ = "time" + delimiter_;
    for( auto t : tables_ )
        text_ += t.first.path() + delimiter_;
    // Remove the last command and add newline.
    text_.pop_back(); text_ += '\n';

    // Write to stream.
    write( text_ );

    // Initialize the clock and it dt.
    int numTick = e.element()->getTick();
    Clock* clk = reinterpret_cast<Clock*>(Id(1).eref().data());
    dt_ = clk->getTickDt( numTick );
}

/**
 * @brief This function is called at its clock tick.
 *
 * @param e
 * @param p
 */
void Streamer::process(const Eref& e, ProcPtr p)
{

    if( tables_.size() <= 0 )
        return;

    vector<vector<double> > data( tables_.size() );
    vector<size_t> dataSize( tables_.size() );

    size_t i = 0;
    for( auto tab : tables_ )
    {
        dataSize[i] = tab.second->getVecSize();

        // If any table has fewer data points then the threshold for writing to
        // file then return without doing anything.
        data[i] = tab.second->getVec();

        // Clear the data from vector
        tab.second->clearVec();

        i++;
    }

    if( std::min_element( dataSize.begin(), dataSize.end() ) !=
            std::max_element( dataSize.begin(), dataSize.end() )
      )
    {
        cout << "WARNING: One or more tables handled by this Streamer are collecting "
             << "data at different rate than others. I'll continue dumping data to "
             << "stream/file but it will get corrupted. I'll advise you to delete  "
             << "such tables."
             << endl;
    }

    // All vectors must be of same size otherwise we are in trouble.
    for (size_t i = 0; i < dataSize[0]; i++)
    {
        text_ += moose::global::to_string<double>(dt_ * numLinesWritten_) + delimiter_;
        for (size_t ii = 0; ii < data.size(); ii++)
            text_ += moose::global::to_string<double>(data[ii][i]) + delimiter_;
        // Remove last "," and append a new line.
        text_.pop_back(); text_ += '\n';
        numLinesWritten_ += 1;
    }

    write( text_ );

}
