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


#include "header.h"
#include "Streamer.h"
#include <algorithm>

const Cinfo* Streamer::initCinfo()
{
    /*-----------------------------------------------------------------------------
     * Finfos
     *-----------------------------------------------------------------------------*/
    static ValueFinfo< Streamer, string > streamname(
        "streamname"
        , "File/stream to write table data to. Default is 'stdout'."
        , &Streamer::setStreamname
        , &Streamer::getStreamname
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

    static DestFinfo removeTable(
        "removeTable"
        , "Remove a table from Streamer"
        , new OpFunc1<Streamer, Id>( &Streamer::removeTable )
    );

    /*-----------------------------------------------------------------------------
     *  ShareMsg definitions.
     *-----------------------------------------------------------------------------*/
    static Finfo* procShared[] =
    {
        &process , &reinit , &addTable, &removeTable
    };

    static SharedFinfo proc(
        "proc",
        "Shared message for process and reinit",
        procShared, sizeof( procShared ) / sizeof( const Finfo* )
    );

    static Finfo * tableStreamFinfos[] =
    {
        &streamname,
        &proc,
        &numTables,
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

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Streamer::Streamer() : streamname_(""), os_( &std::cout )
{
}

Streamer::~Streamer()
{
    delete os_;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

string Streamer::getStreamname() const
{
    return streamname_;
}

void Streamer::setStreamname( string streamname )
{
    streamname_ = streamname;
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
 * @brief Get the number of tables handled by Streamer.
 *
 * @return  Number of tables.
 */
size_t Streamer::getNumTables( void ) const
{
    return tables_.size();
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
    if( streamname_.size() > 0 )
    {
        std::ofstream* f = new std::ofstream( streamname_ );
        if( ! f->is_open() )
            os_ = f;
        else
        {
            std::cerr << "Warn: Could not open file " << streamname_ 
                << ". I am going to write to stdout. " << endl;
        }
    }

    // Now write header to this file. First column is always time
    *os_ << "time(seconds),";
    for( auto t : tables_ )
        *os_ << t.first.path() << "," << endl;
    *os_ << endl;
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
        if( dataSize[i] < criticalSize_ )
            return;

        data[i] = tab.second->getVec();
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
        for (size_t ii = 0; ii < getNumTables(); ii++)
            *os_ << data[ii][i] << ",";
        *os_ << endl;
    }

    for( auto t : tables_ )
        t.second->clearVec();
}
