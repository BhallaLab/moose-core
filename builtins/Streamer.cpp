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
#include "Table.h"


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

Streamer::Streamer() : streamname_("stdout"), of_( &std::cout )
{
}

Streamer::~Streamer()
{
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
    for( auto &t : tables_ )
        if( table.path() == t.path() )
            return;

    tables_.push_back( table );
}

/**
 * @brief Remove a table from Streamer.
 *
 * @param table. Id of table.
 */
void Streamer::removeTable( Id table )
{
    bool found = false;
    vector<Id>::iterator it = tables_.begin ();
    for( ; it != tables_.end(); it++)
        if( table.path() == it->path() )
        {
            found = true;
            break;
        }

    if( found )
    {
        tables_.erase( it );
    }
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
    if( streamname_ == "stdout" )
        of_ = &std::cout;
}

/**
 * @brief This function is called at its clock tick.
 *
 * @param e
 * @param p
 */
void Streamer::process(const Eref& e, ProcPtr p)
{
    if( tables_.size() < 1 )
        return;

    cout << "Total tables: " << tables_.size() << endl;

    for( auto t : tables_ )
    {
        Table* tt  = reinterpret_cast< Table* >( t.eref().data() );
        cout << "Id : " << t << endl;
        cout << tt->getVecSize() << endl;
    }
}
