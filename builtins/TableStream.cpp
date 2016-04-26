/***
 *       Filename:  TableStream.cpp
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
#include "TableStream.h"

const Cinfo* TableStream::initCinfo()
{

    /*-----------------------------------------------------------------------------
     * Finfos
     *-----------------------------------------------------------------------------*/
    static ValueFinfo< TableStream, string > streamname(
        "streamname",
        "File/stream to write table data to. Default is 'stdout'.",
        &TableStream::setStreamname,
        &TableStream::getStreamname
    );


    /*-----------------------------------------------------------------------------
     *
     *-----------------------------------------------------------------------------*/
    static DestFinfo process(
        "process",
        "Handle process call",
        new ProcOpFunc< TableStream >( &TableStream::process)
    );

    static DestFinfo reinit(
        "reinit",
        "Handles reinit call",
        new ProcOpFunc< TableStream > ( &TableStream::reinit)
    );

    /*-----------------------------------------------------------------------------
     *  ShareMsg definitions.
     *-----------------------------------------------------------------------------*/
    static Finfo* procShared[] =
    {
        &process, &reinit
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
    };

    static string doc[] =
    {
        "Name", "TableStream",
        "Author", "Dilawar Singh, 2016, NCBS, Bangalore.",
        "Description", "TableStream: Stream moose.Table data to out-streams\n"
    };

    static Dinfo< TableStream > dinfo;

    static Cinfo tableStreamCinfo(
        "TableStream",
        TableBase::initCinfo(),
        tableStreamFinfos,
        sizeof( tableStreamFinfos )/sizeof(Finfo *),
        &dinfo,
        doc,
        sizeof(doc) / sizeof(string)
    );

    return &tableStreamCinfo;
}

static const Cinfo* tableStreamCinfo = TableStream::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

TableStream::TableStream() : streamname_("stdout")
{
    ;
}

TableStream::~TableStream()
{
    ;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/* Filename */
string TableStream::getStreamname() const
{
    return streamname_;
}

void TableStream::setStreamname( string streamname )
{

}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void TableStream::reinit(const Eref& e, ProcPtr p)
{
}

void TableStream::process(const Eref& e, ProcPtr p)
{

}
