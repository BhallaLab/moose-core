/***
 *       Filename:  Streamer.h
 *
 *    Description:  Stream table data to a  stream.
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

#ifndef  Streamer_INC
#define  Streamer_INC

#include <iostream>
#include <string>
#include <map>
#include <fstream>

#include "TableBase.h"


using namespace std;

class TableBase;

class Streamer : public TableBase
{

public:
    Streamer();
    ~Streamer();

    /* Functions to set and get Streamer fields */
    void setStreamname( string stream );
    string getStreamname() const;

    /*-----------------------------------------------------------------------------
     *  Following function adds or remove a table from vector of table tables_
     *-----------------------------------------------------------------------------*/
    void addTable( Id table );
    void removeTable( Id table );

    size_t getNumTables( void ) const;

    /* Dest functions.
     * The process function called by scheduler on every tick
     */
    void process(const Eref& e, ProcPtr p);

    /**
     * The reinit function called by scheduler for the reset
     */
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

private:

    // Name of the stream to which to write table data.
    string streamname_;

    // These Tables are handled by Streamer 
    map< Id, TableBase* > tables_;

    size_t numTables_;

    /**
     * @brief If vector of all table has entries >= than this  number, dump to
     * given file and delete these elements.
     */
    size_t criticalSize_ = 1000;

    /**
     * @brief If header is already written to stdout/stream, set it true.
     */
    bool isHeaderWritten = false;

    size_t previousWriteIndex_ = 0;
    size_t currentWriteIndex_ = 0;

    // Output stream. Either assign to std::cout or to a file.
    std::ostream* os_;


};

#endif   /* ----- #ifndef Streamer_INC  ----- */
