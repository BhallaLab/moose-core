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

#define STRINGSTREAM_DOUBLE_PRECISION       10

#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

#include "StreamerBase.h"
#include "Table.h"

using namespace std;

class Streamer : private StreamerBase
{

public:
    Streamer();
    ~Streamer();

    Streamer& operator=( const Streamer& st );

    string getOutFilepath( void ) const;
    void setOutFilepath( string path );

    string getFormat( void ) const;
    void setFormat( string format );

    void initOutfile( const Eref& e );
    void writeTablesToOutfile( void );

    size_t getNumTables( void ) const;

    void write( string& text );

    /*-----------------------------------------------------------------------------
     *  Following function adds or remove a table from vector of table tables_
     *-----------------------------------------------------------------------------*/
    void addTable( Id table );
    void addTables( vector<Id> tables);

    void removeTable( Id table );
    void removeTables( vector<Id> table );

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

    ofstream of_;
    string outfilePath_;
    string text_ = "";
    string delimiter_= ",";
    string format_ = "csv";

    // dt_ of its clock
    vector<double> tableDt_;
    double dt_;

    // No of lines written.
    size_t numLinesWritten_ = 0;

    // Total tables handled by this class.
    unsigned int numTables_ = 0;

    // These Tables are handled by StreamerBase 
    map< Id, Table* > tables_;
};

#endif   /* ----- #ifndef Streamer_INC  ----- */
