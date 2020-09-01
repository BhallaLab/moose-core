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

class Streamer : public StreamerBase
{

public:
    Streamer();
    ~Streamer();

    Streamer& operator=( const Streamer& st );

    /* Cleaup before quitting */
    void cleanUp( void );

    string getDatafilePath( void ) const;
    void setDatafilePath( string path );

    string getFormat( void ) const;
    void setFormat( string format );

    unsigned int getNumTables( void ) const;
    unsigned int getNumWriteEvents( void ) const;

    void addTable( ObjId table );
    void addTables( vector<ObjId> tables);

    void removeTable( ObjId table );
    void removeTables( vector<ObjId> table );

    void zipWithTime( );

    /** Dest functions.
     * The process function called by scheduler on every tick
     */
    void process(const Eref& e, ProcPtr p);

    /**
     * The reinit function called by scheduler for the reset
     */
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

private:

    string datafilePath_;
    string format_;

    unsigned int numWriteEvents_;

    bool isOutfilePathSet_;

    // dt_ and tick number of Table's clock
    vector<double> tableDt_;
    vector<unsigned int> tableTick_;

    // This currTime is not computed using the ProcPtr but rather using Tables
    // dt_ and number of entries written.
    double currTime_;

    // Used for adding or removing tables
    vector<ObjId> tableIds_;
    vector<Table*> tables_;
    vector<string> columns_;

    /*  Keep data in vector */
    vector<double> data_;

};

#endif   /* ----- #ifndef Streamer_INC  ----- */
