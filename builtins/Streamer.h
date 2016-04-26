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

#include "TableBase.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

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

    /* Dest functions */
    /**
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
    vector< Id > tables_;
};

#endif   /* ----- #ifndef Streamer_INC  ----- */
