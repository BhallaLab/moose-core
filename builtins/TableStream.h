/***
 *       Filename:  TableStream.h
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

#ifndef  TableStream_INC
#define  TableStream_INC

#include "TableBase.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class TableStream : public TableBase
{

public:
    TableStream();
    ~TableStream();

    /* Functions to set and get TableStream fields */
    void setStreamname( string stream );
    string getStreamname() const;

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

    /*
     * Fields
     */
    string streamname_;

    /* The table with (spike)times */
    vector < double > timeTable_;

};

#endif   /* ----- #ifndef TableStream_INC  ----- */
