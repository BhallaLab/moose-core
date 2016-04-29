/***
 *       Filename:  StreamerBaseBase.h
 *
 *    Description:  Stream table data to a  stream.
 *
 *        Created:  2016-04-26
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */

#ifndef  StreamerBase_INC
#define  StreamerBase_INC

#define STRINGSTREAM_DOUBLE_PRECISION       10

#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

#include "TableBase.h"

using namespace std;

class TableBase;

class StreamerBase : public TableBase
{

public:
    StreamerBase();
    ~StreamerBase();

    StreamerBase& operator=( const StreamerBase& st );

    /* Functions to set and get Streamer fields */
    void setOutFilepath( string stream );
    string getOutFilepath() const;

    /*  To set and get format names.  */
    void setFormat( string formatname );
    string getFormat( ) const;


    // Write given text to output file. Clear the text after writing it.
    void write( string& text );

    // Initialize output file
    void initOutfile( const Eref& e );


private:

    string outfilePath_;

    // format and delimiter 
    string format_ = "csv";
    string delimiter_ = ",";


    // How many lines are written to the file.
    size_t numLinesWritten_ = 0;

    // Write to file stream.
    std::ofstream of_;

    // Temporary storage to lines
    std::string text_;

    /*  Step size of this class */
    double dt_;
};

#endif   /* ----- #ifndef StreamerBase_INC  ----- */
