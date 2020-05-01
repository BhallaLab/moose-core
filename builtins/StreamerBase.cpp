/***
 *       Filename:  StreamerBase.cpp
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


#include "../basecode/global.h"
#include "../basecode/header.h"
#include "StreamerBase.h"

#include "../scheduling/Clock.h"
#include "../utility/cnpy.hpp"

#include <algorithm>
#include <sstream>
#include <memory>

// Class function definitions
StreamerBase::StreamerBase()
{
}

StreamerBase& StreamerBase::operator=( const StreamerBase& st )
{
    this->outfilePath_ = st.outfilePath_;
    return *this;
}


StreamerBase::~StreamerBase()
{

}


string StreamerBase::getOutFilepath( void ) const
{
    return outfilePath_;
}

void StreamerBase::setOutFilepath( string filepath )
{
    outfilePath_ = filepath;
}


void StreamerBase::writeToOutFile( const string& filepath
        , const string& outputFormat
        , const OpenMode openmode
        , const vector<double>& data
        , const vector<string>& columns
        )
{
    if( data.size() == 0 )
        return;

    if("npy" == outputFormat  || "npz" == outputFormat)
    {
        OpenMode m = (openmode == WRITE)?WRITE_BIN:APPEND_BIN;
        writeToNPYFile( filepath, m, data, columns );
    }
    else if( "csv" == outputFormat or "dat" == outputFormat )
    {
        OpenMode m = (openmode == WRITE)?WRITE_STR:APPEND_STR;
        writeToCSVFile( filepath, m, data, columns );
    }
    else
    {
        LOG( moose::warning, "Unsupported format " << outputFormat
                << ". Use npy or csv. Falling back to default csv"
           );
        OpenMode m = (openmode == WRITE)?WRITE_STR:APPEND_STR;
        writeToCSVFile( filepath, m, data, columns );
    }
}

/*  Write to a csv file.  */
void StreamerBase::writeToCSVFile( const string& filepath, const OpenMode openmode
        , const vector<double>& data, const vector<string>& columns )
{
    string m = (openmode == WRITE_STR)?"w":"a";
    FILE* fp = fopen( filepath.c_str(), m.c_str());

    if( NULL == fp )
    {
        LOG( moose::warning, "Failed to open " << filepath );
        return;
    }

    // If writing in "w" mode, write the header first.
    if(openmode == WRITE_STR)
    {
        string headerText = "";
        for( vector<string>::const_iterator it = columns.begin();
            it != columns.end(); it++ )
            headerText += ( *it + delimiter_ );
        headerText += eol;
        fprintf(fp, "%s", headerText.c_str());
    }

    string text = "";
    for( unsigned int i = 0; i < data.size(); i+=columns.size() )
    {
        // Start of a new row.
        for( unsigned int ii = 0; ii < columns.size(); ii++ )
            text += moose::toString( data[i+ii] ) + delimiter_;

        // At the end of each row, we remove the delimiter_ and append newline_.
        *(text.end()-1) = eol;
    }
    fprintf(fp, "%s", text.c_str() );
    fclose(fp);
}

/*  write data to a numpy file */
void StreamerBase::writeToNPYFile( const string& filepath, const OpenMode openmode
        , const vector<double>& data, const vector<string>& columns )
{
    //for(auto v: data) cout << v << ' ';
    //cout << endl;

    if(openmode == APPEND_BIN)
        return cnpy2::appendNumpy( filepath, data, columns);

    if(openmode == WRITE_BIN)
        return cnpy2::writeNumpy( filepath, data, columns);
}

string StreamerBase::vectorToCSV( const vector<double>& ys, const string& fmt )
{
    string res{""};
    for( auto v : ys )
        res += std::to_string(v) + ",";
    return res;
}
