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


#include "global.h"
#include "header.h"
#include "StreamerBase.h"

#include "../scheduling/Clock.h"

#include <algorithm>
#include <sstream>

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
    // Before closing the stream, write the left-over of stringstream to ss_
    write( text_ );
    of_.close( );
}


/**
 * @brief Write  text to of_. Clean the text.
 * FIXME: Currently only csv is supported.
 */
void StreamerBase::write( string& text )
{
    of_ << text;
    text = "";
}


void StreamerBase::initOutfile( const Eref& e )
{
    if( ! of_.is_open() )
        std::cerr << "Warn: Could not open file " << outfilePath_
                  << ". I am going to write to 'tables.dat'. "
                  << endl;

    // Now write header to this file. First column is always time
    text_ = "time" + delimiter_ + "value\n";
    // Write to stream.
    write( text_ );

    // Initialize the clock and it dt.
    int numTick = e.element()->getTick();
    Clock* clk = reinterpret_cast<Clock*>(Id(1).eref().data());
    dt_ = clk->getTickDt( numTick );
}


string StreamerBase::getOutFilepath( void ) const
{
    return outfilePath_;
}

void StreamerBase::setOutFilepath( string filepath )
{
    outfilePath_ = filepath;
}

void StreamerBase::setFormat( string format )
{
    format_ = format;
}

string StreamerBase::getFormat( void ) const
{
    return format_;
}
