/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TABLE_H
#define _TABLE_H

#include <boost/filesystem.hpp>

/**
 * Receives and records inputs. Handles plot and spiking data in batch mode.
 */
class Table: public TableBase
{
public:
    Table();
    ~Table();

    Table& operator=( const Table& tab );

    //////////////////////////////////////////////////////////////////
    // Field assignment stuff
    //////////////////////////////////////////////////////////////////

    void setThreshold( double v );
    double getThreshold() const;

    void setFormat( string format );
    string getFormat( ) const;

    void setUseStreamer( bool status );
    bool getUseStreamer( void ) const;

    void setOutfile( string outfilepath );
    string getOutfile( void ) const;

    void writeToOutfile( );

    //////////////////////////////////////////////////////////////////
    // Dest funcs
    //////////////////////////////////////////////////////////////////

    void process( const Eref& e, ProcPtr p );
    void reinit( const Eref& e, ProcPtr p );

    void input( double v );
    void spike( double v );

    //////////////////////////////////////////////////////////////////
    // Lookup funcs for table
    //////////////////////////////////////////////////////////////////

    static const Cinfo* initCinfo();

private:
    double threshold_;
    double lastTime_;
    double input_;

    /**
     * @brief If stream is set to true, then stream to outfile_. Default value
     * of outfile_ is table path starting from `pwd`/_tables_ . On table, set
     * streamToFile to true.
     */
    bool useStreamer_ = false;

    /**
     * @brief Table directory into which dump the stream data.
     */
    boost::filesystem::path rootdir_;

    // On Table, set outfile to change this variable. By default it sets to,
    // `pwd1/_tables_/table.path().
    boost::filesystem::path outfile_;
    bool outfileIsSet = false;

    /**
     * @brief format of data. Currently fixed to csv.
     */
    string format_ = "csv";
    string delimiter_ = ",";

    /**
     * @brief text_ to write.
     */
    string text_ = "";

    /**
     * @brief dt of its clock. Needed for creating time co-ordinates,
     */
    double dt_ = 0.0;
    size_t numLines = 0;

    /**
     * @brief Output stream.
     */
    ofstream of_;

};

#endif	// _TABLE_H
