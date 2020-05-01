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

using namespace std;

/**
 * Receives and records inputs. Handles plot and spiking data in batch mode.
 */
class Table: public TableBase
{
public:
    Table();
    ~Table();

    Table& operator= ( const Table& tab );

    //////////////////////////////////////////////////////////////////
    // Field assignment stuff
    //////////////////////////////////////////////////////////////////

    void setThreshold ( double v );
    double getThreshold() const;

    void setFormat ( const string format );
    string getFormat( ) const;

    void setColumnName( const string colname );
    string getColumnName( ) const;

    void setUseStreamer ( bool status );
    bool getUseStreamer ( void ) const;

    void setUseSpikeMode ( bool status );
    bool getUseSpikeMode ( void ) const;

    void setDatafile ( string filepath );
    string getDatafile ( void ) const;

    // Access the dt_ of table.
    double getDt ( void ) const;

    // merge time value among values. e.g. t1, v1, t2, v2, etc.
    void mergeWithTime( vector<double>& data );

    string toJSON(bool withTime=true, bool clear = false);

    void collectData(vector<double>& data, bool withTime=true, bool clear = false);


    void clearAllVecs();

    //////////////////////////////////////////////////////////////////
    // Dest funcs
    //////////////////////////////////////////////////////////////////

    void process ( const Eref& e, ProcPtr p );
    void reinit ( const Eref& e, ProcPtr p );

    void input ( double v );
    void spike ( double v );

    //////////////////////////////////////////////////////////////////
    // Lookup funcs for table
    //////////////////////////////////////////////////////////////////

    static const Cinfo* initCinfo();

private:

    double threshold_;
    double lastTime_;
    double input_;
    bool fired_;
    bool useSpikeMode_;

    vector<double> data_;
    vector<double> tvec_;                       /* time data */

    // A table have 2 columns. First is time. We initialize this in reinit().
    vector<string> columns_; 

    /**
     * @brief dt of its clock. Needed for creating time co-ordinates,
     */
    double dt_;

    // Upto which indices we have read the data. This variable is used when
    // SocketStreamer is used.
    unsigned int lastN_ = 0;

    string tablePath_;

    /**
     * @brief Column name of this table. Use it when writing data to a datafile.
     */
    string tableColumnName_;

    /**
     * @brief If stream is set to true, then stream to datafile_. Default value
     * of datafile_ is table path starting from `pwd`/_tables_ . On table, set
     * streamToFile to true.
     */
    bool useFileStreamer_;

    // On Table, set datafile to change this variable. By default it sets to,
    // `pwd1/_tables_/table.path().
    string datafile_;

    /**
     * @brief format of data. Default to csv.
     */
    string format_;

};

#endif	// _TABLE_H
