#ifndef _Interpol_h
#define _Interpol_h

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Interpol
{
	public:
		Interpol()
		{
			xmin_ = 0.0;
			xmax_ = 1.0;
			mode_ = 1; // Mode 1 is linear interpolation. 0 is indexing.
			invDx_ = 1.0;
			sy_ = 1.0;
			table_.resize( 2, 0.0 );
		}
		virtual ~Interpol() { ; }

		Interpol( unsigned long xdivs, double xmin, double xmax );

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static void setXmin( const Conn* c, double value );
		static double getXmin( Eref e );
		static void setXmax( const Conn* c, double value );
		static double getXmax( Eref e );
		static void setXdivs( const Conn* c, int value );
		static int getXdivs( Eref e );
		static void setDx( const Conn* c, double value );
		static double getDx( Eref e );
		static void setSy( const Conn* c, double value );
		static double getSy( Eref e );
		static void setMode( const Conn* c, int value );
		static int getMode( Eref e );

		static void setTable(
					const Conn* c, double val, const unsigned int& i );
		static double getTable(
					Eref e,const unsigned int& i );
		static void setTableVector( const Conn* c, vector< double > value );
		static vector< double > getTableVector( Eref e );

		////////////////////////////////////////////////////////////
		// Here are the Interpol Destination functions
		////////////////////////////////////////////////////////////
		static void lookupReturn( const Conn* c, double val );
		static void lookup( const Conn* c, double val );
		static void tabFill( const Conn* c, int xdivs, int mode );
		static void print( const Conn* c, string fname );
		static void append( const Conn* c, string fname );
		static void load( const Conn* c, string fname,
			unsigned int skiplines );
		static void appendTableVector( const Conn* c, 
			vector< double > value );

		static void push( const Conn* c, double value );
		static void clear( const Conn* c );
		static void pop( const Conn* c );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		double interpolateWithoutCheck( double x ) const;
		double indexWithoutCheck( double x ) const {
			return table_[ static_cast< int >( (x - xmin_) * invDx_ ) ];
		}
		double innerLookup( double x ) const;
		bool operator==( const Interpol& other ) const;
		bool operator<( const Interpol& other ) const;
		void localSetXmin( double value );
		void localSetXmax( double value );
		virtual void localSetXdivs( int value );
		virtual int localGetXdivs( ) const;
		/// \todo Later do interpolation etc to preserve contents.
		void localSetDx( double value );
		double localGetDx() const;
		double invDx() const {
			return invDx_;
		}
		virtual void localSetSy( double value );

		void setTableValue( double value, unsigned int index );
		double getTableValue( unsigned int index );
		void localSetTableVector( const vector< double >& value );

		void innerPush( double value );
		void innerClear();
		void innerPop();

		void localAppendTableVector( const vector< double >& value );
		unsigned long size( ) const {
			return table_.size();
		}
		void resize( unsigned int size, double init = 0.0 ) {
			table_.resize( size, init );
		}
		void push_back( double value ) {
			table_.push_back( value );
		}

		double xmin() const {
			return xmin_;
		}

		double xmax() const {
			return xmax_;
		}

		virtual int xdivs() const {
			if ( table_.empty() )
				return 0;
			
			return table_.size() - 1;
		}

		int mode() const {
			return mode_;
		}

		/**
		 * Expand out the table, using the specified mode.  
		 * Mode 0 : B-Splines
		 * Mode 2 : Linear interpolation for fill 
		 */
		virtual void innerTabFill( int xdivs, int mode );
		virtual void innerPrint( const string& fname, bool doAppend ) const;
		virtual void innerLoad( const string& fname, unsigned int skiplines );

	protected:
		double xmin_;
		double xmax_;
		double invDx_;
		int mode_;
		double sy_;
		vector < double > table_;

		static const double EPSILON;
		static const unsigned int MAX_DIVS;
};

extern const Cinfo* initInterpolCinfo();

#endif // _Interpol_h
