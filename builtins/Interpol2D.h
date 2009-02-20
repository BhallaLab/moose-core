/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Interpol2D_h
#define _Interpol2D_h

/**
 * 2 Dimensional table, with interpolation. The internal vector is accessed like
 * this: table_[ xIndex ][ yIndex ], with the x- and y-coordinates used as the
 * first and second indices respectively.
 */
class Interpol2D: public Interpol
{
	public:
		Interpol2D()
		{
			ymin_ = 0.0;
			ymax_ = 1.0;
			invDy_ = 1.0;
			
			Interpol::table_.clear( );
			table_.resize( 2 );
			table_[ 0 ].resize( 2, 0.0 );
			table_[ 1 ].resize( 2, 0.0 );
		}

		Interpol2D(
			unsigned long xdivs, double xmin, double xmax,
			unsigned long ydivs, double ymin, double ymax );

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static void setYmin( const Conn* c, double value );
		static double getYmin( Eref e );
		static void setYmax( const Conn* c, double value );
		static double getYmax( Eref e );
		static void setYdivs( const Conn* c, int value );
		static int getYdivs( Eref e );
		static void setDy( const Conn* c, double value );
		static double getDy( Eref e );

		/**
		 * These belong to Interpol, and should never be accessed.
		 */
		static void setTable( const Conn* c, double val, const unsigned int& i )
		{ assert( 0 ); }
		
		static double getTable( Eref e,const unsigned int& i )
		{ assert( 0 ); }

		static void setTableVector( const Conn* c, vector< double > value )
		{ assert( 0 ); }

		static vector< double > getTableVector( Eref e )
		{ assert( 0 ); }
		
		/**
		 * Interpol2D counterparts of similar Interpol functions. Note that
		 * the signatures are different in the 2 cases. The Finfo names must be
		 * new, so naming the functions again too. Also, the signature of
		 * getTableVector is same in both cases, so the 2 functions cannot
		 * be distinguished while creating a Finfo.
		 */
		static void setTable2D(
					const Conn* c,
					double val,
					const vector< unsigned int >& i );
		static double getTable2D(
					Eref e,
					const vector< unsigned int >& i );
		static void setTableVector2D(
					const Conn* c,
					vector< vector< double > > value );
		static vector< vector< double > > getTableVector2D( Eref e );

		////////////////////////////////////////////////////////////
		// Here are the Interpol2D Destination functions
		////////////////////////////////////////////////////////////
		static void lookupReturn( const Conn* c, double v1, double v2 );
		static void lookup( const Conn* c, double v1, double v2 );
		static void appendTableVector( const Conn* c, 
			vector< vector< double > > value );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		double interpolateWithoutCheck( double x, double y ) const;
		double indexWithoutCheck( double x, double y ) const;
		double innerLookup( double x, double y ) const;
		bool operator==( const Interpol2D& other ) const;
		bool operator<( const Interpol2D& other ) const;
		virtual void localSetXdivs( int value );
		virtual int localGetXdivs( ) const;
		void localSetYmin( double value );
		void localSetYmax( double value );
		void localSetYdivs( int value );
		int localGetYdivs( ) const;
		void localSetDy( double value );
		double localGetDy() const;
		double invDy() const {
			return invDy_;
		}
		virtual void localSetSy( double value );

		void setTableValue(
			double value,
			const vector< unsigned int >& index );
		double getTableValue(
			const vector< unsigned int >& index );
		void localSetTableVector(
			const vector< vector< double > >& value );
		void localAppendTableVector(
			const vector< vector< double > >& value );
		void resize( unsigned int xsize, unsigned int ysize, double init = 0.0  );

		virtual int xdivs() const {
			if ( Interpol2D::table_.empty() )
				return 0;
			
			return table_.size() - 1;
		}

		double ymin() const {
			return ymin_;
		}

		double ymax() const {
			return ymax_;
		}

		int ydivs() const {
			if ( table_.empty() || table_[ 0 ].empty() )
				return 0;
			
			return table_[ 0 ].size() - 1;
		}

		virtual void innerTabFill( int xdivs, int mode ) {
			cerr << "Error: Interpol2D::innerTabFill: " <<
				"This function belongs to Interpol, not Interpol2D.\n";
		}
		virtual void innerPrint( const string& fname, bool doAppend ) const;
		virtual void innerLoad( const string& fname, unsigned int skiplines );

	protected:
		double ymin_;
		double ymax_;
		double invDy_;
		vector< vector< double > > table_;
};

extern const Cinfo* initInterpol2DCinfo();

#endif // _Interpol2D_h
