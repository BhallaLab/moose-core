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
		Interpol( unsigned long xdivs, double xmin, double xmax );

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static void setXmin( const Conn& c, double value );
		static double getXmin( const Element* e );
		static void setXmax( const Conn& c, double value );
		static double getXmax( const Element* e );
		static void setXdivs( const Conn& c, int value );
		static int getXdivs( const Element* e );
		static void setDx( const Conn& c, double value );
		static double getDx( const Element* e );
		static void setSy( const Conn& c, double value );
		static double getSy( const Element* e );
		static void setMode( const Conn& c, int value );
		static int getMode( const Element* e );

		static void setTable(
					const Conn& c, double val, const unsigned int& i );
		static double getTable(
					const Element* e,const unsigned int& i );

		////////////////////////////////////////////////////////////
		// Here are the Interpol Destination functions
		////////////////////////////////////////////////////////////
		static void lookupReturn( const Conn& c, double val );
		static void lookup( const Conn& c, double val );
		static void tabFill( const Conn& c, int xdivs, int mode );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		double interpolateWithoutCheck( double x ) const;
		double indexWithoutCheck( double x ) const {
			return table_[ static_cast< int >( (x - xmin_) * invDx_) ];
		}
		double innerLookup( double x ) const;
		bool operator==( const Interpol& other ) const;
		bool operator<( const Interpol& other ) const;
		void localSetXmin( double value );
		void localSetXmax( double value );
		void localSetXdivs( int value );
		// Later do interpolation etc to preserve contents.
		void localSetDx( double value );
		double localGetDx() const;
		void localSetSy( double value );

		void setTableValue( double value, unsigned int index );
		double getTableValue( unsigned int index ) const;
		void push_back( double value ) {
			table_.push_back( value );
		}

		/**
		 * Expand out the table, using the specified mode.  
		 * Mode 0 : Linear interpolation for fill 
		 * Mode 1 : Splines (Not yet implemented. )
		 */
		void innerTabFill( int xdivs, int mode );

	private:
		double xmin_;
		double xmax_;
		int mode_;
		double invDx_;
		double sy_;
		vector < double > table_;
		static const double EPSILON;
		static const int MAX_DIVS;
};

extern const Cinfo* initInterpolCinfo();

#endif // _Interpol_h
