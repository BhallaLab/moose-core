#ifndef _Interpol_h
#define _Interpol_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
class Interpol
{
	friend class InterpolWrapper;
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

		double localGetXmin() const {
			return xmin_;
		}
		double localGetXmax()const {
			return xmax_;
		}
		int localGetXdivs() const {
			return table_.size() - 1;
		}
		double interpolateWithoutCheck( double x ) const;
		double indexWithoutCheck( double x ) const {
			return table_[ static_cast< int >( (x - xmin_) * invDx_) ];
		}
		double doLookup( double x ) const;
		bool operator==( const Interpol& other ) const;
		bool operator<( const Interpol& other ) const;
		void localSetXmin( double value );
		void localSetXmax( double value );
		void localSetXdivs( int value );
		// Later do interpolation etc to preseve contents.
		void localSetDx( double value );
		double localGetDx() const;
		void localSetSy( double value );
		double localGetSy() const {
			return sy_;
		}
		void setTableValue( double value, int index );
		double getTableValue( int index ) const;
		void push_back( double value ) {
			table_.push_back( value );
		}

		void localSetMode( int mode ) {
			mode_ = mode;
		}
		int localGetMode( ) const {
			return mode_;
		}
		// Expand out the table, using the specified mode.
		// Mode 0 : Linear interpolation for fill
		// Mode 1 : Splines (Not yet implemented. )
		void tabFill( int xdivs, int mode );

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

#endif // _Interpol_h
