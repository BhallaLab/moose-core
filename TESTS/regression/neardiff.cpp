/**********************************************************************
** This program compares two files with numbers in xplot format.
** It allows for a difference EPSILON between corresponding numbers.
**           Copyright (C) 2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <math.h>

const int TAB_SIZE = 1000;

using namespace std;

class Plot {
		public:
				Plot()
				{
						x.reserve(TAB_SIZE);
						y.reserve(TAB_SIZE);
						parmname = "";
				}

				friend istream& operator>>(istream& s, Plot& p);

				bool operator< (const Plot& other) const ;

				bool IsInited() {
						return (x.size() > 0);
				}

				const string& name() const {
						return parmname;
				}

				bool differs( const Plot* other, double epsilon ) const;

				bool empty() const {
					return ( y.size() == 0 );
				}
				double max() const {
					return *max_element( y.begin(), y.end() );
				}

		private:
			string parmname;
			vector<double> x;
			vector<double> y;
};

bool Plot::differs( const Plot* other, double epsilon ) const
{
	unsigned int mysize = y.size();
	unsigned int othersize = other->y.size();
	
	// Number of points often differ, so leaving out this message.
	//~ if ( mysize != othersize )
		//~ cerr << "Warning: Number of points differ in " << parmname
			//~ << ". Ignoring extra points in larger plot.\n";

	unsigned int minsize = mysize < othersize ? mysize : othersize;

	for ( unsigned int i = 0; i < minsize; i++ )
		if ( fabs( y[ i ] - other->y[ i ] ) > epsilon )
			return 1;

	return 0;
}

istream& operator>>(istream& s, Plot& p)
{
	string n = "";
	double x, y;

	while((n != "/plotname") && (s >> n))
			;

	s >> n;
	p.parmname = n;

	while (s >> y) {
			p.y.push_back(y);
	}
	// s.clear();
	// s.clear(ios_base::badbit); // clear error state
	if (!s.eof() || p.y.size() > 0 )
			s.clear();
	cerr << ".";
	return s;
}

Plot* findPlot( vector< Plot* >& p, const string& name )
{
	vector< Plot* >::iterator i;
	for ( i = p.begin(); i != p.end(); i++ )
		if ( (*i)->name() == name )
			return *i;
	return 0;
}

int main( int argc, char** argv )
{
		if (argc < 4) {
				cerr << "Usage: " << argv[0] << " file1 file2 epsilon [-fraction_of_peak]\n";
				return 0;
		}
		
		fstream f0( argv[ 1 ] );
		if ( ! f0.good() ) {
			cerr << "Error: Unable to open file " << argv[ 1 ] << ".\n";
			return 1;
		}
		
		fstream f1( argv[ 2 ] );
		if ( ! f1.good() ) {
			cerr << "Error: Unable to open file " << argv[ 2 ] << ".\n";
			return 1;
		}
		
		double EPSILON = atof( argv[ 3 ] );
		if ( EPSILON <= 0.0 ) {
			cerr << "Error: epsilon must be positive.\n";
			return 1;
		}
		
		bool useFrac = 
			(argc == 5 && argv[4][0] == '-' && argv[4][1] == 'f' );

		vector< Plot* > p0;
		vector< Plot* > p1;
		Plot* p;

		for ( p = new Plot(); f0 >> *p; p = new Plot )
			p0.push_back( p );

		for ( p = new Plot(); f1 >> *p; p = new Plot )
			p1.push_back( p );
		
		if ( p0.size() != p1.size() ) {
			cout << argv[1] << ": diff # of plots\n";
			return 0;
		}

		if ( p0.size() == 0 ) {
			cout << argv[1] << ": empty plots\n";
			return 0;
		}
		
		if ( p0[0]->empty() || p1[0]->empty() ) {
			cout << argv[1] << ": empty plots\n";
			return 0;
		}

		// Go through all plots in p0 and compare with matching plots in p1
		// If any point differs by more than EPSILON, complain.
		vector< Plot* >::iterator i;
		for ( i = p0.begin(); i != p0.end(); i++ ) {
			double eps = EPSILON;
			if ( useFrac ) {
				eps = EPSILON * (*i)->max();
			}
			Plot* temp = findPlot( p1, ( *i )->name() );
			if ( !temp ) {
				cout << argv[1] << ": " << ( *i )->name() << 
					": no matching plotname\n";
				return 0;
			}
			if ( ( *i )->differs( temp, eps ) ) {
				cout << argv[1] << ": " << ( *i )->name() << 
					": plot differs\n";
				return 0;
			}
		}
		cout << ".";
}
