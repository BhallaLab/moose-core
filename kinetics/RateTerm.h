/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/*
double assignRates( RateCalculation& r )
{
	return r();
}
*/

class RateTerm
{
	public:
		RateTerm() {;}
		virtual ~RateTerm() {;}
		virtual double operator() () const = 0;
};

// Single substrate MMEnzyme: by far the most common.
class MMEnzyme1: public RateTerm
{
	public:
		MMEnzyme1( double Km, double kcat, 
			const double* enz, const double* sub )
			: Km_( Km ), kcat_( kcat ), enz_( enz ), sub_( sub )
		{
			;
		}

		double operator() () const {
			return ( -kcat_ * *sub_ * *enz_ ) / ( Km_ + *sub_ );
		}
	private:
		double Km_;
		double kcat_;
		const double *enz_;
		const double *sub_;
};

class MMEnzyme: public RateTerm
{
	public:
		MMEnzyme( double Km, double kcat, 
			const double* enz, RateTerm* sub )
			: Km_( Km ), kcat_( kcat ), enz_( enz ), substrates_( sub )
		{
			;
		}

		double operator() () const {
			double sub = -(*substrates_)();
			// the subtrates_() operator returns the negative of 
			// the conc product.
			// Here we return negative of the overall rate.
			return ( -sub * kcat_ * *enz_ ) / ( Km_ + sub );
		}
	private:
		double Km_;
		double kcat_;
		const double *enz_;
		RateTerm* substrates_;
};

class ExternReac: public RateTerm
{
	public:
		// All the terms will have been updated separately, and
		// a reply obtained to this pointer here:
		double operator() () const {
			double ret = 0.0;
			return ret;
		}
	private:
};

class ZeroOrder: public RateTerm
{
	public:
		ZeroOrder( double k )
			: k_( -k )
		{;}

		double operator() () const {
			return k_;
		}

		void setK( double k ) {
			k_ = k;
		}
	protected:
		double k_;
};

class FirstOrder: public ZeroOrder
{
	public:
		FirstOrder( double k, const double* y )
			: ZeroOrder( k ), y_( y )
		{;}

		double operator() () const {
			return k_ * *y_;
		}

	private:
		const double *y_;
};

class SecondOrder: public ZeroOrder
{
	public:
		SecondOrder( double k, const double* y1, const double* y2 )
			: ZeroOrder( k ), y1_( y1 ), y2_( y2 )
		{;}

		double operator() () const {
			return k_ * *y1_ * *y2_;
		}

	private:
		const double *y1_;
		const double *y2_;
};

class NOrder: public ZeroOrder
{
	public:
		NOrder( double k, vector< const double* > v )
			: ZeroOrder( k ), v_( v )
		{;}

		double operator() () const {
			double ret = k_;
			vector< const double* >::const_iterator i;
			for ( i = v_.begin(); i != v_.end(); i++)
				ret *= *( *i );
			return ret;
		}

	private:
		vector< const double* > v_;
};

extern class ZeroOrder* 
	makeHalfReaction( double k, vector< const double*> v );

class BidirectionalReaction: public RateTerm
{
	public:
		BidirectionalReaction(
			ZeroOrder* forward, ZeroOrder* backward) 
			: forward_( forward ), backward_( backward )
		{ // Here we allocate internal forward and backward terms
		// with the correct number of args.
		;
		}
		~BidirectionalReaction()
		{
			delete forward_;
			delete backward_;
		}

		double operator() () const {
			return (*forward_)() - (*backward_)();
		}
	private:
		ZeroOrder* forward_;
		ZeroOrder* backward_;

};

class SumTotal
{
	public:
		SumTotal()
		{
			;
		}
};
