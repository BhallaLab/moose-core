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
		virtual void setRates( double k1, double k2 ) = 0;
		// These next 4 terms are used for talking back to the
		// original rate objects in MOOSE
		virtual void setR1( double k1 ) = 0;
		virtual void setR2( double k2 ) = 0;
		virtual double getR1() const = 0;
		virtual double getR2() const = 0;
		/**
		 * This function finds the reactant indices in the vector
		 * S. It returns the number of substrates found, which are the
		 * first entries in molIndex. The products are the remaining ones.
		 * Note that it does NOT find products for unidirectional
		 * reactions, which is a bit of a problem.
		 */
		virtual unsigned int  getReactants( 
			vector< unsigned int >& molIndex,
			const vector< double >& S ) const = 0;
		static const double EPSILON;
};

// Base class MMEnzme for the purposes of setting rates
// Pure base class: cannot be instantiated, but useful as a handle.
class MMEnzymeBase: public RateTerm
{
	public:
		MMEnzymeBase( double Km, double kcat )
			: Km_( Km ), kcat_( kcat )
		{
			assert( Km_ > 0.0 );
		}

		void setKm( double Km ) {
			if ( Km > 0.0 )
				Km_ = Km;
		}

		void setKcat( double kcat ) {
			if ( kcat > 0 )
				kcat_ = kcat;
		}

		void setRates( double Km, double kcat ) {
			setKm( Km );
			setKcat( kcat );
		}

		void setR1( double Km ) {
			setKm( Km );
		}

		void setR2( double kcat ) {
			setKcat( kcat );
		}

		double getR1() const {
			return Km_;
		}

		double getR2() const {
			return kcat_;
		}


	protected:
		double Km_;
		double kcat_;
};

// Single substrate MMEnzyme: by far the most common.
class MMEnzyme1: public MMEnzymeBase
{
	public:
		MMEnzyme1( double Km, double kcat, 
			const double* enz, const double* sub )
			: MMEnzymeBase( Km, kcat ), enz_( enz ), sub_( sub )
		{
			;
		}

		double operator() () const {
			assert( *sub_ >= -EPSILON );
			return ( kcat_ * *sub_ * *enz_ ) / ( Km_ + *sub_ );
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			molIndex.resize( 2 );
			molIndex[0] = enz_ - &S[0];
			molIndex[1] = sub_ - &S[0];
			return 2;
		}

	private:
		const double *enz_;
		const double *sub_;
};

class MMEnzyme: public MMEnzymeBase
{
	public:
		MMEnzyme( double Km, double kcat, 
			const double* enz, RateTerm* sub )
			: MMEnzymeBase( Km, kcat ), enz_( enz ), substrates_( sub )
		{
			;
		}

		double operator() () const {
			double sub = (*substrates_)();
			// the subtrates_() operator returns the negative of 
			// the conc product.
			// Here we the overall rate.
			assert( sub >= -EPSILON );
			return ( sub * kcat_ * *enz_ ) / ( Km_ + sub );
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			substrates_->getReactants( molIndex, S );
			molIndex.insert( molIndex.begin(), enz_ - &S[0] );
			return molIndex.size();
		}
	private:
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
		void setRates( double k1, double k2 ) {
			; // Dummy function to keep compiler happy
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			molIndex.resize( 0 );
			return 0;
		}
	private:
};

class ZeroOrder: public RateTerm
{
	public:
		ZeroOrder( double k )
			: k_( k )
		{
			assert( !isnan( k_ ) );
		}

		double operator() () const {
			assert( !isnan( k_ ) );
			return k_;
		}

		void setK( double k ) {
			assert( !isnan( k ) );
			if ( k >= 0.0 )
				k_ = k;
		}

		void setRates( double k1, double k2 ) {
			setK( k1 );
		}

		void setR1( double k1 ) {
			setK( k1 );
		}

		void setR2( double k2 ) {
			;
		}

		double getR1() const {
			return k_;
		}

		double getR2() const {
			return 0.0;
		}
		
		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			molIndex.resize( 0 );
			return 0;
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
			assert( !isnan( *y_ ) );
			return k_ * *y_;
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			molIndex.resize( 1 );
			molIndex[0] = y_ - &S[0];
			return 1;
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
			assert( !isnan( *y1_ ) );
			assert( !isnan( *y2_ ) );
			return k_ * *y1_ * *y2_;
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			molIndex.resize( 2 );
			molIndex[0] = y1_ - &S[0];
			molIndex[1] = y2_ - &S[0];
			return 2;
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
			for ( i = v_.begin(); i != v_.end(); i++) {
				assert( !isnan( **i ) );
				ret *= *( *i );
			}
			return ret;
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			molIndex.resize( sizeof( v_ ) );
			for ( unsigned int i = 0; i < sizeof( v_ ); i++ )
				molIndex[i] = v_[i] - &S[0];
			return sizeof( v_ );
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

		void setRates( double kf, double kb ) {
			forward_->setK( kf );
			backward_->setK( kb );
		}

		void setR1( double kf ) {
			forward_->setK( kf );
		}

		void setR2( double kb ) {
			backward_->setK( kb );
		}

		double getR1() const {
			return forward_->getR1();
		}

		double getR2() const {
			return backward_->getR1();
		}

		unsigned int getReactants( vector< unsigned int >& molIndex,
			const vector< double >& S ) const {
			forward_->getReactants( molIndex, S );
			unsigned int ret = molIndex.size();
			vector< unsigned int > temp;
			backward_->getReactants( temp, S );
			molIndex.insert( molIndex.end(), temp.begin(), temp.end() );
			return ret;
		}

	private:
		ZeroOrder* forward_;
		ZeroOrder* backward_;

};

class SumTotal
{
	public:
		SumTotal() // Needed for array setup.
		{;}

		SumTotal( double* target, const vector< const double* >& mol )
			: target_( target ), mol_( mol )
		{;}

		void sum() const {
			double ret = 0.0;
			for( vector< const double* >::const_iterator i = mol_.begin(); 
				i != mol_.end(); i++ )
				ret += **i;
			*target_ = ret;
			assert( !isnan( ret ) );
		}
	private:
		double* target_;
		vector< const double* > mol_;
};

/*
class SumTotal: public RateTerm
{
	public:
		SumTotal( const vector< const double* >& mol )
			: mol_( mol )
		{;}

		// Should be able to do an algorithm operation here.
		double operator() () const {
			double ret = 0.0;
			for( vector< const double* >::const_iterator i = mol_.begin(); 
				i != mol_.end(); i++ )
				ret += **i;
			return ret;
		}

		void setRates( double k1, double k2 )
		{;}
		// These next 4 terms are used for talking back to the
		// original rate objects in MOOSE
		virtual void setR1( double k1 )
		{;}

		virtual void setR2( double k2 )
		{;}

		double getR1() const
		{
			return 0.0;
		}
		
		double getR2() const
		{
			return 0.0;
		}
	private:
		vector< const double* > mol_;
};
*/
