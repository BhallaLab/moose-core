/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
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
#include "../utility/numutil.h"
class RateTerm
{
	public:
		RateTerm() {;}
		virtual ~RateTerm() {;}
		/// Computes the rate. The argument is the molecule array.
		virtual double operator() ( const double* S ) const = 0;

		/**
		 * Assign the rates.
		 */
		virtual void setRates( double k1, double k2 ) = 0;

		/// Used by Zombie to assign rate terms
		virtual void setR1( double k1 ) = 0;

		/// Used by Zombie to assign rate terms
		virtual void setR2( double k2 ) = 0;

		/// Used by Zombie to return rate terms
		virtual double getR1() const = 0;

		/// Used by Zombie to return rate terms
		virtual double getR2() const = 0;

		/**
		 * This function finds the reactant indices in the vector
		 * S. It returns the number of substrates found, which are the
		 * first entries in molIndex. The products are the remaining ones.
		 * Note that it does NOT find products for unidirectional
		 * reactions, which is a bit of a problem.
		 */
		virtual unsigned int  getReactants( 
			vector< unsigned int >& molIndex ) const = 0;
		static const double EPSILON;

		/**
		 * This is used to rescale the RateTerm kinetics when the 
		 * compartment volume changes. This is needed because the kinetics
		 * are in extensive units, that is, mol numbers, rather than in
		 * intensive units like concentration. So when the volume changes
		 * the rate terms change. Each Rate term checks if any of its
		 * reactant molecules are affected, and if so, rescales.
		 * Ratio is newVol / oldVol
		 */
		virtual void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio ) = 0;
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

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			Km_ *= ratio;
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
			unsigned int enz, unsigned int sub )
			: MMEnzymeBase( Km, kcat ), enz_( enz ), sub_( sub )
		{
			;
		}

		double operator() ( const double* S ) const {
			assert( S[ sub_ ] >= -EPSILON );
			return ( kcat_ * S[ sub_ ] * S[ enz_ ] ) / ( Km_ + S[ sub_ ] );
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 2 );
			molIndex[0] = enz_;
			molIndex[1] = sub_;
			return 2;
		}

	private:
		unsigned int enz_;
		unsigned int sub_;
};

class MMEnzyme: public MMEnzymeBase
{
	public:
		MMEnzyme( double Km, double kcat, 
			unsigned int enz, RateTerm* sub )
			: MMEnzymeBase( Km, kcat ), enz_( enz ), substrates_( sub )
		{
			;
		}

		double operator() ( const double* S ) const {
			double sub = (*substrates_)( S );
			// the subtrates_() operator returns the conc product.
			assert( sub >= -EPSILON );
			return ( sub * kcat_ * S[ enz_ ] ) / ( Km_ + sub );
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			substrates_->getReactants( molIndex );
			molIndex.insert( molIndex.begin(), enz_ );
			return molIndex.size();
		}
	private:
		unsigned int enz_;
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

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 0 );
			return 0;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			return; // Need to figure out what to do here.
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

		double operator() ( const double* S ) const {
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
		
		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 0 );
			return 0;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			return; // Nothing needs to be scaled.
		}
	protected:
		double k_;
};

/**
 * This rather odd reaction is used when we have an amount y of a molecule
 * and we want it to proceed to zero at a fixed rate k. k would usually
 * be 1/dt. Following dt seconds, we need to update y as it has nominally
 * all been used up.
 */
class Flux: public ZeroOrder
{
	public:
		Flux( double k, unsigned int y )
			: ZeroOrder( k ), y_( y )
		{;}

		double operator() ( const double* S ) const {
			assert( !isnan( S[ y_ ] ) );
			return k_ * S[ y_ ];
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 0 );
			return 0;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			return; // Nothing needs to be scaled.
		}

	private:
		unsigned int y_;
};

class FirstOrder: public ZeroOrder
{
	public:
		FirstOrder( double k, unsigned int y )
			: ZeroOrder( k ), y_( y )
		{;}

		double operator() ( const double* S ) const {
			assert( !isnan( S[ y_ ] ) );
			return k_ * S[ y_ ];
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 1 );
			molIndex[0] = y_;
			return 1;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			return; // Nothing needs to be scaled.
		}

	private:
		unsigned int y_;
};

class SecondOrder: public ZeroOrder
{
	public:
		SecondOrder( double k, unsigned int y1, unsigned int y2 )
			: ZeroOrder( k ), y1_( y1 ), y2_( y2 )
		{;}

		double operator() ( const double* S ) const {
			assert( !isnan( S[ y1_ ] ) );
			assert( !isnan( S[ y2_ ] ) );
			return k_ * S[ y1_ ] * S[ y2_ ];
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 2 );
			molIndex[0] = y1_;
			molIndex[1] = y2_;
			return 2;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			if ( comptIndex == compartmentLookup[ y1_ ] || 
				comptIndex == compartmentLookup[ y2_ ] )
			k_ /= ratio;
		}

	private:
		unsigned int y1_;
		unsigned int y2_;
};

/**
 * The reason for this class is that if we have a single substrate
 * acting in 2nd order, we must account for the removal of one molecule
 * halfway through the reaction. This has a major effect when there
 * is only one molecule to start with: the eventual # would become
 * negative if we didn't do this correction!
 */
class StochSecondOrderSingleSubstrate: public ZeroOrder
{
	public:
		StochSecondOrderSingleSubstrate( double k, unsigned int y )
			: ZeroOrder( k ), y_( y )
		{;}

		double operator() ( const double* S ) const {
			double y = S[ y_ ];
			assert( !isnan( y ) );
			return k_ * ( y - 1 ) * y;
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex.resize( 2 );
			molIndex[0] = y_;
			molIndex[1] = y_;
			return 2;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			if ( comptIndex == compartmentLookup[ y_ ] )
				k_ /= ratio;
		}

	private:
		const unsigned int y_;
};

class NOrder: public ZeroOrder
{
	public:
		NOrder( double k, vector< unsigned int > v )
			: ZeroOrder( k ), v_( v )
		{;}

		double operator() ( const double* S ) const {
			double ret = k_;
			vector< unsigned int >::const_iterator i;
			for ( i = v_.begin(); i != v_.end(); i++) {
				assert( !isnan( S[ *i ] ) );
				ret *= S[ *i ];
			}
			return ret;
		}

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			molIndex = v_;
			return v_.size();
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			for ( unsigned int i = 1; i < v_.size(); ++i ) {
				if ( comptIndex == compartmentLookup[ v_[i] ] )
					k_ /= ratio;
			}
		}

	protected:
		vector< unsigned int > v_;
};

/**
 * This is an unpleasant case, like the StochSecondOrderSingleSubstrate.
 * Here we deal with the possibility that one or more of the substrates
 * may be of order greater than one. If so, we need to diminish the N
 * of each substrate by one for each time the substrate is factored 
 * into the rate.
 */
class StochNOrder: public NOrder
{
	public:
		StochNOrder( double k, vector< unsigned int > v );

		double operator() ( const double* S ) const;
};

extern class ZeroOrder* 
	makeHalfReaction( double k, vector< unsigned int > v );

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

		double operator() ( const double* S ) const {
			return (*forward_)( S ) - (*backward_)( S );
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

		unsigned int getReactants( vector< unsigned int >& molIndex ) const{
			forward_->getReactants( molIndex );
			unsigned int ret = molIndex.size();
			vector< unsigned int > temp;
			backward_->getReactants( temp );
			molIndex.insert( molIndex.end(), temp.begin(), temp.end() );
			return ret;
		}

		void rescaleVolume( short comptIndex, 
			const vector< short >& compartmentLookup, double ratio )
		{
			forward_->rescaleVolume( comptIndex, compartmentLookup, ratio );
			backward_->rescaleVolume( comptIndex, compartmentLookup, ratio);
		}

	private:
		ZeroOrder* forward_;
		ZeroOrder* backward_;
};
