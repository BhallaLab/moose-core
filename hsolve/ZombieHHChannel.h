#ifndef _Zombie_HHChannel_h
#define _Zombie_HHChannel_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-20012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
*********************************************************************
*/

/**
 * Zombie object that lets HSolve do its calculations, while letting the user
 * interact with this object as if it were the original object.
 * 
 * ZombieHHChannel derives directly from Neutral, unlike the regular
 * HHChannel which derives from ChanBase. ChanBase handles fields like
 * Gbar, Gk, Ek, Ik, which are common to HHChannel, SynChan, etc. On the
 * other hand, these fields are stored separately for HHChannel and SynChan
 * in the HSolver. Hence we cannot have a ZombieChanBase which does, for
 * example:
 *           hsolve_->setGk( id, Gk );
 * Instead we must have ZombieHHChannel and ZombieSynChan which do:
 *           hsolve_->setHHChannelGk( id, Gk );
 * and:
 *           hsolve_->setSynChanGk( id, Gk );
 * respectively.
 */
class ZombieHHChannel
{
	public:
		ZombieHHChannel();
		
		/////////////////////////////////////////////////////////////
		// Value field access function definitions
		/////////////////////////////////////////////////////////////
		
		void setGbar( const Eref& e, const Qinfo* q, double Gbar );
		double getGbar( const Eref& e, const Qinfo* q ) const;
		void setGk( const Eref& e, const Qinfo* q, double Gk );
		double getGk( const Eref& e, const Qinfo* q ) const;
		void setEk( const Eref& e, const Qinfo* q, double Ek );
		double getEk( const Eref& e, const Qinfo* q ) const;
		/// Ik is read-only
		double getIk( const Eref& e, const Qinfo* q ) const;
		void setXpower( const Eref& e, const Qinfo* q, double Xpower );
		double getXpower( const Eref& e, const Qinfo* q ) const;
		void setYpower( const Eref& e, const Qinfo* q, double Ypower );
		double getYpower( const Eref& e, const Qinfo* q ) const;
		void setZpower( const Eref& e, const Qinfo* q, double Zpower );
		double getZpower( const Eref& e, const Qinfo* q ) const;
		void setInstant( const Eref& e, const Qinfo* q, int instant );
		int getInstant( const Eref& e, const Qinfo* q ) const;
		void setX( const Eref& e, const Qinfo* q, double X );
		double getX( const Eref& e, const Qinfo* q ) const;
		void setY( const Eref& e, const Qinfo* q, double Y );
		double getY( const Eref& e, const Qinfo* q ) const;
		void setZ( const Eref& e, const Qinfo* q, double Z );
		double getZ( const Eref& e, const Qinfo* q ) const;
		/**
		 * Not trivial to change Ca-dependence once HSolve has been set up, and
		 * unlikely that one would want to change this field after setup, so
		 * keeping this field read-only.
		 */
		void setUseConcentration( int value );
		int getUseConcentration() const;
		
		/////////////////////////////////////////////////////////////
		// Dest function definitions
		/////////////////////////////////////////////////////////////
		
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
        void handleConc( double value);
    void createGate(const Eref& e, const Qinfo* q, string name);
    // Not sure if the Zombie should hold these. Keeping them out for now.
		 /////////////////////////////////////////////////////////////
		 // Gate handling functions
		 /////////////////////////////////////////////////////////////
		 /**
		  * Access function used for the X gate. The index is ignored.
		  */
		 HHGate* getXgate( unsigned int i );
		 
		 /**
		  * Access function used for the Y gate. The index is ignored.
		  */
		 HHGate* getYgate( unsigned int i );
		 
		 /**
		  * Access function used for the Z gate. The index is ignored.
		  */
		 HHGate* getZgate( unsigned int i );
    void setNumGates(unsigned int num);
    unsigned int getNumXgates() const;
    unsigned int getNumYgates() const;
    unsigned int getNumZgates() const;
		/////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
		
		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );
		
	private:
		HSolve* hsolve_;
		
		/// Exponent for X gate
		double Xpower_;
		/// Exponent for Y gate
		double Ypower_;
		/// Exponent for Z gate
		double Zpower_;
		
		/// Flag for use of conc for input to Z gate calculations.
		bool useConcentration_;
		
		void copyFields( Id chanId, HSolve* hsolve_ );
		
		
		// Not sure if the Zombie should hold these. Keeping them out for now.
		 /**
		  * HHGate data structure for the xGate. This is writable only
		  * on the HHChannel that originally created the HHGate, for others
		  * it must be treated as readonly.
		  */
		 // HHGate* xGate_;
		 
		 // /// HHGate data structure for the yGate. 
		 // HHGate* yGate_;
		 
		 // /// HHGate data structure for the yGate. 
		 // HHGate* zGate_;
		 
		//~ Id myId_;
};


#endif // _Zombie_HHChannel_h
