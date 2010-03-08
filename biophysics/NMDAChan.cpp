// NMDAChan.cpp --- 
// 
// Filename: NMDAChan.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sun Feb 28 18:17:56 2010 (+0530)
// Version: 
// Last-Updated: Mon Mar  8 18:14:04 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 134
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
//  The implementation is for the model by Traub, et al. 2005.
// 

// Change log:
// 
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include <cmath>

const Cinfo* initNMDAChanCinfo()
{
	static Finfo* processShared[] =
	{
            new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &NMDAChan::processFunc ) ),
	    new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &NMDAChan::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process message from the scheduler." );
	static Finfo* channelShared[] =
	{
            new SrcFinfo( "channel", Ftype2< double, double >::global() ),
            new DestFinfo( "Vm", Ftype1< double >::global(), 
                           RFCAST( &NMDAChan::channelFunc ) ),
	};

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static Finfo* NMDAChanFinfos[] =
	{
            // The commented out fields are already inherited from SynChan
		// new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getGbar ), 
		// 	RFCAST( &NMDAChan::setGbar )
		// ),
		// new ValueFinfo( "Ek", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getEk ), 
		// 	RFCAST( &NMDAChan::setEk )
		// ),
		// new ValueFinfo( "tau1", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getTau1 ), 
		// 	RFCAST( &NMDAChan::setTau1 )
		// ),
		// new ValueFinfo( "tau2", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getTau2 ), 
		// 	RFCAST( &NMDAChan::setTau2 )
		// ),
		// new ValueFinfo( "normalizeWeights", 
		// 	ValueFtype1< bool >::global(),
		// 	GFCAST( &NMDAChan::getNormalizeWeights ), 
		// 	RFCAST( &NMDAChan::setNormalizeWeights )
		// ),
		// new ValueFinfo( "Gk", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getGk ), 
		// 	RFCAST( &NMDAChan::setGk )
		// ),
		// new ValueFinfo( "Ik", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getIk ), 
		// 	&dummyFunc
		// ),

		// new ValueFinfo( "numSynapses",
		// 	ValueFtype1< unsigned int >::global(),
		// 	GFCAST( &NMDAChan::getNumSynapses ), 
		// 	&dummyFunc // Prohibit reassignment of this index.
		// ),

		// new LookupFinfo( "weight",
		// 	LookupFtype< double, unsigned int >::global(),
		// 	GFCAST( &NMDAChan::getWeight ),
		// 	RFCAST( &NMDAChan::setWeight )
		// ),

		// new LookupFinfo( "delay",
		// 	LookupFtype< double, unsigned int >::global(),
		// 	GFCAST( &NMDAChan::getDelay ),
		// 	RFCAST( &NMDAChan::setDelay )
		// ),
///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		// process,
		// new SharedFinfo( "process", processShared,
		// 	sizeof( processShared ) / sizeof( Finfo* ),
		// 	"This is a shared message to receive Process message from the scheduler." ), 
		// new SharedFinfo( "channel", channelShared,
		// 	sizeof( channelShared ) / sizeof( Finfo* ),
		// 	"This is a shared message to couple channel to compartment. "
		// 	"The first entry is a MsgSrc to send Gk and Ek to the compartment "
		// 	"The second entry is a MsgDest for Vm from the compartment." ),

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
		// new SrcFinfo( "IkSrc", Ftype1< double >::global() ),
		// new SrcFinfo( "GkSrc", Ftype1< double >::global() ),
		// new SrcFinfo( "origChannel", Ftype2< double, double >::
		// 	global() ),

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		// new DestFinfo( "synapse", Ftype1< double >::global(),
		// 		RFCAST( &NMDAChan::synapseFunc ) ,
		// 		"Arrival of a spike. Arg is time of sending of spike." ),
		// new DestFinfo( "activation", Ftype1< double >::global(),
		// 		RFCAST( &NMDAChan::activationFunc ),
		// 		"Sometimes we want to continuously activate the channel" ),
		// new DestFinfo( "modulator", Ftype1< double >::global(),
		// 		RFCAST( &NMDAChan::modulatorFunc ),
		// 		"Modulate channel response" ),
	};

	// NMDAChan is scheduled after the compartment calculations.
	static SchedInfo schedInfo[] = { { process, 0, 1 } };

	static string doc[] =
	{
		"Name", "NMDAChan",
		"Author", "Subhasis Ray, 2010, NCBS",
		"Description", "NMDAChan: Extracellular [Mg2+] dependent NMDA channel."
                               "This channel has four states as described by Jahr and Stevens (J. Neurosci. 1990, 10(9)) "
			       "This implementation is based on equation 4(a) in that article. "
                               "The channel conductance is defined as :"
                               " c * g(V, [Mg2+]o) * S(t) "
                               "where c is a scaling constant. "
                               "S(t) is the legand gated component of the conductance. It rises "
                               "linearly for t = tau1. Then decays exponentially with time constant "
                               "t = tau2. "
                               "g is a function of voltage and the extracellular [Mg2+] defined as: "
                               "1 / { 1 + (a1 + a2) * (a1 * B1 + a2 * B2)/ [A * a1 * (b1 + B1) + A * a2 * (b2 + B2)]} "
                               "a1 = 1e3 * exp( - c1 * V - c2) s^-1, c1 = - 16.0 / V, c2 = 2.91 "
                               "a2 = 1e-3 * [Mg2+] * exp( -c3 * V - c4) mM^-1 s, c3 = 45.0 / V, c4 = 6.97 "
                               "b1 = 1e3 * exp(c5  * V + c6) s^-1, c5 = 9.0 / V, c6 = 1.22 "
                               "b2 = 1e3 * exp(c7 * V + c8) s^-1, c7 = 17.0 / V, c8 = 0.96 "
                               "A = 1e3 * exp(-c9) s^-1, c9 = 2.847 "
                               "B1 = 1e3 * exp(-c10) s^-1, c10 = 0.693 s^-1 "
                               "B2 = 1e3 * exp(-c11) s^-1, c11 = 3.101. "
                               "The behaviour of S(t) is as follows: "
                               "If a spike arrives, then the slope of the linear rise of S(t) is incremented by weight / tau1. "
                               "After tau1 time, this component is removed from the slope (reduced by weight/tau) "
                               "and added over to the rate of decay of S(t)."
                , 
	};

	static Cinfo NMDAChanCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initSynChanCinfo(),
		NMDAChanFinfos,
		sizeof( NMDAChanFinfos )/sizeof(Finfo *),
		ValueFtype1< NMDAChan >::global(),
		schedInfo, 1
	);

	return &NMDAChanCinfo;
}

NMDA::NMDA():c1_(-16.0),
             c2_(2.91),
             c3_(45.0),
             c4_(6.97),
             c5_(9.0),
             c6_(1.22),
             c7_(17.0),
             c8_(0.96),
             c9_(2.847),
             c10_(0.693),
             c11_(3.101),
             slope_(0.0)
    {
        A_ = 1e3 * exp(-c1);
        B1_ = 1e3 * exp(-c10);
        B2_ = 1e3 * exp(-c11);
    }

void NMDA::innerProcessFunc(Eref e, ProcInfo info)
{
    while ( !pendingEvents_.empty() &&
            pendingEvents_.top().delay <= info->currTime_ ) {
        event = pendingEvents_.pop();
        activation_ += event.weight / info->dt_;
        oldEvents_.push(event.event(tau1_));
    }
    while (!oldEvents_.empty() && oldEvents_.top().delay <= info->currTime_){
        x_ -= event.weight;
        y_ += event.weight;
    }
    
    a1_ = exp(-c1 * V_ - c2);
    a2_ = 1e-3 * Mg_ * exp(-c3 * V_ - c4_);
    b1_ = 1e3 * exp(c5_ * V_ + c6_ );
    b2_ = 1e3 * exp(c7_ * V + c8_);
    x_ += activation_ * info->dt_; // forward euler
    y_ += -y * info->dt_ / tau2_;
    unblockedMg_ = 1.0 / ( 1.0 + (a1_ + a2_) * (a1_ * b1_ + a2_ * b2_) / (A_ * (a1_ * (B1_ + b1_) + a2_ * (B2_ + b2_))));
    Gk_ = (x_ + y_) * unblockedMg_ * (V_ - Ek_);
    Ik_ = ( Ek_ - Vm_ ) * Gk_;
    activation_ = 0.0;
    modulation_ = 1.0;
    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );
    // Usually needed by GHK-type objects
    send1< double >( e, gkSlot, Gk_ );
    
}


// 
// nmdachan.cpp ends here
