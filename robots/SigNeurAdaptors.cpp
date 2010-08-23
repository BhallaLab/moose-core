/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../biophysics/Compartment.h"
#include "../element/Neutral.h"
#include "../kinetics/Molecule.h"
#include "../biophysics/HHChannel.h"
// #include "../biophysics/SynInfo.h"
// #include "../biophysics/SynChan.h"
#include "../biophysics/CaConc.h"
#include "SigNeur.h"
#include "Adaptor.h"

extern const Cinfo* initSynChanCinfo();

//////////////////////////////////////////////////////////////////////////

void adaptCa2Sig( TreeNode& t, 
	map< string, Element* >& m, unsigned int offset, double calciumScale,
	Id caId, const string& mol )
{
	static const Finfo* inputFinfo = 
		initAdaptorCinfo()->findFinfo( "inputRequest" );
	static const Finfo* outputFinfo = 
		initAdaptorCinfo()->findFinfo( "outputSrc" );
	static const Finfo* scaleFinfo = 
		initAdaptorCinfo()->findFinfo( "scale" );
	static const Finfo* inputOffsetFinfo = 
		initAdaptorCinfo()->findFinfo( "inputOffset" );
	static const Finfo* outputOffsetFinfo = 
		initAdaptorCinfo()->findFinfo( "outputOffset" );
	
	// Not sure if these update when solved
	static const Finfo* sumTotalFinfo =  
		initMoleculeCinfo()->findFinfo( "sumTotal" );
	static const Finfo* modeFinfo =  
		initMoleculeCinfo()->findFinfo( "mode" );
	static const Finfo* concInitFinfo =  
		initMoleculeCinfo()->findFinfo( "nInit" );
	static const Finfo* volumeScaleFinfo =  
		initMoleculeCinfo()->findFinfo( "volumeScale" );

	// This isn't yet a separate destMsg. Again, issue with update.
	static const Finfo* concFinfo =  
		initCaConcCinfo()->findFinfo( "Ca" );
	static const Finfo* caBasalFinfo =  
		initCaConcCinfo()->findFinfo( "CaBasal" );
	
	// cout << "adaptCa2Sig( el=" << t.compt.path() << ", Ca=" << caId.path() << ", mol=" << mol << ", offset = " << offset << ", cascale=" << calciumScale << endl;
	if ( t.sigEnd <= t.sigStart ) // Nothing to do here, move along
		return;
	// Look up matching molecule
	map< string, Element* >::iterator i = m.find( mol );
	if ( i != m.end() ) {
		Element* e = i->second;
		// cout << "Adding adaptor from " << caId.path() << " to " << i->second->name() << endl;

		assert( t.sigStart >= offset );
		assert( t.sigEnd - offset <= e->numEntries() );

		// Create the adaptor
		string name = "adapt_Ca_2_" + mol;
		Element* adaptor = Neutral::create( "Adaptor", name,
			t.compt, Id::childId( t.compt ) );
		assert( adaptor != 0 );
		Eref adaptorE( adaptor, 0 );
		double caCell = 0.0;
		bool ret = get< double >( caId.eref(), caBasalFinfo, caCell );
		double caSig = 0.08;
		double vs = 1.0; // Converts uM into # for this sig compartment
		Eref MolE( e, t.sigStart - offset );
		ret = get< double >( e, concInitFinfo, caSig );
		assert( ret );
		ret = get< double >( e, volumeScaleFinfo, vs );
		assert( ret );
		vs *= calciumScale;
		ret = set< double >( adaptorE, inputOffsetFinfo, caCell );
		assert( ret );
		ret = set< double >( adaptorE, outputOffsetFinfo, caSig );
		assert( ret );
		ret = set< double >( adaptorE, scaleFinfo, vs );
		assert( ret );

		for ( unsigned int j = t.sigStart; j < t.sigEnd; ++j ) {
			// Connect up the adaptor.
			Eref molE( e, j - offset );
			// Put the molE into sumtotal mode.
			set< int >( molE, modeFinfo, 1 );
			ret = adaptorE.add( outputFinfo->msg(), molE, 
				sumTotalFinfo->msg(), ConnTainer::Default );
			assert( ret );
		}
		// Here we need the adaptor to ask the object for the data,
		// because the solver doesn't push out data at this point.
		ret = adaptorE.add( inputFinfo->msg(), caId.eref(),
			concFinfo->msg(), ConnTainer::Default );
		/*
		ret = caId.eref().add( concFinfo->msg(), adaptorE,
			inputFinfo->msg(), ConnTainer::Default );
			*/
		assert( ret );
	}
}

void SigNeur::makeCell2SigAdaptors()
{
	static const Finfo* lookupChildFinfo = 
		initCompartmentCinfo()->findFinfo( "lookupChild" );
	for ( map< string, string >::iterator i = calciumMap_.begin();
		i != calciumMap_.end(); ++i ) {
		for ( vector< TreeNode >::iterator j = tree_.begin();
			j != tree_.end(); ++j ) {
			Id caId;
			bool ret = lookupGet< Id, string >( j->compt.eref(), lookupChildFinfo, caId, i->first );
			if ( ret && caId.good() ) {
				if ( j->category == SOMA ) {
					adaptCa2Sig( *j, somaMap_, 0, calciumScale_, 
						caId, i->second );
				} else if ( j->category == DEND ) {
					adaptCa2Sig( *j, dendMap_, numSoma_, calciumScale_, 
						caId, i->second );
				} else if ( j->category == SPINE ) {
					adaptCa2Sig( *j, spineMap_, numDend_ + numSoma_,
						calciumScale_,
						caId, i->second );
				}
			}
		}
	}
}

void adaptSig2Chan( TreeNode& t, 
	map< string, Element* >& m,
	unsigned int offset,
	const string& mol, Id chanId )
{
	static const Finfo* inputFinfo = 
		initAdaptorCinfo()->findFinfo( "input" );
	static const Finfo* outputFinfo = 
		initAdaptorCinfo()->findFinfo( "outputSrc" );
	static const Finfo* scaleFinfo = 
		initAdaptorCinfo()->findFinfo( "scale" );
	static const Finfo* inputOffsetFinfo = 
		initAdaptorCinfo()->findFinfo( "inputOffset" );
	static const Finfo* outputOffsetFinfo = 
		initAdaptorCinfo()->findFinfo( "outputOffset" );
	
	// Not sure if these update when solved
	static const Finfo* molNumFinfo =  
		initMoleculeCinfo()->findFinfo( "nSrc" );
	static const Finfo* nInitFinfo =  
		initMoleculeCinfo()->findFinfo( "nInit" );

	// This isn't yet a separate destMsg. Again, issue with update.
	static const Finfo* hhChanGbarFinfo =  
		initHHChannelCinfo()->findFinfo( "Gbar" );

	static const Finfo* synChanGbarFinfo =  
		initSynChanCinfo()->findFinfo( "Gbar" );


	// Set up the correct Finfo. Complain if we try to connect up an 
	// unknown chan type.
	const Finfo* gbarFinfo = 0;
	if ( chanId()->cinfo()->isA( initHHChannelCinfo() ) )
		gbarFinfo = hhChanGbarFinfo;
	if ( chanId()->cinfo()->isA( initSynChanCinfo() ) )
		gbarFinfo = synChanGbarFinfo;
	if ( gbarFinfo == 0 ) {
		cout << "Error: attempt to set up adaptor from signaling to \n" <<
		" biophysics for an unknown channel type: " << 
		chanId()->cinfo()->name() << endl <<
		"for channel " << chanId.path() << endl;
		return;
	}

	// Look up matching molecule
	map< string, Element* >::iterator i = m.find( mol );
	if ( i != m.end() ) {
		Element* e = i->second;
		// cout << "Adding adaptor from " << e->name() << " to " << chanId.path() << endl;
		assert( t.sigStart >= offset );
		assert( t.sigEnd - offset <= e->numEntries() );


		// Create the adaptor
		string name = "adapt_" + mol + "_2_" + chanId.eref().e->name();
		Element* adaptor = Neutral::create( "Adaptor", name,
			t.compt, Id::childId( t.compt ) );
		assert( adaptor != 0 );
		Eref adaptorE( adaptor, 0 );

		double n = 0.00;
		Eref MolE( e, t.sigStart - offset );
		bool ret = get< double >( e, nInitFinfo, n );
		assert( ret );
		if ( n > 0.0 ) {
			double gbar = 0.0;
			ret = get< double >( chanId.eref(), gbarFinfo, gbar );
			assert( ret );
			double scale =  gbar / n;
			ret = set< double >( adaptorE, scaleFinfo, scale );
			assert( ret );
		}
		ret = set< double >( adaptorE, inputOffsetFinfo, 0.0 );
		assert( ret );
		ret = set< double >( adaptorE, outputOffsetFinfo, 0.0 );
		assert( ret );

		for ( unsigned int j = t.sigStart; j < t.sigEnd; ++j ) {
			// Connect up the adaptor.
			Eref molE( e, j - offset );
			ret = molE.add( molNumFinfo->msg(), adaptorE,
				inputFinfo->msg(), ConnTainer::Default );
			assert( ret );

			// Here we set the parameters of the adaptor.
		}
		ret = adaptorE.add( outputFinfo->msg(), chanId.eref(), 
			gbarFinfo->msg(), ConnTainer::Default );
		assert( ret );
	}
}


void SigNeur::makeSig2CellAdaptors()
{
	static const Finfo* lookupChildFinfo = 
		initCompartmentCinfo()->findFinfo( "lookupChild" );
	for ( map< string, string >::iterator i = channelMap_.begin();
		i != channelMap_.end(); ++i ) {
		for ( vector< TreeNode >::iterator j = tree_.begin();
			j != tree_.end(); ++j ) {
			Id chanId;
			bool ret = lookupGet< Id, string >( j->compt.eref(), lookupChildFinfo, chanId, i->second );
			if ( ret && chanId.good() ) {
				if ( j->category == SOMA ) {
					adaptSig2Chan( *j, somaMap_, 0, i->first, chanId );
				} else if ( j->category == DEND ) {
					adaptSig2Chan( *j, dendMap_, numSoma_, 
						i->first, chanId );
				} else if ( j->category == SPINE ) {
					adaptSig2Chan( *j, spineMap_, numDend_ + numSoma_,
						i->first, chanId );
				}
			}
		}
	}
}
