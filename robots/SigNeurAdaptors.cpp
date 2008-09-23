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
#include "../biophysics/CaConc.h"
#include "SigNeur.h"
#include "Adaptor.h"

//////////////////////////////////////////////////////////////////////////

void adaptCa2Sig( TreeNode& t, 
	map< string, Element* >& m, unsigned int offset,
	Id caId, const string& mol )
{
	static const Finfo* inputFinfo = 
		initAdaptorCinfo()->findFinfo( "input" );
	static const Finfo* outputFinfo = 
		initAdaptorCinfo()->findFinfo( "outputSrc" );
	
	// Not sure if these update when solved
	static const Finfo* sumTotalFinfo =  
		initMoleculeCinfo()->findFinfo( "sumTotal" );
	static const Finfo* modeFinfo =  
		initMoleculeCinfo()->findFinfo( "mode" );

	// This isn't yet a separate destMsg. Again, issue with update.
	static const Finfo* concFinfo =  
		initCaConcCinfo()->findFinfo( "concSrc" );
	
	// Look up matching molecule
	map< string, Element* >::iterator i = m.find( mol );
	if ( i != m.end() ) {
		Element* e = i->second;
		cout << "Adding adaptor from " << caId.path() << " to " <<
			i->second->name() << endl;

		assert( t.sigStart >= offset );
		assert( t.sigEnd - offset <= e->numEntries() );

		// Create the adaptor
		string name = "ca2" + mol;
		Element* adaptor = Neutral::create( "Adaptor", name,
			t.compt, Id::childId( t.compt ) );
		assert( adaptor != 0 );
		Eref adaptorE( adaptor, 0 );

		for ( unsigned int j = t.sigStart; j < t.sigEnd; ++j ) {
			// Connect up the adaptor.
			Eref molE( e, j - offset );
			// Put the molE into sumtotal mode.
			set< int >( molE, modeFinfo, 1 );
			bool ret = adaptorE.add( outputFinfo->msg(), molE, 
				sumTotalFinfo->msg(), ConnTainer::Default );
			assert( ret );

			// Here we set the parameters of the adaptor.
		}
		bool ret = caId.eref().add( concFinfo->msg(), adaptorE,
			inputFinfo->msg(), ConnTainer::Default );
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
					adaptCa2Sig( *j, somaMap_, 0, caId, i->second );
				} else if ( j->category == DEND ) {
					adaptCa2Sig( *j, dendMap_, numSoma_, caId, i->second );
				} else if ( j->category == SPINE ) {
					adaptCa2Sig( *j, spineMap_, numDend_ + numSoma_,
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
	
	// Not sure if these update when solved
	static const Finfo* molNumFinfo =  
		initMoleculeCinfo()->findFinfo( "nSrc" );

	// This isn't yet a separate destMsg. Again, issue with update.
	static const Finfo* gbarFinfo =  
		initHHChannelCinfo()->findFinfo( "Gbar" );

	// Look up matching molecule
	map< string, Element* >::iterator i = m.find( mol );
	if ( i != m.end() ) {
		Element* e = i->second;
		cout << "Adding adaptor from " << e->name() << " to " << 
			chanId.path() << endl;
		assert( t.sigStart >= offset );
		assert( t.sigEnd - offset <= e->numEntries() );

		// Create the adaptor
		Element* adaptor = Neutral::create( "Adaptor", "sig2chan",
			t.compt, Id::childId( t.compt ) );
		assert( adaptor != 0 );
		Eref adaptorE( adaptor, 0 );

		for ( unsigned int j = t.sigStart; j < t.sigEnd; ++j ) {
			// Connect up the adaptor.
			Eref molE( e, j - offset );
			bool ret = molE.add( molNumFinfo->msg(), adaptorE,
				inputFinfo->msg(), ConnTainer::Default );
			assert( ret );

			// Here we set the parameters of the adaptor.
		}
		bool ret = adaptorE.add( outputFinfo->msg(), chanId.eref(), 
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
