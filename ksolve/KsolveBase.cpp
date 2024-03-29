/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "../basecode/header.h"
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#endif

#include "VoxelPoolsBase.h"
// #include "VoxelPools.h"
#include "../mesh/VoxelJunction.h"
#include "XferInfo.h"
#include "KsolveBase.h"

#include "RateTerm.h"
#include "FuncTerm.h"
#include "../basecode/SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "Stoich.h"
#include "../shell/Shell.h"
#include "../mesh/MeshEntry.h"
#include "../mesh/Boundary.h"
#include "../mesh/ChemCompt.h"

KsolveBase::KsolveBase()
    : stoich_(), compartment_(), isBuilt_(false)
{;}

void KsolveBase::updateJunctions( double dt )
{;}

void KsolveBase::setPrev()
{;}

/////////////////////////////////////////////////////////////////////

Id KsolveBase::getCompartment() const
{
    return compartment_;
}

void KsolveBase::setCompartment( Id compt )
{
    isBuilt_ = false; // We will have to now rebuild the whole thing.
    if ( compt.element()->cinfo()->isA( "ChemCompt" ) )
    {
        compartment_ = compt;
        vector<double> vols = Field<vector<double>>::get( compt, "voxelVolume" );

        if (vols.size() > 0)
        {
            setNumAllVoxels( vols.size() );
            for ( unsigned int i = 0; i < vols.size(); ++i )
                pools(i)->setVolume( vols[i] );
        }
    }
}
	
void KsolveBase::notifyDestroyPool( const Eref& e )
{;}

void KsolveBase::notifyAddPool( const Eref& e )
{;}

void KsolveBase::notifyRemovePool( const Eref& e )
{;}

void KsolveBase::notifyAddMsgSrcPool( const Eref& e, ObjId msgId )
{;}

void KsolveBase::notifyAddMsgDestPool( const Eref& e, ObjId msgId )
{;}

