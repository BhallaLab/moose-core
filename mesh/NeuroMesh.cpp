/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "Stencil.h"
#include "ChemMesh.h"
#include "CylBase.h"
#include "NeuroNode.h"
#include "NeuroStencil.h"
#include "NeuroMesh.h"
#include "../utility/numutil.h"
const Cinfo* NeuroMesh::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		/*
		static ReadOnlyValueFinfo< NeuroMesh, vector< unsigned int > > 
			chemTree(
			"chemTree",
			"Branching structure of neuronal compartments."
			"Ignores startNodes if any",
			&NeuroMesh::getChemTree
		);

		static ReadOnlyValueFinfo< NeuroMesh, vector< double >  > cylCoords(
			"cylCoords",
			"End coordinates of cylinders in the NeuroMesh diffusive "
			"compartments. Taken directly from the neuronal compartments."
			"Organized as x, y, z. Length is therefore 3 * numCompts."
			&NeuroMesh::getCylCoords
		);
		*/
		static ValueFinfo< NeuroMesh, Id > cell(
			"cell",
			"Id for base element of cell model. Uses this to traverse the"
			"entire tree of the cell to build the mesh.",
			&NeuroMesh::setCell,
			&NeuroMesh::getCell
		);
		static ValueFinfo< NeuroMesh, vector< Id > > subTree(
			"subTree",
			"Set of compartments to model. If they happen to be contiguous"
			"then also set up diffusion between the compartments. Can also"
			"handle cases where the same cell is divided into multiple"
			"non-diffusively-coupled compartments",
			&NeuroMesh::setSubTree,
			&NeuroMesh::getSubTree
		);
		static ValueFinfo< NeuroMesh, bool > skipSpines(
			"skipSpines",
			"Flag: when skipSpines is true, the traversal does not include"
			"any compartment with the string 'spine' or 'neck' in its name,"
			"and also then skips compartments below this skipped one."
			"Allows to set up separate mesh for spines, based on the "
			"same cell model.",
			&NeuroMesh::setSkipSpines,
			&NeuroMesh::getSkipSpines
		);
		static ReadOnlyValueFinfo< NeuroMesh, unsigned int > numSegments(
			"numSegments",
			"Number of cylindrical/spherical segments in model",
			&NeuroMesh::getNumSegments
		);
		static ReadOnlyValueFinfo< NeuroMesh, unsigned int > numDiffCompts(
			"numDiffCompts",
			"Number of diffusive compartments in model",
			&NeuroMesh::getNumDiffCompts
		);

		static ValueFinfo< NeuroMesh, double > diffLength(
			"diffLength",
			"Diffusive length constant to use for subdivisions. "
			"The system will"
			"attempt to subdivide cell using diffusive compartments of"
			"the specified diffusion lengths as a maximum."
			"In order to get integral numbers"
			"of compartments in each segment, it may subdivide more "
			"finely."
			"Uses default of 0.5 microns, that is, half typical lambda."
			"For default, consider a tau of about 1 second for most"
		   "reactions, and a diffusion const of about 1e-12 um^2/sec."
		   "This gives lambda of 1 micron",
			&NeuroMesh::setDiffLength,
			&NeuroMesh::getDiffLength
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////

	static Finfo* neuroMeshFinfos[] = {
		&cell,			// Value
		&subTree,		// Value
		&skipSpines,	// Value
		&numSegments,		// ReadOnlyValue
		&numDiffCompts,		// ReadOnlyValue
		&diffLength,			// Value
	};

	static Cinfo neuroMeshCinfo (
		"NeuroMesh",
		ChemMesh::initCinfo(),
		neuroMeshFinfos,
		sizeof( neuroMeshFinfos ) / sizeof ( Finfo* ),
		new Dinfo< NeuroMesh >()
	);

	return &neuroMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* neuroMeshCinfo = NeuroMesh::initCinfo();

//////////////////////////////////////////////////////////////////
// Class stuff.
//////////////////////////////////////////////////////////////////
NeuroMesh::NeuroMesh()
	:
		size_( 0.0 ),
		diffLength_( 0.5e-6 ),
		skipSpines_( false ),
		ns_( nodes_, nodeIndex_, vs_, area_ )
{
	stencil_.resize( 1, &ns_ );
}

NeuroMesh::NeuroMesh( const NeuroMesh& other )
	:
		size_( other.size_ ),
		diffLength_( other.diffLength_ ),
		cell_( other.cell_ ),
		ns_( nodes_, nodeIndex_, vs_, area_ )
{
	stencil_.resize( 1, &ns_ );
}

NeuroMesh& NeuroMesh::operator=( const NeuroMesh& other )
{
	nodes_ = other.nodes_;
	nodeIndex_ = other.nodeIndex_;
	vs_ = other.vs_;
	area_ = other.area_;
	size_ = other.size_;
	diffLength_ = other.diffLength_;
	cell_ = other.cell_;
	skipSpines_ = other.skipSpines_;
	stencil_.resize( 1, &ns_ );
	return *this;
}

NeuroMesh::~NeuroMesh()
{
	;
}

//////////////////////////////////////////////////////////////////
// Field assignment stuff
//////////////////////////////////////////////////////////////////

/**
 * This assumes that lambda is the quantity to preserve, over numEntries.
 * So when the compartment changes size, so does numEntries.
 */
void NeuroMesh::updateCoords()
{
		/*
	double temp = sqrt( 
		( x1_ - x0_ ) * ( x1_ - x0_ ) + 
		( y1_ - y0_ ) * ( y1_ - y0_ ) + 
		( z1_ - z0_ ) * ( z1_ - z0_ )
	);

	if ( doubleEq( temp, 0.0 ) ) {
		cout << "Error: NeuroMesh::updateCoords:\n"
		"total length of compartment = 0 with these parameters\n";
		return;
	}
	totLen_ = temp;


	temp = totLen_ / lambda_;
	if ( temp < 1.0 ) {
		lambda_ = totLen_;
		numEntries_ = 1;
	} else {
		numEntries_ = static_cast< unsigned int >( round ( temp ) );
		lambda_ = totLen_ / numEntries_;
	}
	rSlope_ = ( r1_ - r0_ ) / numEntries_;
	lenSlope_ = lambda_ * rSlope_ * 2 / ( r0_ + r1_ );

	*/
	buildStencil();
}

void NeuroMesh::setDiffLength( double v )
{
	diffLength_ = v;
	updateCoords();
}

double NeuroMesh::getDiffLength() const
{
	return diffLength_;
}

unsigned int NeuroMesh::innerGetDimensions() const
{
	return 3;
}

void NeuroMesh::setCell( Id cell )
{
		// Much more to do here.
		cell_ = cell;
}

Id NeuroMesh::getCell() const
{
		return cell_;
}

void NeuroMesh::setSubTree( vector< Id > compartments )
{
		;
}

vector< Id > NeuroMesh::getSubTree() const
{
		vector< Id > ret;
		return ret;
}

void NeuroMesh::setSkipSpines( bool v )
{
		if ( v != skipSpines_ ) {
				skipSpines_ = v;
				updateCoords();
		}
}

bool NeuroMesh::getSkipSpines() const
{
		return skipSpines_;
}

unsigned int NeuroMesh::getNumSegments() const
{
		unsigned int ret = 0;
		for ( vector< NeuroNode >::const_iterator i = nodes_.begin();
						i != nodes_.end(); ++i )
				ret += !i->isDummyNode();
		return ret;
}

unsigned int NeuroMesh::getNumDiffCompts() const
{
	return nodeIndex_.size();
}

//////////////////////////////////////////////////////////////////
// FieldElement assignment stuff for MeshEntries
//////////////////////////////////////////////////////////////////

/// Virtual function to return MeshType of specified entry.
unsigned int NeuroMesh::getMeshType( unsigned int fid ) const
{
	assert( fid < nodeIndex_.size() );
	assert( nodeIndex_[fid] < nodes_.size() );
	if ( nodes_[ nodeIndex_[fid] ].isSphere() )
		return SPHERE_SHELL_SEG;

	return CYL;
}

/// Virtual function to return dimensions of specified entry.
unsigned int NeuroMesh::getMeshDimensions( unsigned int fid ) const
{
	return 3;
}

/// Virtual function to return volume of mesh Entry.
double NeuroMesh::getMeshEntrySize( unsigned int fid ) const
{
	assert( fid < nodeIndex_.size() );
	assert( nodeIndex_[fid] < nodes_.size() );
	const NeuroNode& node = nodes_[ nodeIndex_[fid] ];
	assert( fid >= node.startFid() );
	assert ( node.parent() < nodes_.size() );
	const NeuroNode& parent = nodes_[ node.parent() ];
	return node.voxelVolume( parent, fid - node.startFid() );
}

/// Virtual function to return coords of mesh Entry.
/// For Cylindrical mesh, coords are x1y1z1 x2y2z2 r0 r1 phi0 phi1
vector< double > NeuroMesh::getCoordinates( unsigned int fid ) const
{
	assert( fid < nodeIndex_.size() );
	assert( nodeIndex_[fid] < nodes_.size() );
	const NeuroNode& node = nodes_[ nodeIndex_[fid] ];
	assert( fid >= node.startFid() );
	assert ( node.parent() < nodes_.size() );
	const NeuroNode& parent = nodes_[ node.parent() ];

	return node.getCoordinates( parent, fid - node.startFid() );
}
/// Virtual function to return info on Entries connected to this one
vector< unsigned int > NeuroMesh::getNeighbors( unsigned int fid ) const
{
	vector< unsigned int > ret;
	assert( fid < nodeIndex_.size() );
	assert( nodeIndex_[fid] < nodes_.size() );
	const NeuroNode& node = nodes_[ nodeIndex_[fid] ];
	assert( fid >= node.startFid() );
	assert( node.getNumDivs() > 0 );

	// First, fill in fids closer to soma.
	if ( fid == node.startFid() ) { // Check for parental fid.
		assert ( node.parent() < nodes_.size() );
		const NeuroNode* parent = &nodes_[ node.parent() ];
		while ( !parent->isStartNode() ) {
			if ( parent->isDummyNode() ) {
				parent = &nodes_[ parent->parent() ];
			} else {
				ret.push_back( 
					parent->startFid() + parent->getNumDivs() - 1 );
				break;
			}
		}
	} else {
		ret.push_back( fid - 1 );
	}

	// Next, fill in nodes further away from soma.
	if ( fid == node.startFid() + node.getNumDivs() - 1 ) { 
		// TODO: check for child fids
	} else {
		ret.push_back( fid + 1 );
	}

	return ret;	
}

/// Virtual function to return diffusion X-section area for each neighbor
vector< double > NeuroMesh::getDiffusionArea( unsigned int fid ) const
{
	assert( fid < nodeIndex_.size() );
	assert( nodeIndex_[fid] < nodes_.size() );
	const NeuroNode& node = nodes_[ nodeIndex_[fid] ];
	assert( fid >= node.startFid() );
	assert ( node.parent() < nodes_.size() );
	const NeuroNode& parent = nodes_[ node.parent() ];
	vector< double > ret;
	vector< unsigned int > neighbors = getNeighbors( fid );
	for ( unsigned int i = 0; i < neighbors.size(); ++i ) {
		ret.push_back( node.getDiffusionArea( parent, neighbors[ i ] ) );
	}
	return ret;
}

/// Virtual function to return scale factor for diffusion.
/// I think all dendite tips need to return just one entry of 1.
//  Regular points return vector( 2, 1.0 );
vector< double > NeuroMesh::getDiffusionScaling( unsigned int fid ) const
{
		/*
	if ( nodeIndex_.size() <= 1 )
		return vector< double >( 0 );

	if ( !isToroid_ && ( fid == 0 || fid == (nodeIndex_.size() - 1) ) )
		return vector< double >( 1, 1.0 );
		*/

	return vector< double >( 2, 1.0 );
}

//////////////////////////////////////////////////////////////////
// Dest funcsl
//////////////////////////////////////////////////////////////////

/// More inherited virtual funcs: request comes in for mesh stats
/// Not clear what this does.
void NeuroMesh::innerHandleRequestMeshStats( const Eref& e, const Qinfo* q, 
		const SrcFinfo2< unsigned int, vector< double > >* meshStatsFinfo
	)
{
	vector< double > ret( size_ / nodeIndex_.size() ,1 );
	meshStatsFinfo->send( e, q->threadNum(), 1, ret );
}

void NeuroMesh::innerHandleNodeInfo(
			const Eref& e, const Qinfo* q, 
			unsigned int numNodes, unsigned int numThreads )
{
	unsigned int numEntries = nodeIndex_.size();
	vector< double > vols( numEntries, 0.0 );
	for ( unsigned int i = 0; i < numEntries; ++i ) {
		assert( nodeIndex_[i] < nodes_.size() );
		NeuroNode& node = nodes_[ nodeIndex_[i] ];
		assert( nodes_.size() > node.parent() );
		NeuroNode& parent = nodes_[ node.parent() ];
		vols[i] = node.voxelVolume( parent, i );
	}
	vector< unsigned int > localEntries( numEntries );
	vector< vector< unsigned int > > outgoingEntries;
	vector< vector< unsigned int > > incomingEntries;
	meshSplit()->send( e, q->threadNum(), 
		vols, localEntries,
		outgoingEntries, incomingEntries );
}
//////////////////////////////////////////////////////////////////

/**
 * Inherited virtual func. Returns number of MeshEntry in array
 */
unsigned int NeuroMesh::innerGetNumEntries() const
{
	return nodeIndex_.size();
}

/**
 * Inherited virtual func. Assigns number of MeshEntries.
 * This one doesn't set exact number, because each compartment has to
 * fit integral numbers of voxels.
 */
void NeuroMesh::innerSetNumEntries( unsigned int n )
{
	static const unsigned int WayTooLarge = 1000000;
	if ( n == 0 || n > WayTooLarge ) {
		cout << "Warning: NeuroMesh::innerSetNumEntries( " << n <<
		" ): out of range\n";
		return;
	}
	double totalLength = 0;
	for ( vector< NeuroNode >::iterator i = nodes_.begin(); 
					i != nodes_.end(); ++i )
	{
			if ( !i->isDummyNode() ) {
					if ( i->isSphere() )
							totalLength += i->getDia();
					else
							totalLength += i->getLength();
			}
	}
	assert( totalLength > 0 );
	diffLength_ = totalLength / n;
	updateCoords();
}


/**
 * This is a bit odd, effectively asks to build an imaginary neuron and
 * then subdivide it. I'll make do with a ball-and-stick model: Soma with
 * a single apical dendrite with reasonable diameter. I will interpret
 * size as total length of neuron, not as volume. 
 * Soma will have a diameter of up to 20 microns, anything bigger than
 * this is treated as soma of 20 microns + 
 * dendrite of (specified length - 10 microns) for radius of soma.
 * This means we avoid having miniscule dendrites protruding from soma,
 * the shortest one will be 10 microns.
 */
void NeuroMesh::innerBuildDefaultMesh( const Eref& e, const Qinfo* q,
	double size, unsigned int numEntries )
{
	
	if ( size > 10e-3 ) {
		cout << "Warning: attempt to build a neuron of dendritic length " <<
				size << " metres.\n Seems improbable.\n" <<
				"Using default of 0.001 m\n";
		size = 1e-3;
	}

	diffLength_ = size / numEntries;

	vector< unsigned int > noChildren( 0 );
	vector< unsigned int > oneChild( 1, 2 );

	if ( size < 20e-6 ) {
			CylBase cb( 0, 0, 0, size, 0, numEntries );
			NeuroNode soma( cb, 0, noChildren, 0, Id(), true );
			nodes_.resize( 1, soma );
			nodeIndex_.resize( 1, 0 );
	} else {
			CylBase cb( 0, 0, 0, 20e-6, 0, 1 );
			NeuroNode soma( cb, 0, oneChild, 0, Id(), true );
			nodes_.resize( 1, soma );
			nodeIndex_.resize( 1, 0 );

			CylBase cbDummy( 0, 0, 10e-6, 4e-6, 0, 0 );
			NeuroNode dummy( cbDummy, 0, noChildren, 1, Id(), false );
			nodes_.push_back( dummy );

			CylBase cbDend( 0, 0, size, 2e-6, size - 10e-6, numEntries - 1);
			NeuroNode dend( cbDend, 1, noChildren, 2, Id(), false );
			nodes_.push_back( dend );
			for ( unsigned int i = 1; i < numEntries; ++i )
				nodeIndex_.push_back( 2 );
	}
	updateCoords();
}

//////////////////////////////////////////////////////////////////
// Utility function to transmit any changes to target nodes.
//////////////////////////////////////////////////////////////////

void NeuroMesh::transmitChange( const Eref& e, const Qinfo* q )
{
	Id meshEntry( e.id().value() + 1 );
	assert( 
		meshEntry.eref().data() == reinterpret_cast< char* >( lookupEntry( 0 ) )
	);
	unsigned int totalNumEntries = nodeIndex_.size();
	unsigned int localNumEntries = totalNumEntries;
	unsigned int startEntry = 0;
	vector< unsigned int > localIndices( localNumEntries ); // empty
	for ( unsigned int i = 0; i < localNumEntries; ++i )
		localIndices[i] = i;
	vector< double > vols( localNumEntries, 0.0 );
	vector< vector< unsigned int > > outgoingEntries; // [node#][Entry#]
	vector< vector< unsigned int > > incomingEntries; // [node#][Entry#]

	// This function updates the size of the FieldDataHandler for the 
	// MeshEntries.
	DataHandler* dh = meshEntry.element()->dataHandler();
	FieldDataHandlerBase* fdh = dynamic_cast< FieldDataHandlerBase* >( dh );
	assert( fdh );
	if ( totalNumEntries > fdh->getMaxFieldEntries() ) {
		fdh->setMaxFieldEntries( localNumEntries );
	}

	// This message tells the Stoich about the new mesh, and also about
	// how it communicates with other nodes.
	meshSplit()->fastSend( e, q->threadNum(), 
		vols, localIndices, 
		outgoingEntries, incomingEntries );

	// This func goes down to the MeshEntry to tell all the pools and
	// Reacs to deal with the new mesh. They then update the stoich.
	lookupEntry( 0 )->triggerRemesh( meshEntry.eref(), q->threadNum(), 
		startEntry, localIndices, vols );
}

//////////////////////////////////////////////////////////////////
// Utility function to set up Stencil for diffusion
//////////////////////////////////////////////////////////////////
void NeuroMesh::buildStencil()
{
	; // stencil_.resize( 1, &ns_ );
}

