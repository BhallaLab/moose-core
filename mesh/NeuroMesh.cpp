/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cctype>
#include "header.h"
#include "SparseMatrix.h"
#include "Vec.h"

#include "ElementValueFinfo.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "ChemCompt.h"
#include "MeshCompt.h"
#include "CubeMesh.h"
#include "CylBase.h"
#include "NeuroNode.h"
#include "NeuroMesh.h"
#include "SpineEntry.h"
#include "../utility/numutil.h"
#include "../shell/Wildcard.h"

static SrcFinfo4< Id, vector< Id >, vector< Id >, vector< unsigned int > >* 
	spineListOut()
{
	static SrcFinfo4< Id, vector< Id >, vector< Id >, vector< unsigned int > >
   		spineListOut(
		"spineListOut",
		"Request SpineMesh to construct self based on list of electrical "
		"compartments that this NeuroMesh has determined are spine shaft "
		"and spine head respectively. Also passes in the info about where "
		"each spine is connected to the NeuroMesh. "
		"Arguments: Cell Id, shaft compartment Ids, head compartment Ids,"
		"index of matching parent voxels for each spine"
	);
	return &spineListOut;
}

static SrcFinfo3< Id, vector< double >, vector< unsigned int > >* 
	psdListOut()
{
	static SrcFinfo3< Id, vector< double >, vector< unsigned int > >
   		psdListOut(
		"psdListOut",
		"Tells PsdMesh to build a mesh. "
		"Arguments: Cell Id, Coordinates of each psd, "
		"index of matching parent voxels for each spine"
		"The coordinates each have 8 entries:"
		"xyz of centre of psd, xyz of vector perpendicular to psd, "
		"psd diameter, "
		" diffusion distance from parent compartment to PSD"
	);
	return &psdListOut;
}

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
		static ElementValueFinfo< NeuroMesh, Id > cell(
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
		static ValueFinfo< NeuroMesh, bool > separateSpines(
			"separateSpines",
			"Flag: when separateSpines is true, the traversal separates "
			"any compartment with the strings "
			"'spine', 'head', 'shaft' or 'neck' in its name,"
			"Allows to set up separate mesh for spines, based on the "
			"same cell model. Requires for the spineListOut message to"
			"be sent to the target SpineMesh object.",
			&NeuroMesh::setSeparateSpines,
			&NeuroMesh::getSeparateSpines
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

		static ValueFinfo< NeuroMesh, string > geometryPolicy(
			"geometryPolicy",
			"Policy for how to interpret electrical model geometry (which "
			"is a branching 1-dimensional tree) in terms of 3-D constructs"
			"like spheres, cylinders, and cones."
			"There are three options, default, trousers, and cylinder:"
			"default mode:"
	" - Use frustrums of cones. Distal diameter is always from compt dia."
	" - For linear dendrites (no branching), proximal diameter is "
	" diameter of the parent compartment"
	" - For branching dendrites and dendrites emerging from soma,"
	" proximal diameter is from compt dia. Don't worry about overlap."
	" - Place somatic dendrites on surface of spherical soma, or at ends"
	" of cylindrical soma"
	" - Place dendritic spines on surface of cylindrical dendrites, not"
	" emerging from their middle."
	"trousers mode:"
	" - Use frustrums of cones. Distal diameter is always from compt dia."
	" - For linear dendrites (no branching), proximal diameter is "
	" diameter of the parent compartment"
	" - For branching dendrites, use a trouser function. Avoid overlap."
	" - For soma, use some variant of trousers. Here we must avoid overlap"
	" - For spines, use a way to smoothly merge into parent dend. Radius of"
	" curvature should be similar to that of the spine neck."
	" - Place somatic dendrites on surface of spherical soma, or at ends"
	" of cylindrical soma"
	" - Place dendritic spines on surface of cylindrical dendrites, not"
	" emerging from their middle."
	"cylinder mode:"
	" - Use cylinders. Diameter is just compartment dia."
	" - Place somatic dendrites on surface of spherical soma, or at ends"
	" of cylindrical soma"
	" - Place dendritic spines on surface of cylindrical dendrites, not"
	" emerging from their middle."
	" - Ignore spatial overlap.",
			&NeuroMesh::setGeometryPolicy,
			&NeuroMesh::getGeometryPolicy
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo setCellPortion( "setCellPortion",
			"Tells NeuroMesh to mesh up a subpart of a cell. For now"
			"assumed contiguous."
			"The first argument is the cell Id. The second is the vector"
			"of Ids to consider in meshing up the subpart.",
			new OpFunc2< NeuroMesh, Id, vector< Id > >(
				&NeuroMesh::setCellPortion )
		);

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////

	static Finfo* neuroMeshFinfos[] = {
		&cell,			// Value
		&subTree,		// Value
		&separateSpines,	// Value
		&numSegments,		// ReadOnlyValue
		&numDiffCompts,		// ReadOnlyValue
		&diffLength,			// Value
		&geometryPolicy,		// Value
		&setCellPortion,			// DestFinfo
		spineListOut(),			// SrcFinfo
		psdListOut(),			// SrcFinfo
	};

	static Cinfo neuroMeshCinfo (
		"NeuroMesh",
		ChemCompt::initCinfo(),
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
		nodes_(1),
		nodeIndex_(1, 0 ),
		vs_( 1, NA * 1e-9 ),
		area_( 1, 1.0e-12 ),
		size_( 1.0e-18 ),
		diffLength_( 0.5e-6 ),
		separateSpines_( false ),
		geometryPolicy_( "default" ),
		surfaceGranularity_( 0.1 )
{
	nodes_[0].setLength( diffLength_ );
	nodes_[0].setDia( diffLength_ );
	nodes_[0].setNumDivs( 1 );
}

NeuroMesh::NeuroMesh( const NeuroMesh& other )
	:
		size_( other.size_ ),
		diffLength_( other.diffLength_ ),
		cell_( other.cell_ ),
		separateSpines_( other.separateSpines_ ),
		geometryPolicy_( other.geometryPolicy_ ),
		surfaceGranularity_( other.surfaceGranularity_ )
{;}

NeuroMesh& NeuroMesh::operator=( const NeuroMesh& other )
{
	nodes_ = other.nodes_;
	nodeIndex_ = other.nodeIndex_;
	vs_ = other.vs_;
	area_ = other.area_;
	size_ = other.size_;
	diffLength_ = other.diffLength_;
	cell_ = other.cell_;
	separateSpines_ = other.separateSpines_;
	geometryPolicy_ = other.geometryPolicy_;
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
 * Assumes that the soma node is at index 0.
 */
void NeuroMesh::updateCoords()
{
	unsigned int startFid = 0;
	for ( vector< NeuroNode >::iterator i = nodes_.begin();
				i != nodes_.end(); ++i ) {
		if ( !i->isDummyNode() ) {
			double len = i->getLength();
			unsigned int numDivs = floor( 0.5 + len / diffLength_ );
			if ( numDivs < 1 ) 
				numDivs = 1;
			i->setNumDivs( numDivs );
			i->setStartFid( startFid );
			startFid += numDivs;
		}
	}
	nodeIndex_.resize( startFid );
	for ( unsigned int i = 0; i < nodes_.size(); ++i ) {
		if ( !nodes_[i].isDummyNode() ) {
			unsigned int end = nodes_[i].startFid() +nodes_[i].getNumDivs();
			assert( end <= startFid );
			assert( nodes_[i].getNumDivs() > 0 );
			for ( unsigned int j = nodes_[i].startFid(); j < end; ++j )
				nodeIndex_[j] = i;
		}
	}
	// Assign volumes and areas
	vs_.resize( startFid );
	area_.resize( startFid );
	for ( unsigned int i = 0; i < nodes_.size(); ++i ) {
		const NeuroNode& nn = nodes_[i];
		if ( !nn.isDummyNode() ) {
			assert( nn.parent() < nodes_.size() );
			const NeuroNode& parent = nodes_[ nn.parent() ];
			for ( unsigned int j = 0; j < nn.getNumDivs(); ++j ) {
				vs_[j + nn.startFid()] = NA * nn.voxelVolume( parent, j );
				area_[j + nn.startFid()] = nn.getDiffusionArea( parent, j );
			}
		}
	}
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

void NeuroMesh::setGeometryPolicy( string v )
{
	// STL magic! Converts the string v to lower case. Unfortunately
	// someone has overloaded tolower so it can have one or two args, so
	// this marvellous construct is useless.
	// std::transform( v.begin(), v.end(), v.begin(), std::tolower );
	for( string::iterator i = v.begin(); i != v.end(); ++i )
		*i = tolower( *i );

	if ( !( v == "cylinder" || v == "trousers" || v == "default" ) ) {
		cout << "Warning: NeuroMesh::setGeometryPolicy( " << v << 
			" ):\n Mode must be one of cylinder, trousers, or default."
			"Using default\n";
		v = "default";
	}

	if ( v == geometryPolicy_ )
			return;
	geometryPolicy_ = v;
   	bool isCylinder = ( v == "cylinder" );
	for ( vector< NeuroNode >::iterator 
			i = nodes_.begin(); i != nodes_.end(); ++i )
			i->setIsCylinder( isCylinder );
	/*
	if ( cell_ != Id() )
		setCell( cell_ );
		*/
}

string NeuroMesh::getGeometryPolicy() const
{
	return geometryPolicy_;
}

unsigned int NeuroMesh::innerGetDimensions() const
{
	return 3;
}

Id tryParent( Id id, const string& msgName )
{
	const Finfo* finfo = id.element()->cinfo()->findFinfo( msgName );
	if ( !finfo )
		return Id();
	vector< Id > ret;
	id.element()->getNeighbours( ret, finfo );
	assert( ret.size() <= 1 );
	if ( ret.size() == 1 )
		return ret[0];
	return Id();
}

Id getParentFromMsg( Id id )
{
	if ( id.element()->cinfo()->isA( "Compartment" ) )
		return tryParent( id, "axialOut" );
	if ( id.element()->cinfo()->isA( "SymCompartment" ) )
		return tryParent( id, "proximalOut" );
	return Id();
}

Id NeuroMesh::putSomaAtStart( Id origSoma, unsigned int maxDiaIndex )
{
	Id soma = origSoma;
	if ( nodes_[maxDiaIndex].elecCompt() == soma ) { // Happy, happy
		;
	} else if ( soma == Id() ) {
		soma = nodes_[maxDiaIndex].elecCompt();
	} else { // Disagreement. Ugh.
		string name = nodes_[ maxDiaIndex ].elecCompt().element()->getName();
		// OK, somehow this model has more than one soma compartment.
		if ( strncasecmp( name.c_str(), "soma", 4 ) == 0 ) {
			soma = nodes_[maxDiaIndex].elecCompt();
		} else { 
			cout << "Warning: named 'soma' compartment isn't biggest\n";
			soma = nodes_[maxDiaIndex].elecCompt();
		}
	}
	// Move soma to start of nodes_ vector.
	if ( maxDiaIndex != 0 ) {
		NeuroNode temp = nodes_[0];
		nodes_[0] = nodes_[maxDiaIndex];
		nodes_[maxDiaIndex] = temp;
	}
	return soma;
}

void NeuroMesh::insertSingleDummy( 
				unsigned int parent, unsigned int self,
			   double x, double y, double z	)
{
	static const double EPSILON = 1e-8;
	NeuroNode dummy( nodes_[ self ] );
	dummy.clearChildren();
	dummy.setNumDivs( 0 );
	bool isCylinder = (geometryPolicy_ == "cylinder" );
	dummy.setIsCylinder( isCylinder );
	dummy.setX( x );
	dummy.setY( y );
	dummy.setZ( z );
	// Now insert the dummy as a surrogate parent.
	dummy.setParent( parent );
	dummy.addChild( self );
	nodes_[ self ].setParent( nodes_.size() );
	// Idiot check for a bad dimensioned compartment.
	if ( nodes_[self].calculateLength( dummy ) < EPSILON ) {
		double length = nodes_[self].getLength();
		dummy.setX( x - length );
		double temp = nodes_[self].calculateLength( dummy );
		assert( doubleEq( temp, length ) );
	}
	nodes_.push_back( dummy );
}

void NeuroMesh::insertDummyNodes()
{
	// First deal with the soma, always positioned at node 0.
	Id elec = nodes_[0].elecCompt();
	double x = Field< double >::get( elec, "x0" );
	double y = Field< double >::get( elec, "y0" );
	double z = Field< double >::get( elec, "z0" );
	insertSingleDummy( ~0U, 0, x, y, z );

	// Second pass: insert dummy nodes for children.
	// Need to know if parent has multiple children, because each of
	// them will need a dummyNode to connect to.
	// In all the policies so far, the dummy nodes take the same diameter
	// as the children that they host.
	for ( unsigned int i = 0; i < nodes_.size(); ++i ) {
		vector< unsigned int > kids = nodes_[i].children();
		if ( (!nodes_[i].isDummyNode()) && kids.size() > 1 ) {
			for( unsigned int j = 0; j < kids.size(); ++j ) {
				x = nodes_[i].getX(); // use coords of parent.
				y = nodes_[i].getY();
				z = nodes_[i].getZ();
				insertSingleDummy( i, kids[j], x, y, z );
				// Replace the old kid entry with the dummy
				kids[j] = nodes_.size() - 1; 
			}
			// Connect up the parent to the dummy nodes.
			nodes_[i].clearChildren();
			for( unsigned int j = 0; j < kids.size(); ++j )
				nodes_[i].addChild( kids[j] );
		}
	}
}

/*
void NeuroMesh::buildNodeTree( const map< Id, unsigned int >& comptMap )
{
	const double EPSILON = 1e-8;
		// First pass: just build up the tree.
	bool isCylinder = (geometryPolicy_ == "cylinder" );
	unsigned int numOrigNodes = nodes_.size();
	if ( numOrigNodes == 0 )
		return;
	for ( unsigned int i = 0; i < numOrigNodes; ++i ) {
		// returns Id() if no parent found.
		Id pa = getParentFromMsg( nodes_[i].elecCompt() ); 
		if ( pa != Id() ) {
			map< Id, unsigned int >::const_iterator mapEntry = 
					comptMap.find( pa );
			assert( mapEntry != comptMap.end() );
			unsigned int ipa = mapEntry->second;
			// unsigned int ipa = comptMap[pa];
			nodes_[i].setParent( ipa );
			nodes_[ipa].addChild( i );
		} else { 
			// Here we need to track the coords of the other end 
			// 	of the parent-less compartment. It is typically a soma, 
			// 	but not always. These coords get assigned to a dummy node.
			NeuroNode dummy( nodes_[ i ] );
			dummy.clearChildren();
			dummy.setNumDivs( 0 ); // Identifies it as a dummy.
			dummy.setIsCylinder( isCylinder );
			Id elec = nodes_[i].elecCompt();
			assert( elec.element()->cinfo()->isA( "Compartment" ) );
			dummy.setX( Field< double >::get( elec, "x0" ) );
			dummy.setY( Field< double >::get( elec, "y0" ) );
			dummy.setZ( Field< double >::get( elec, "z0" ) );
			// This dummy has no parent. Use self index for parent.
			dummy.setParent( nodes_.size() );
			dummy.addChild( i );
			nodes_[i].setParent( nodes_.size() );
			double length = nodes_[i].getLength();
			// Idiot check for a bad dimensioned compartment.
			if ( nodes_[i].calculateLength( dummy ) < EPSILON ) {
				dummy.setX( -length );
				double temp = nodes_[i].calculateLength( dummy );
				assert( doubleEq( temp, length ) );
			}
			nodes_.push_back( dummy );
		}
	}
	// Second pass: insert dummy nodes.
	// Need to know if parent has multiple children, because each of
	// them will need a dummyNode to connect to.
	// In all the policies so far, the dummy nodes take the same diameter
	// as the children that they host.
	for ( unsigned int i = 0; i < nodes_.size(); ++i ) {
		vector< unsigned int > kids = nodes_[i].children();
		if ( (!nodes_[i].isDummyNode()) && kids.size() > 1 ) {
			for( unsigned int j = 0; j < kids.size(); ++j ) {
				NeuroNode dummy( nodes_[ kids[j] ] );
				dummy.clearChildren();
				dummy.setNumDivs( 0 );
				dummy.setIsCylinder( isCylinder );
				dummy.setX( nodes_[i].getX() ); // Use coords of parent.
				dummy.setY( nodes_[i].getY() );
				dummy.setZ( nodes_[i].getZ() );
				// Now insert the dummy as a surrogate parent.
				dummy.setParent( i );
				dummy.addChild( kids[j] );
				nodes_[ kids[j] ].setParent( nodes_.size() );
				kids[j] = nodes_.size(); // Replace the old kid entry with the dummy
				nodes_.push_back( dummy );
			}
			// Connect up the parent to the dummy nodes.
			nodes_[i].clearChildren();
			for( unsigned int j = 0; j < kids.size(); ++j )
				nodes_[i].addChild( kids[j] );
		}
	}
}
*/

bool NeuroMesh::filterSpines( Id compt )
{
	if ( compt.element()->getName().find( "shaft" ) != string::npos ||
		compt.element()->getName().find( "neck" ) != string::npos ) {
		shaft_.push_back( compt );
		return true;
	}
	if ( compt.element()->getName().find( "spine" ) != string::npos ||
		compt.element()->getName().find( "head" ) != string::npos ) {
		head_.push_back( compt );
		return true;
	}
	return false;
}

// I assume 'cell' is the parent of the compartment tree.
void NeuroMesh::setCell( const Eref& e, const Qinfo* q, Id cell )
{
	double oldVol = getMeshEntrySize( 0 );
	vector< Id > compts;
	wildcardFind( cell.path() + "/##", compts );
	setCellPortion( cell, compts );
	transmitChange( e, q, oldVol );
	if ( separateSpines_ ) {
		// unsigned int thread = q->threadNum();
		// Hack to send data directly, bypassing queueueue
		vector< Id > ids;
		e.element()->getNeighbours( ids, spineListOut() );
		if ( ids.size() > 0 ) {
			SetGet4< Id, vector< Id >, vector< Id >, 
					vector< unsigned int > >::set( 
					ids[0], "spineList", cell_, shaft_, head_, parent_ );
		}
		// spineListOut()->send( e, thread, cell_, shaft_, head_, parent_ );
		vector< double > ret;
		vector< double > psdCoords;
		vector< unsigned int > index( head_.size(), 0 );
		if ( e.element()->hasMsgs( psdListOut()->getBindIndex() ) ) {
			// Later can refine to deal with spineless PSDs.
			for ( unsigned int i = 0; i < head_.size(); ++i ) {
				SpineEntry se = 
						SpineEntry( shaft_[i], head_[i], parent_[i] );
				ret = se.psdCoords();
				assert( ret.size() == 8 );
				psdCoords.insert( psdCoords.end(), ret.begin(), ret.end() );
				index[i] = i;
			}
			ids.clear();
			e.element()->getNeighbours( ids, psdListOut() );
			// psdListOut()->send( e, thread, cell_, psdCoords, index );
			SetGet3< Id, vector< double >, vector< unsigned int > >::set( 
					ids[0], "psdList", cell_, psdCoords, index );
		}
	}
}


// Here we set a portion of a cell, specified by a vector of Ids. We
// also need to define the cell parent.
void NeuroMesh::setCellPortion( Id cell, vector< Id > portion )
{
		cell_ = cell;
		NeuroNode::buildTree( nodes_, portion );
		if ( separateSpines_ )
			NeuroNode::filterSpines( nodes_, shaft_, head_, parent_ );
		/*
		vector< Id >& compts = portion;
		map< Id, unsigned int > comptMap;

		Id soma;
		double maxDia = 0.0;
		unsigned int maxDiaIndex = 0;
		nodes_.resize( 0 );
		bool isCylinder = ( geometryPolicy_ == "cylinder" );
		for ( unsigned int i = 0; i < compts.size(); ++i ) {
			if ( compts[i].element()->cinfo()->isA( "Compartment" ) ) {
				if ( separateSpines_ && filterSpines( compts[i] ) )
					continue;
				comptMap[ compts[i] ] = nodes_.size();
				nodes_.push_back( NeuroNode( compts[i] ) );
				if ( nodes_.back().getDia() > maxDia ) {
					maxDia = nodes_.back().getDia();
					maxDiaIndex = nodes_.size() -1;
				}
				nodes_.back().setIsCylinder( isCylinder );
				string name = compts[i].element()->getName();
				if ( strncasecmp( name.c_str(), "soma", 4 ) == 0 ) {
					soma = compts[i];
				}
			}
		}
		// Figure out soma compartment.
		soma = putSomaAtStart( soma, maxDiaIndex );

		// Assign parent and child compts to node entries.
		buildNodeTree( comptMap );
		*/
		insertDummyNodes();

		updateCoords();

		if ( separateSpines_ )
			updateShaftParents();

	

		/* Need to fill in: July 2013.
		if ( separateSpines_ )
			buildSpineList( comptMap );
			*/
}

/** 
 * converts the parents_ vector from identifying the parent NeuroNode
 * to identifying the parent voxel, for each shaft entry.
 */
void NeuroMesh::updateShaftParents()
{
	assert( shaft_.size() == parent_.size() );
	vector< unsigned int > pa = parent_;
	for ( unsigned int i = 0; i < shaft_.size(); ++i ) {
		const NeuroNode& nn = nodes_[ pa[i] ];
		double x0 = Field< double >::get( shaft_[i], "x0" );
		double y0 = Field< double >::get( shaft_[i], "y0" );
		double z0 = Field< double >::get( shaft_[i], "z0" );
		const NeuroNode& pn = nodes_[ nn.parent() ];
		unsigned int index = 0;
		double r = nn.nearest( x0, y0, z0, pn, index );
		if ( r >= 0.0 ) {
			parent_[i] = index + nn.startFid();
		} else {
			assert( 0 );
		}
	}
}

Id NeuroMesh::getCell( const Eref& e, const Qinfo* q ) const
{
		return cell_;
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

void NeuroMesh::setSeparateSpines( bool v )
{
		if ( v != separateSpines_ ) {
				separateSpines_ = v;
				updateCoords();
		}
}

bool NeuroMesh::getSeparateSpines() const
{
		return separateSpines_;
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

// Deprecated
vector< Id > spineVec( const vector< Id >& head )
{
	const Cinfo* ccinfo = Cinfo::find( "Compartment" );
	const Finfo* axialFinfo = ccinfo->findFinfo( "axialOut" ); // to pa
	const Finfo* raxialFinfo = ccinfo->findFinfo( "raxialOut" ); // to kids
	const Cinfo* scinfo = Cinfo::find( "SymCompartment" );
	const Finfo* r1Finfo = scinfo->findFinfo( "distalOut" ); // to kids
	const Finfo* r2Finfo = scinfo->findFinfo( "proximalOut" ); // to pa
	assert( r1Finfo );
	assert( r2Finfo );
	map< Id, Id > spineMap;
	for ( vector< Id >::const_iterator i = head.begin(); i != head.end(); ++i )
	{
		vector< Id > ret;
		const Element* e = i->element();
		if ( e->cinfo() == ccinfo ) {
			if ( e->getNeighbours( ret, axialFinfo ) ) {
				spineMap[ *i ] = ret[0];
			} else if ( e->getNeighbours( ret, raxialFinfo ) ) {
				spineMap[ *i ] = ret[0];
			} else {
				assert( 0 );
			}
		} else if ( e->cinfo() == scinfo ) {
			if ( e->getNeighbours( ret, r2Finfo ) ) {
				spineMap[ *i ] = ret[0];
			} else if ( e->getNeighbours( ret, r1Finfo ) ) {
				spineMap[ *i ] = ret[0];
			} else {
				assert( 0 );
			}
		} else {
			assert( 0 );
		}
	}
	vector< Id > temp;
	for ( vector< Id >::const_iterator i = head.begin(); i != head.end(); ++i){
		temp.push_back( spineMap[ *i ] );
	}
	return temp;
}

/// Deprecated. I don't trust the message traversal.
Id getSpineParent( Id spine, Id head )
{
	const Cinfo* ccinfo = Cinfo::find( "Compartment" );
	const Finfo* axialFinfo = ccinfo->findFinfo( "axialOut" ); // to pa
	const Finfo* raxialFinfo = ccinfo->findFinfo( "raxialOut" ); // to kids
	const Cinfo* scinfo = Cinfo::find( "SymCompartment" );
	const Finfo* r1Finfo = scinfo->findFinfo( "distalOut" ); // to kids
	const Finfo* r2Finfo = scinfo->findFinfo( "proximalOut" ); // to pa
	assert( r1Finfo );
	assert( r2Finfo );
	Id pa;
	Element* se = spine.element();
	vector< Id > ret;
	if ( se->cinfo() == ccinfo ) {
		if ( se->getNeighbours( ret, axialFinfo ) ) {
			if ( ret[0] != head ) {
				pa = ret[0];
			} else if ( se->getNeighbours( ret, raxialFinfo ) ) {
				assert( ret[0] != head );
				pa = ret[0];
			} 
		}
	} else if ( se->cinfo() == scinfo ) {
		if ( se->getNeighbours( ret, r2Finfo ) ) {
			if ( ret[0] != head ) {
				pa = ret[0];
			} else if ( se->getNeighbours( ret, r1Finfo ) ) {
				assert( ret[0] != head );
				pa = ret[0];
			} 
		}
	}

	return pa;
}

// Deprecated: Jul 2013.
void NeuroMesh::buildSpineList( const map< Id, unsigned int >& comptMap )
{
	assert( shaft_.size() == head_.size() );
	shaft_ = spineVec( head_ );
	parent_.resize( shaft_.size(), 0 );
	for ( unsigned int i = 0; i < shaft_.size(); ++i ) {
		Id pa = getSpineParent( shaft_[i], head_[i] );
		assert( pa != Id() );
		map< Id, unsigned int >::const_iterator q = comptMap.find( pa );
		assert( q != comptMap.end() );
		NeuroNode& nn = nodes_[ q->second ];
		double x0 = Field< double >::get( shaft_[i], "x0" );
		double y0 = Field< double >::get( shaft_[i], "y0" );
		double z0 = Field< double >::get( shaft_[i], "z0" );
		NeuroNode& pn = nodes_[ nn.parent() ];
		unsigned int index = 0;
		double r = nn.nearest( x0, y0, z0, pn, index );
		if ( r >= 0.0 ) {
			parent_[i] = index + nn.startFid();
		} else {
			assert( 0 );
		}
	}
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
	if ( nodeIndex_.size() == 0 ) 
		return 1.0; // A default value to use before init
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

/// Virtual function to return volume of mesh Entry, including
/// for diffusively coupled voxels from other solvers.
double NeuroMesh::extendedMeshEntrySize( unsigned int fid ) const
{
	if ( fid < nodeIndex_.size() ) {
		return getMeshEntrySize( fid );
	} else {
		return MeshCompt::extendedMeshEntrySize( fid - nodeIndex_.size() );
	}
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
	double oldVol = getMeshEntrySize( 0 );
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
		oldVol,
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

void NeuroMesh::transmitChange( const Eref& e, const Qinfo* q, double oldVol )
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
	assert( vols.size() == nodeIndex_.size() );
	for ( unsigned int i = 0; i < vols.size(); ++i ) {
		vols[i] = getMeshEntrySize( i );
	}

	// This message tells the Stoich about the new mesh, and also about
	// how it communicates with other nodes.
	meshSplit()->fastSend( e, q->threadNum(), 
		oldVol,
		vols, localIndices, 
		outgoingEntries, incomingEntries );

	// This func goes down to the MeshEntry to tell all the pools and
	// Reacs to deal with the new mesh. They then update the stoich.
	lookupEntry( 0 )->triggerRemesh( meshEntry.eref(), q->threadNum(), 
		oldVol, startEntry, localIndices, vols );
}

//////////////////////////////////////////////////////////////////
// Utility function to set up Stencil for diffusion
//////////////////////////////////////////////////////////////////

double NeuroMesh::getAdx( unsigned int i, unsigned int& parentFid ) const
{
	const NeuroNode &nn = nodes_[ nodeIndex_[i] ];
	const NeuroNode *pa = &nodes_[ nn.parent() ];
	double L1 = nn.getLength() / nn.getNumDivs();
	double L2 = L1;
	parentFid = i - 1;
	if ( nn.startFid() == i ) { 
		// We're at the start of the node, need to refer to parent for L
		const NeuroNode* realParent = pa;
		if ( pa->isDummyNode() ) {
			if ( pa->parent() == ~0U )
				return -1; // No diffusion, bail out.
			assert( pa->parent() < nodes_.size() );
			realParent = &nodes_[ realParent->parent() ];
			if ( realParent->isDummyNode() ) {
				// Still dummy. So we're at a terminus. No diffusion
				return -1;
			}
		}
		L2 = realParent->getLength() / realParent->getNumDivs();
		parentFid = realParent->startFid() + 
				realParent->getNumDivs() - 1;
	}
	assert( parentFid < nodeIndex_.size() );
	double length = 0.5 * (L1 + L2 );
	// Note that we use the parent node here even if it is a dummy.
	// It has the correct diameter.
	double adx = nn.getDiffusionArea( *pa, i - nn.startFid() ) / length;
	return adx;
}

void NeuroMesh::buildStencil()
{
// stencil_[0] = new NeuroStencil( nodes_, nodeIndex_, vs_, area_);
	setStencilSize( nodeIndex_.size(), nodeIndex_.size() );
	SparseMatrix< double > sm( nodeIndex_.size(), nodeIndex_.size() );
	vector< vector< double > > paEntry( nodeIndex_.size() );
	vector< vector< unsigned int > > paColIndex( nodeIndex_.size() );
	// It is very easy to set up the matrix using the parent as there is 
	// only one parent for every voxel.
	for ( unsigned int i = 0; i < nodeIndex_.size(); ++i ) {
		unsigned int parentFid;
		double adx = getAdx( i, parentFid );
		if ( adx < 0.0 ) // No diffusion, so don't put in any entry.
				continue;
		/*
		const NeuroNode &nn = nodes_[ nodeIndex_[i] ];
		const NeuroNode *pa = &nodes_[ nn.parent() ];
		double L1 = nn.getLength() / nn.getNumDivs();
		double L2 = L1;
		unsigned int parentFid = i - 1;
		if ( nn.startFid() == i ) { 
			// We're at the start of the node, need to refer to parent for L
			const NeuroNode* realParent = pa;
			if ( pa->isDummyNode() ) {
				realParent = &nodes_[ realParent->parent() ];
				if ( realParent->isDummyNode() ) {
					// Still dummy. So we're at a terminus. No diffusion
					continue;
				}
			}
			L2 = realParent->getLength() / realParent->getNumDivs();
			parentFid = realParent->startFid() + 
					realParent->getNumDivs() - 1;
		}
		assert( parentFid < nodeIndex_.size() );
		double length = 0.5 * (L1 + L2 );
		// Note that we use the parent node here even if it is a dummy.
		// It has the correct diameter.
		double adx = nn.getDiffusionArea( *pa, i - nn.startFid() ) / length;
		*/
		paEntry[ i ].push_back( adx );
		paColIndex[ i ].push_back( parentFid );
		// Now put in the symmetric entries.
		paEntry[ parentFid ].push_back( adx );
		paColIndex[ parentFid ].push_back( i );
	}

	// Now go through the paEntry and paColIndex and build sparse matrix.
	// We have to do this separately because the sparse matrix has to be
	// build up in row order, and sorted, whereas the entries above 
	// are random access.
	for ( unsigned int i = 0; i < nodeIndex_.size(); ++i ) {
		unsigned int num = paColIndex[i].size();
		vector< Ecol > e( num );
		vector< double > entry( num );
		vector< unsigned int > colIndex( num );
		for ( unsigned int j = 0; j < num; ++j ) {
			e[j] = Ecol( paEntry[i][j], paColIndex[i][j] );
		}
		sort( e.begin(), e.end() );

		for ( unsigned int j = 0; j < num; ++j ) {
			entry[j] = e[j].e_;
			colIndex[j] = e[j].col_;
		}
		addRow( i, entry, colIndex );
	}
	innerResetStencil();
}


const vector< NeuroNode >& NeuroMesh::getNodes() const
{
	return nodes_;
}

//////////////////////////////////////////////////////////////////
// Utility function for junctions
//////////////////////////////////////////////////////////////////

void NeuroMesh::matchMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
	const CubeMesh* cm = dynamic_cast< const CubeMesh* >( other );
	if ( cm ) {
		matchCubeMeshEntries( other, ret );
		return;
	}
	/*
	const SpineMesh* sm = dynamic_cast< const SpineMesh* >( other );
	if ( sm ) {
		matchSpineMeshEntries( other, ret );
		return;
	}
	*/
	const NeuroMesh* nm = dynamic_cast< const NeuroMesh* >( other );
	if ( nm ) {
		matchNeuroMeshEntries( other, ret );
		return;
	}
	cout << "Warning: NeuroMesh::matchMeshEntries: unknown class\n";
}

void NeuroMesh::indexToSpace( unsigned int index,
			double& x, double& y, double& z ) const 
{
	if ( index >= innerGetNumEntries() )
		return;
	const NeuroNode& nn = nodes_[ nodeIndex_[ index ] ];
	const NeuroNode& pa = nodes_[ nn.parent() ];
	Vec a( pa.getX(), pa.getY(), pa.getZ() );
	Vec b( nn.getX(), nn.getY(), nn.getZ() );
	double frac = ( ( index - nn.startFid() ) + 0.5 ) / nn.getNumDivs();
	Vec pt = a.pointOnLine( b, frac );
	x = pt.a0();
	y = pt.a1();
	z = pt.a2();
}

double NeuroMesh::nearest( double x, double y, double z, 
				unsigned int& index ) const
{
	double best = 1e12;
	index = 0;
	for( unsigned int i = 0; i < nodes_.size(); ++i ) {
		const NeuroNode& nn = nodes_[i];
		if ( !nn.isDummyNode() ) {
			assert( nn.parent() < nodes_.size() );
			const NeuroNode& pa = nodes_[ nn.parent() ];
			double linePos;
			double r;
			double near = nn.nearest( x, y, z, pa, linePos, r );
			if ( linePos >= 0 && linePos < 1.0 ) {
				if ( best > near ) {
					best = near;
					index = linePos * nn.getNumDivs() + nn.startFid();
				}
			}
		}
	}
	if ( best == 1e12 )
		return -1;
	return best;
}

void NeuroMesh::matchSpineMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
}

void NeuroMesh::matchCubeMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
	for( unsigned int i = 0; i < nodes_.size(); ++i ) {
		const NeuroNode& nn = nodes_[i];
		if ( !nn.isDummyNode() ) {
			assert( nn.parent() < nodes_.size() );
			const NeuroNode& pa = nodes_[ nn.parent() ];
			nn.matchCubeMeshEntries( other, pa, nn.startFid(), 
							surfaceGranularity_, ret, true, false );
		}
	}
}

void NeuroMesh::matchNeuroMeshEntries( const ChemCompt* other,
	   vector< VoxelJunction >& ret ) const
{
}
