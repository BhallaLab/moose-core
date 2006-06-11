/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <fstream>
#include "header.h"

//////////////////////////////////////////////////////////////////
// Neutrals are Elements that can have children. They have no 
// data fields. They equivalent to the GENESIS neutrals.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Neutral initialization
//////////////////////////////////////////////////////////////////

Finfo* Neutral::fieldArray_[] = 
{
	new NSrc0Finfo( "child_out", &Neutral::getChildOutSrc, "child_in" ),
	// This entry is almost identical to the one for the Element, 
	// except that now we have an internal trig target: child_out.
	new Dest0Finfo( "child_in",
		&Neutral::deleteFunc, &Element::getChildIn, "child_out")
};

const Cinfo Neutral::cinfo_(
	"Neutral",
	"Upinder S. Bhalla, NCBS",
	"Neutral class. Manages child Elements.",
	"Element",
	Neutral::fieldArray_,
	sizeof(Neutral::fieldArray_)/sizeof(Finfo *),
	&Neutral::create
);


//////////////////////////////////////////////////////////////////
// Neutral functions
//////////////////////////////////////////////////////////////////

Neutral::~Neutral()
{
	;
}

bool Neutral::adoptChild( Element* child )
{
	// this uses elementary operations for two reasons. First,
	// it is called at static initialization so cannot assume
	// anything about what is already set up.
	// Second, it is faster to go straight to the function calls.
	// This is why we cannot simply ask the fieldArray[] for the
	// delete function.

	// return childSrc_.add( deleteFunc, getChildIn( child ) );
	return childSrc_.add( child->getDeleteFunc(), getChildIn( child ) );

/*
	static Field paf = Cinfo::find( "Neutral" )->field( "child_out" );
	Field childf = child->cinfo()->field( "child_in" );

	childf.setElement( child );
	return paf->add( this, childf );
*/
}

// In this function we ignore the proto, as it has no data.
Element* Neutral::create(
			const string& name, Element* parent, const Element* proto)
{
	return new Neutral( name );
	/*
	Neutral* ret = new Neutral(name);
	if (parent->adoptChild(ret)) {
		return ret;
	} else {
		delete ret;
		return 0;
	}
	return 0;
	*/
}

// Finds an element relative to the current one based on path n
// The string n should not start with a /
Element* Neutral::relativeFind( const string& n )
{
	if (n == "") // end of the line
			return this;

	unsigned long pos = n.find( '/' );
	if (pos == 0) {
			// coping with the error condition
			return root()->relativeFind( n.substr(1) );
	}
	string head = n.substr(0, pos);
	if (head == "") { // there is nothing more
			return this;
	}
	if (head == "..") {
		Element* pa = parent();
		if (pos == string::npos)
			return pa;
		else 
			return pa->relativeFind( n.substr(pos + 1, string::npos));
	}
	if (head == ".") {
		if (pos == string::npos)
			return this;
		else
			return relativeFind(n.substr(pos + 1));
	}

	// The general case: /foo/bar/zod
	vector< Field > f;
	childSrc_.dest( f );
	for ( vector< Field >::iterator i = f.begin(); i != f.end(); i++) {
		if ( i->getElement()->name() == head ) {
			if ( pos == string::npos )
				return i->getElement();
			return i->getElement()->relativeFind( n.substr( pos + 1 ) );
		}
	}
	return 0;
}

Element* Neutral::internalDeepCopy(
	Element* pa, map<const Element*, Element* >& tree) const
{
	Element* e = cinfo()->create( name(), pa, this );
// 	if (pa->AdoptChild(e))
	if (e)
	{
		tree[this] = e;
		vector< Conn* > targets;
		childOut_.listTargets( targets );
		for (unsigned int i = 0; i < targets.size(); i++) {
			Element* child = targets[i]->parent();
			if (tree.find(child) != tree.end())
				cerr << "Warning: Loop in hierarchy at '" << 
					child->path() << "'\n";
			else 
				child->internalDeepCopy(e, tree);
		}
		return e;
	}
	delete e;
	return 0;
}


/*
// Builds a wildcard list based on path. Returns number found.
// We are in an element that matches.
// n is nonzero: there is more to come.
// If doublehash is set then the branches as well as leaves are used.
// Problem is that it currently returns duplicates.
int p_element::WildcardRelativeFind(
	const string& n, vector<element *>& ret, int doublehash)
{
	unsigned long pos = n.find('/');
	string head = n.substr(0, pos);
	element* e;
	int nret = 0;

	if (doublehash == 0 && head.find("##") != string::npos)
		doublehash = 1;

	for (unsigned int i = 0; i < kids.NDest(); i++) {
		e = const_cast<element *>(kids.Dest(i)->Parent());
		if (e->WildcardName(head)) {
			if (pos == string::npos) {
				ret.push_back(e);
				nret++;
			} else {
				nret += e->WildcardRelativeFind(
					n.substr(pos + 1, string::npos), ret, doublehash);
			}
		}
		if (doublehash) {
			nret += e->WildcardRelativeFind(n, ret, doublehash);
		}
	}
	return nret;
}

// Find child with specified name, return 0 on failure
element* p_element::FindNamedChild(const string& n) const
{
	for (unsigned int i = 0; i < kids.NDest(); i++) {
		if (kids.Dest(i)->Parent()->Name() == n)
			return const_cast<element *>(kids.Dest(i)->Parent());
	}
	return 0;
}

// Looks for identically named kids of this and pa, traverses
// tree with these recursively. Additionally, does filter check
// for this and pa, and if it passes, adds it to the tree.
void p_element::BuildMatchingTree(element* pa, filter *fil,
	map<const element*, element*> &tree) const {
	unsigned int i;

	p_element* p = dynamic_cast<p_element *>(pa);
	if (!p) { // Should never happen
		cerr << "Error: element::BuildMatchingTree: mismatch between "
			<< Path() << " and " << pa->Path() << "\n";
		return;
	}
	for (i = 0; i < kids.NDest(); i++) {
		const element* child = kids.Dest(i)->Parent();
		element *pchild = p->FindNamedChild(child->Name());
		if (pchild)
			child->BuildMatchingTree(pchild, fil, tree);
	}

	if (fil->do_filter(this) && fil->do_filter(pa))
		tree[this] = pa;
}
*/
