/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2004 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include <set>
#include "header.h"

//////////////////////////////////////////////////////////////////////

// ReversePiggyback is triggered from the destination of a msg,
// and invites the src to send the message.
// This is allowed only if:
// - The msgsrc can do a Send()... CanLookup() is true for the field.
// - The msg has only one src (?)
/*
bool ReversePiggyback(msgdestlist *dl)
{
	if (dl->NSrc() == 0)
		return 0 ;
	for (unsigned int i = 0; i < dl->NSrc(); i++) {
		const finfo* f = dl->Src(i)->Field();
		if (f->CanLookup()) {
			f->Send(dl->Src(i)->Parent(), dl);
			return 1;
		}
	}
	return 0;
}

void Piggyback(const msgsrclist *csl, const string& mdest) {
	vector<msgdestlist*>::iterator dl;
	msgsrclist* sl = const_cast<msgsrclist *>(csl);
	for (dl = sl->begin(); dl != sl->end(); dl++) {
		Call((*dl)->Parent(), mdest);
	}
}
*/


// Goes through all first entries on tree, looks for outgoing msgs,
// checks if the targets of these messages are on the tree.
// If so, it makes an identical message between the corresponding
// second entries which comprise the duplicate of the original tree.

// Also need to go through all generic messages.
void duplicateMessagesOnTree(map<const Element*, Element*>& tree)
{
	map<const Element*, Element*>::iterator ti;

	for (ti = tree.begin(); ti != tree.end(); ti++) {
		// Corresponding parts of tree hould be of identical classes.
		if ( ti->first->cinfo() != ti->second->cinfo() ) {
			cout << "Error: duplicateMessgesOnTree(): Tree mismatch\n";
			return;
		}
	}
	for (ti = tree.begin(); ti != tree.end(); ti++) {
		vector< Finfo* > finfos;
		ti->first->cinfo()->listFields( finfos );
		for (unsigned int i = 0; i < finfos.size(); i++ ) {
			Element* temp = const_cast< Element* >( ti->first );
			Field f1( finfos[i], temp );
			vector< Field > dests;
			f1.dest( dests );
			// Scan dests for entries in tree but avoid kids and 
			// predefined msgs.
			// Connect up to corresponding entries in tree
			// f1 = ti->first->GetCinfo()->GetField(++i);
			// f2 = ti->second->GetCinfo()->GetField(i);
		}
	}
}

/*
// Duplicates any message going from f1 to outside. The corresponding
// message is set up from f2 to outside.
void DuplicateOutgoing(const finfo* f1, const finfo* f2, 
	set<const element*> &outside,
	const element* e1, element* e2)
{
	// Connect up messages going from tree to outside
	const src_field* sf1 = dynamic_cast<const src_field *>(f1);
	const src_field* sf2 = dynamic_cast<const src_field *>(f2);
	if (!sf1 || sf1 != sf2)
		return;
	msgsrclist* m1 = const_cast<msgsrclist*>(
		sf1->Msg(const_cast<element *>(e1))
	);
	msgsrclist* m2 = const_cast<msgsrclist*>(sf2->Msg(e2));
	
	for (unsigned int j = 0; j < m1->NDest(); j++) {
		const msgdestlist* old_d = m1->Dest(j);
		if ( outside.find(old_d->Parent()) != outside.end() ) {
			m2->Add(const_cast<msgdestlist *>(old_d));
		}
	}
}

// Duplicates any message going from outside to f1. The corresponding
// message is set up from outside to f2.
void DuplicateIncoming(const finfo* f1, const finfo* f2, 
	set<const element*> &outside,
	const element* e1, element* e2)
{
	// Connect up messages going from outside to tree
	const dest_field* df1 = dynamic_cast<const dest_field *>(f1);
	const dest_field* df2 = dynamic_cast<const dest_field *>(f2);
	if (!df1 || df1 != df2)
		return;
	msgdestlist* m1 = const_cast<msgdestlist*>(
		df1->Msg(const_cast<element *>(e1))
	);
	msgdestlist* m2 = const_cast<msgdestlist*>( df2->Msg(e2) );
	
	for (unsigned int j = 0; j < m1->NSrc(); j++) {
		const msgsrclist* old_s = m1->Src(j);
		if ( outside.find(old_s->Parent()) != outside.end() ) {
			if (dynamic_cast<const singlemsgsrc *>(old_s) &&
				old_s->Field()->CanLookup()) {
			//	cerr << "Probably a genmsgsrc to the tree\n";
				element* e = const_cast<element *>(old_s->Parent());

				AddGenericMsgSrc(e, old_s->Field(), e2, f2);
			} else {
				const_cast<msgsrclist *>(old_s)->Add(m2);
			}
		}
	}
}

// Takes tree A and its duplicate tree B. A connects to outside tree T,
// and this function conects B equivalently to this outside tree T.
// A and B are in the map called 'tree'. 
// Still need to handle messages coming in to generic msg dests.
void DuplicateMessagesOutsideTree(map<const element*, element*>& tree,
	set<const element*> outside)
{
	map<const element*, element*>::iterator ti;

	for (ti = tree.begin(); ti != tree.end(); ti++) {
		unsigned int i = 0;
		const finfo* f1 = ti->first->GetCinfo()->GetField(i);
		const finfo* f2 = ti->second->GetCinfo()->GetField(i);
		while (f1) {
			DuplicateOutgoing(f1, f2, outside, ti->first, ti->second);
			DuplicateIncoming(f1, f2, outside, ti->first, ti->second);

			f1 = ti->first->GetCinfo()->GetField(++i);
			f2 = ti->second->GetCinfo()->GetField(i);
		}
	}
}
*/
