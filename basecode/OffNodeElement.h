/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _OFF_NODE_ELEMENT_H
#define _OFF_NODE_ELEMENT_H

/**
 * This handles references to off-node elements and fields. It does 
 * so by maintaining the id of the off-node element, the postmaster
 * that leads to that node, and a string for the name of the 
 * field, if any. This Element does NOT create its own id upon
 * construction.
 */

class OffNodeElement: public Element
{
	public:
		OffNodeElement( unsigned int id, unsigned int node );

		~OffNodeElement();

		const std::string& name( ) const;
		void setName( const std::string& name );
		const std::string& className( ) const;
		vector< Conn >::const_iterator 
				lookupConn( unsigned int i ) const;
		vector< Conn >::iterator lookupVariableConn( unsigned int i );
		unsigned int connIndex( const Conn* ) const;
		unsigned int connDestRelativeIndex(
				const Conn& c, unsigned int slot ) const;
		unsigned int connSize() const;
		vector< Conn >::const_iterator
				connSrcBegin( unsigned int src ) const;
		vector< Conn >::const_iterator
				connSrcEnd( unsigned int src ) const;
		vector< Conn >::const_iterator
				connSrcVeryEnd( unsigned int src ) const;
		unsigned int nextSrc( unsigned int src ) const;
		vector< Conn >::const_iterator
				connDestBegin( unsigned int dest ) const;
		vector< Conn >::const_iterator
				connDestEnd( unsigned int dest ) const;
		void connect( unsigned int myConn,
			Element* targetElement, unsigned int targetConn);
		void disconnect( unsigned int connIndex );
		void deleteHalfConn( unsigned int connIndex );
		bool isMarkedForDeletion() const;
		bool isGlobal() const;
		void prepareForDeletion( bool stage );
		unsigned int insertConnOnSrc(
				unsigned int src, FuncList& rf,
				unsigned int dest, unsigned int nDest);
		unsigned int insertConnOnDest(
				unsigned int dest, unsigned int nDest);
		void* data() const;
		const Finfo* findFinfo( const string& name );
		const Finfo* findFinfo( unsigned int connIndex ) const;
		unsigned int listFinfos( vector< const Finfo* >& flist ) const;
		unsigned int listLocalFinfos( vector< Finfo* >& flist );
		void addFinfo( Finfo* f );
		bool isConnOnSrc( unsigned int src, unsigned int conn ) const;
		bool isConnOnDest( unsigned int dest, unsigned int conn ) const;
		Element* copy( Element* parent, const string& newName ) const;
		bool isDescendant( const Element* ancestor ) const;
		Element* innerDeepCopy( map< const Element*, Element* >& tree )
				const;
		void replaceCopyPointers(
						map< const Element*, Element* >& tree );
		void copyMsg( map< const Element*, Element* >& tree );

		/** 
		 * Finally we come to the actual useful part of OffNodeElement:
		 * the fields.
		 */
		unsigned int destId() const;
		unsigned int node() const;
		void setPost( Element* post );
		Element* post() const;
		void setFieldName( const string& name );
		const string& fieldName() const;

	protected:
		Element* innerCopy() const;
		bool innerCopyMsg( Conn& c, const Element* orig, Element* dup );

	private:
		unsigned int destId_;
		unsigned int node_;
		Element* post_;
		string fieldName_;
};

#endif // _OFF_NODE_ELEMENT_H
