/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ARRAY_WRAPPER_ELEMENT_H
#define _ARRAY_WRAPPER_ELEMENT_H

/**
 * The SimpleElement class implements Element functionality in the
 * most common vanilla way. It manages a set of vectors and pointers
 * that keep track of messaging and field information.
 */
class ArrayWrapperElement: public SimpleElement
{
	public:
		ArrayWrapperElement( Element* arrayElement, unsigned int index );

		/// This cleans up the data_ and finfo_ if needed.
		~ArrayWrapperElement();

		const Finfo* findFinfo( const string& name );

		void* data() const;

		unsigned int numEntries() const;

	private:
		Element* arrayElement_;
		unsigned int index_;
};

#endif // _ARRAY_WRAPPER_ELEMENT_H
