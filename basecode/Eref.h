/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _EREF_H
#define _EREF_H

class Eref
{
	public:
		friend ostream& operator <<( ostream& s, const Eref& e );
		Eref( Element* e, DataId index );

		/**
		 * Returns data entry
		 */
		char* data() const;

		/**
		 * Returns data entry for parent object. Relevant for
		 * FieldDataHandlers.
		 */
		char* parentData() const;

		/**
		 * Returns data entry of parent object of field array
		 */

		/**
		 * Returns Element part
		 */
		Element* element() const {
			return e_;
		}

		/**
		 * Returns index part
		 */
		DataId index() const {
			return i_;
		}

		/**
		 * Returns the ObjId corresponding to the Eref. All info is kept.
		 */
		ObjId objId() const;

		/**
		 * Returns the Id corresponding to the Eref. Loses information.
		 */
		Id id() const;

		/**
		 * True if the data are on the current node
		 */
		bool isDataHere() const;
	private:
		Element* e_;
		DataId i_;
};

#endif // _EREF_H
