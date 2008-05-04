/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FID_H
#define _FID_H

/**
 * Utility class to hold all the information necessary to identify a
 * field on a specific object.
 */
class Fid
{
		public:
			/**
			 * Define the Fid from the Id and the field name
			 */
			Fid();
			Fid( Id id, const string& fname );
			Fid( const string& name );
			Fid( Id id, int f );

			Id id() const {
				return id_;
			}

			Eref eref() const {
				return id_.eref();
			}

			const Finfo* finfo();

			int fieldNum() const {
				return fieldNum_;
			}

			string fieldName() const;

			string name() const;

		private:
			Id id_;
			int fieldNum_;
};

#endif // _FID_H
