/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GLOBAL_MARKER_FINFO_H
#define _GLOBAL_MARKER_FINFO_H

/**
 * Finfo for reporting that host element is a global; should not
 * be copied.
 * Does not participate in anything else.
 */
class GlobalMarkerFinfo: public DeletionMarkerFinfo
{
		public:
			~GlobalMarkerFinfo()
			{;}


			Finfo* copy() const {
				return new GlobalMarkerFinfo( *this );
			}

			///////////////////////////////////////////////////////
			// Class-specific functions below
			///////////////////////////////////////////////////////

			/**
			 * This function allows us to place a single statically
			 * created GlobalMarkerFinfo wherever it is needed.
			 */
			static GlobalMarkerFinfo* global();

		private:
};

#endif // _GLOBAL_MARKER_FINFO_H
