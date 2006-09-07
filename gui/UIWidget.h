/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _UIWidget_h
#define _UIWidget_h
class UIWidget
{
	friend class UIWidgetWrapper;
	public:
		UIWidget()
		{
			defaultBase_ = "/output";
		}

	private:
		string defaultBase_;
};
#endif // _UIWidget_h
