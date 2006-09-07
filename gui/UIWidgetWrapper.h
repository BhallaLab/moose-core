/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _UIWidgetWrapper_h
#define _UIWidgetWrapper_h
class UIWidgetWrapper: 
	public UIWidget, public Neutral
{
	friend Element* uiActionConnUIWidgetLookup( const Conn* );
    public:
		UIWidgetWrapper(const string& n)
		:
			Neutral( n ),
			uiActionSrc_( &uiActionConn_ )
			// uiActionConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setDefaultBase( Conn* c, string value ) {
			static_cast< UIWidgetWrapper* >( c->parent() )->defaultBase_ = value;
		}
		static string getDefaultBase( const Element* e ) {
			return static_cast< const UIWidgetWrapper* >( e )->defaultBase_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getUiActionSrc( Element* e ) {
			return &( static_cast< UIWidgetWrapper* >( e )->uiActionSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void uiActionFuncLocal(  ) {
		}
		static void uiActionFunc( Conn* c ) {
			static_cast< UIWidgetWrapper* >( c->parent() )->
				uiActionFuncLocal(  );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getUiActionConn( Element* e ) {
			return &( static_cast< UIWidgetWrapper* >( e )->uiActionConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const UIWidget* p = dynamic_cast<const UIWidget *>(proto);
			// if (p)... and so on. 
			return new UIWidgetWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc0 uiActionSrc_;
		UniConn< uiActionConnUIWidgetLookup > uiActionConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _UIWidgetWrapper_h
