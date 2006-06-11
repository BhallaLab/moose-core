#ifndef _CinfoWrapper_h
#define _CinfoWrapper_h
class CinfoWrapper: public Neutral
{
    public:
		CinfoWrapper(const string& n, Cinfo* data = 0)
		:
			Neutral( n ), data_( data )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static string getAuthor( const Element* e ) {
			return static_cast< const CinfoWrapper* >( e )->
				data_->author_;
		}
		static string getDescription( const Element* e ) {
			return static_cast< const CinfoWrapper* >( e )->
				data_->description_;
		}
		static string getBaseName( const Element* e ) {
			return static_cast< const CinfoWrapper* >( e )->
				data_->baseName_;
		}
		static string getFields(
			const Element* e, unsigned long index );

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			const CinfoWrapper* p = 
				dynamic_cast<const CinfoWrapper *>(proto);
			if (p)
				return new CinfoWrapper(name, p->data_);
			return new CinfoWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
		Cinfo* data_;
};
#endif // _CinfoWrapper_h
