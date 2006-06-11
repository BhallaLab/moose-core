#include "header.h"
#include "Cinfo.h"
#include "CinfoWrapper.h"

Finfo* CinfoWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ReadOnlyValueFinfo< string >(
		"name", &CinfoWrapper::getName, "string" ),
	new ReadOnlyValueFinfo< string >(
		"author", &CinfoWrapper::getAuthor, "string" ),
	new ReadOnlyValueFinfo< string >(
		"description", &CinfoWrapper::getDescription, "string" ),
	new ReadOnlyValueFinfo< string >(
		"baseName", &CinfoWrapper::getBaseName, "string" ),
	new ReadOnlyArrayFinfo< string >(
		"fields", &CinfoWrapper::getFields, "string" ),
};

const Cinfo CinfoWrapper::cinfo_(
	"Cinfo",
	"Upinder S. Bhalla, 2005, NCBS",
	"Cinfo: Cinfo class. Provides class information.",
	"Neutral",
	CinfoWrapper::fieldArray_,
	sizeof(CinfoWrapper::fieldArray_)/sizeof(Finfo *),
	&CinfoWrapper::create
);

///////////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////////

// Later we should elaborate to get more info about the fields.
string CinfoWrapper::getFields(
	const Element* e , unsigned long index )
{
	const CinfoWrapper* f = static_cast< const CinfoWrapper* >( e );
	if ( f->data_->nFields_ > index )
		return f->data_->fieldArray_[ index ]->name();
	return f->data_->fieldArray_[ 0 ]->name();
}
