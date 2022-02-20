// NSDFWriter2.cpp ---
//
// Filename: NSDFWriter2.cpp
// Description:
// Author: subha
// Maintainer:
// Created: Thu Jun 18 23:16:11 2015 (-0400)
// Version:
// Last-Updated: Sat Jan 29 2022
//           By: bhalla
//     Update #: 50
// URL:
// Keywords:
// Compatibility:
//

// Commentary:
//

// Change log:
// Jan 2022: Many changes added
//
//
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
//
//

// Code:
#ifdef USE_HDF5

#include "hdf5.h"
#include "hdf5_hl.h"

#include <fstream>
#include <ctime>
#include <cctype>
#include <deque>
#include "../basecode/header.h"
#include "../utility/utility.h"
#include "../utility/strutil.h"
#include "../shell/Wildcard.h"
#include "../shell/Shell.h"

#include "HDF5WriterBase.h"
#include "HDF5DataWriter.h"

#include "NSDFWriter.h"
#include "NSDFWriter2.h"
#include "InputVariable.h"

extern template herr_t writeScalarAttr(hid_t file_id, string path, string value);
extern template herr_t writeScalarAttr(hid_t file_id, string path, double value);

const char* const EVENTPATH = "/data/event";
const char* const UNIFORMPATH = "/data/uniform";
const char* const STATICPATH = "/data/static";
const char* const MODELTREEPATH = "/model/modeltree";
const char* const MODELFILEPATH = "/model/modelfile";
const char* const MAPUNIFORMSRC = "/map/uniform";
const char* const MAPSTATICSRC = "/map/static";
const char* const MAPEVENTSRC = "/map/event";

string iso_time(time_t * t);
/*
string iso_time(time_t * t)
{
    struct tm * timeinfo;
    if (t == NULL){
        time_t current;
        std::time(&current);
        timeinfo = std::gmtime(&current);
    } else {
        timeinfo = std::gmtime(t);
    }
    assert(timeinfo != NULL);
    char buf[32];
    strftime(buf, 32, "%FT%T", timeinfo);
    return string(buf);
}
*/

const Cinfo * NSDFWriter2::initCinfo()
{
    static FieldElementFinfo< NSDFWriter2, InputVariable > eventInputFinfo(
        "eventInput",
        "Sets up field elements for event inputs",
        InputVariable::initCinfo(),
        &NSDFWriter2::getEventInput,
        &NSDFWriter2::setNumEventInputs,
        &NSDFWriter2::getNumEventInputs);

    static ValueFinfo <NSDFWriter2, string > modelRoot(
      "modelRoot",
      "The moose element tree root to be saved under /model/modeltree. If blank, nothing is saved. Default: root object, '/'", 
      &NSDFWriter2::setModelRoot,
      &NSDFWriter2::getModelRoot);

    static ValueFinfo <NSDFWriter2, string > modelFileNames(
      "modelFileNames",
      "Comma separated list of model files to save into the NSDF file.",
      &NSDFWriter2::setModelFiles,
      &NSDFWriter2::getModelFiles);

    static ValueFinfo <NSDFWriter2, vector< string > > blocks(
      "blocks",
      "Vector of strings to specify data blocks. Format: path.field"
	  "The path is a wildcard path. It will be split into a single path"
	  "to a container such as a Neuron or a Mesh, and below this a "
	  "wildcard path to the actual objects",
      &NSDFWriter2::setBlocks,
      &NSDFWriter2::getBlocks);

    static DestFinfo process(
        "process",
        "Handle process calls. Collects data in buffer and if number of steps"
        " since last write exceeds flushLimit, writes to file.",
        new ProcOpFunc<NSDFWriter2>( &NSDFWriter2::process));

    static  DestFinfo reinit(
        "reinit",
        "Reinitialize the object. If the current file handle is valid, it tries"
        " to close that and open the file specified in current filename field.",
        new ProcOpFunc<NSDFWriter2>( &NSDFWriter2::reinit ));

    static Finfo * processShared[] = {
        &process, &reinit
    };

    static SharedFinfo proc(
        "proc",
        "Shared message to receive process and reinit",
        processShared, sizeof( processShared ) / sizeof( Finfo* ));

    static Finfo * finfos[] = {
        &eventInputFinfo,	// FieldElementFinfo
		&modelRoot,	// ValueFinfo
		&modelFileNames,	// ValueFinfo
		&blocks,	// ValueFinfo
        &proc,
    };

    static string doc[] = {
        "Name", "NSDFWriter2",
        "Author", "Subhasis Ray",
        "Description", "NSDF file writer for saving data."
    };

    static Dinfo< NSDFWriter2 > dinfo;
    static Cinfo cinfo(
        "NSDFWriter2",
        HDF5DataWriter::initCinfo(),
        finfos,
        sizeof(finfos)/sizeof(Finfo*),
        &dinfo,
	doc, sizeof( doc ) / sizeof( string ));

    return &cinfo;
}

static const Cinfo * nsdfWriterCinfo = NSDFWriter2::initCinfo();

NSDFWriter2::NSDFWriter2(): eventGroup_(-1), uniformGroup_(-1), dataGroup_(-1), modelGroup_(-1), mapGroup_(-1), modelRoot_("/")
{
    ;
}

NSDFWriter2::~NSDFWriter2()
{
    close();
}

void NSDFWriter2::close()
{
    if (filehandle_ < 0){
        return;
    }
    flush();
    closeUniformData();
    if (uniformGroup_ >= 0){
        H5Gclose(uniformGroup_);
    }
    closeEventData();
    if (eventGroup_ >= 0){
        H5Gclose(eventGroup_);
    }
    if (dataGroup_ >= 0){
        H5Gclose(dataGroup_);
    }
    HDF5DataWriter::close();
}

void NSDFWriter2::closeUniformData()
{
	for ( vector< Block >::iterator ii = blocks_.begin(); ii != blocks_.end(); ++ii ) {
		if ( ii->dataset >= 0 ) {
			H5Dclose( ii->dataset );
		}
	}
    vars_.clear();
    data_.clear();
    src_.clear();
    func_.clear();
    datasets_.clear();

}

void NSDFWriter2::sortMsgs(const Eref& eref)
{
    const Finfo* tmp = eref.element()->cinfo()->findFinfo("requestOut");
    const SrcFinfo1< vector < double > *>* requestOut = static_cast<const SrcFinfo1< vector < double > * > * >(tmp);
	vector< ObjId > tgts = eref.element()->getMsgTargets( eref.dataIndex(), requestOut );
	// Make a map from ObjId to index of obj in objVec.
	map< ObjId, unsigned int > mapMsgs;
	for (unsigned int tgtIdx = 0; tgtIdx < tgts.size(); ++tgtIdx)
		mapMsgs[ tgts[ tgtIdx ] ] = tgtIdx;

	mapMsgIdx_.resize( tgts.size() );
	unsigned int consolidatedBlockMsgIdx = 0;
	for (unsigned int blockIdx = 0; blockIdx < blocks_.size(); ++blockIdx) {
		vector< ObjId >&  objVec = blocks_[blockIdx].objVec;
		for ( auto obj = objVec.begin(); obj != objVec.end(); ++obj ) {
			mapMsgIdx_[consolidatedBlockMsgIdx] = mapMsgs[*obj];
			consolidatedBlockMsgIdx++;
		}
	}
	assert( tgts.size() == consolidatedBlockMsgIdx );
	// make a vector where tgtMsgIdx = mapMsgIdx_[consolidatedBlockMsgIdx]
}

void NSDFWriter2::buildUniformSources(const Eref& eref)
{
	Shell* shell = reinterpret_cast<Shell*>(Id().eref().data());
	for ( auto bb = blocks_.begin(); bb != blocks_.end(); ++bb ) {
		if ( bb->hasMsg )
			continue;
		const vector< ObjId >& objVec = bb->objVec;
		for( vector< ObjId >::const_iterator obj = objVec.begin(); obj != objVec.end(); ++obj ) {

			ObjId ret = shell->doAddMsg( "single", eref.objId(), "requestOut", *obj, bb->getField ); 
			if (ret == ObjId() ) {
				cout << "Error: NSDFWriter2::buildUniformSources: Failed to build msg from '" << eref.id().path() << "' to '" << bb->containerPath << "/" << bb->relPath << "." << bb->field << endl;
				return;
			}
		}

		bb->hasMsg = true;
	}
	sortMsgs( eref );
}

/**
   Handle the datasets for the requested fields (connected to
   requestOut). This is is similar to what HDF5DataWriter does.
 */
void NSDFWriter2::openUniformData(const Eref &eref)
{
    buildUniformSources(eref);
    htri_t exists;
    herr_t status;
    if (uniformGroup_ < 0){
        uniformGroup_ = require_group(filehandle_, UNIFORMPATH);
    }
	for ( auto bb = blocks_.begin(); bb != blocks_.end(); ++bb ) {
		if ( bb->hasContainer )
			continue;
		// From the documentation: 
		// https://support.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html
		// "Component link names may be any string of ASCII characters not containing a slash or a dot (/ and ., which are reserved as noted above)."
		// So I need to replace path with a string with the slashes
        bb->container = require_group(uniformGroup_, bb->nsdfContainerPath);
        bb->relPathContainer = require_group(bb->container,bb->nsdfRelPath);
        hid_t dataset = createDataset2D(bb->relPathContainer, bb->field.c_str(), bb->data.size());
		bb->dataset = dataset;
        writeScalarAttr<string>(dataset, "field", bb->field);
        H5Gclose(bb->container);
        H5Gclose(bb->relPathContainer);
		bb->hasContainer = true;
	}
}

/**
   create the DS for uniform data.
 */
void NSDFWriter2::createUniformMap()
{
	innerCreateMaps( MAPUNIFORMSRC );
}

/**
   create the DS for static data.
 */
void NSDFWriter2::createStaticMap()
{
	innerCreateMaps( MAPSTATICSRC );
}


/**
   Generic call for create the DS for static/uniform data
 */
void NSDFWriter2::innerCreateMaps( const char* const mapSrcStr )
{
    // Create the container for all the DS
    htri_t exists;
    herr_t status;
    hid_t uniformMapContainer = require_group(filehandle_, mapSrcStr );
    // Create the DS themselves
    for (map< string, vector < unsigned int > >::iterator ii = classFieldToSrcIndex_.begin();
         ii != classFieldToSrcIndex_.end(); ++ii){
        vector < string > pathTokens;
        moose::tokenize(ii->first, "/", pathTokens);
        string className = pathTokens[0];
        string fieldName = pathTokens[1];
		if (mapSrcStr == MAPSTATICSRC ) //Hack. for now only static field is coords
			fieldName = "coords";
        hid_t container = require_group(uniformMapContainer, className);
        char ** sources = (char **)calloc(ii->second.size(), sizeof(char*));
        for (unsigned int jj = 0; jj < ii->second.size(); ++jj){
            sources[jj] = (char*)calloc(src_[ii->second[jj]].path().length()+1, sizeof(char));
            strcpy(sources[jj],src_[ii->second[jj]].path().c_str());
        }
        hid_t ds = createStringDataset(container, fieldName, (hsize_t)ii->second.size(), (hsize_t)ii->second.size());
        hid_t memtype = H5Tcopy(H5T_C_S1);
        status = H5Tset_size(memtype, H5T_VARIABLE);
        assert(status >= 0);
        status = H5Dwrite(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, sources);
        assert(status >= 0);
        for (unsigned int jj = 0; jj < ii->second.size(); ++jj){
            free(sources[jj]);
        }
        free(sources);
        status = H5DSset_scale(ds, "source");
        status = H5DSattach_scale(classFieldToUniform_[ii->first], ds, 0);
        status = H5DSset_label(classFieldToUniform_[ii->first], 0, "source");
        status = H5Dclose(ds);
        status = H5Tclose(memtype);
    }
}

void NSDFWriter2::closeEventData()
{
    for (unsigned int ii = 0; ii < eventDatasets_.size(); ++ii){
        if (eventDatasets_[ii] >= 0){
            H5Dclose(eventDatasets_[ii]);
        }
    }
    events_.clear();
    eventInputs_.clear();
    eventDatasets_.clear();
    eventSrc_.clear();
    eventSrcFields_.clear();
}

/**
   Populates the vector of event data buffers (vectors), vector of
   event source objects, vector of event source fields and the vector
   of event datasets by querying the messages on InputVariables.
 */
void NSDFWriter2::openEventData(const Eref &eref)
{
    if (filehandle_ <= 0){
        return;
    }
    for (unsigned int ii = 0; ii < eventInputs_.size(); ++ii){
        stringstream path;
        path << eref.objId().path() << "/" << "eventInput[" << ii << "]";
        ObjId inputObj = ObjId(path.str());
        Element * el = inputObj.element();
        const DestFinfo * dest = static_cast<const DestFinfo*>(el->cinfo()->findFinfo("input"));
        vector < ObjId > src;
        vector < string > srcFields;
        el->getMsgSourceAndSender(dest->getFid(), src, srcFields);
        if (src.size() > 1){
            cerr << "NSDFWriter2::openEventData - only one source can be connected to an eventInput" <<endl;
        } else if (src.size() == 1){
            eventSrcFields_.push_back(srcFields[0]);
            eventSrc_.push_back(src[0].path());
            events_.resize(eventSrc_.size());
            stringstream path;
            path << src[0].path() << "." << srcFields[0];
            hid_t dataSet = getEventDataset(src[0].path(), srcFields[0]);
            eventDatasets_.push_back(dataSet);
        } else {
            cerr <<"NSDFWriter2::openEventData - cannot handle multiple connections at single input." <<endl;
        }
    }
}

void NSDFWriter2::createEventMap()
{
    herr_t status;
    hid_t eventMapContainer = require_group(filehandle_, MAPEVENTSRC);
    // Open the container for the event maps
    // Create the Datasets themselves (one for each field - each row
    // for one object).
    for (map< string, vector < string > >::iterator ii = classFieldToEventSrc_.begin();
         ii != classFieldToEventSrc_.end();
         ++ii){
        vector < string > pathTokens;
        moose::tokenize(ii->first, "/", pathTokens);
        string className = pathTokens[0];
        string fieldName = pathTokens[1];
        hid_t classGroup = require_group(eventMapContainer, className);
        hid_t strtype = H5Tcopy(H5T_C_S1);
        status = H5Tset_size(strtype, H5T_VARIABLE);
        // create file space
        hid_t ftype = H5Tcreate(H5T_COMPOUND, sizeof(hvl_t) +sizeof(hobj_ref_t));
        status = H5Tinsert(ftype, "source", 0, strtype);
        status = H5Tinsert(ftype, "data", sizeof(hvl_t), H5T_STD_REF_OBJ);
        hsize_t dims[1] = {ii->second.size()};
        hid_t space = H5Screate_simple(1, dims, NULL);
        // The dataset for mapping is named after the field
        hid_t ds = H5Dcreate2(classGroup, fieldName.c_str(), ftype, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Sclose(space);
        map_type * buf = (map_type*)calloc(ii->second.size(), sizeof(map_type));
        // Populate the buffer entries with source uid and data
        // reference
        for (unsigned int jj = 0; jj < ii->second.size(); ++jj){
            buf->source = ii->second[jj].c_str();
            char * dsname = (char*)calloc(256, sizeof(char));
            hsize_t size = H5Iget_name(classFieldToEvent_[ii->first][jj], dsname, 255);
            if (size > 255){
                free(dsname);
                dsname = (char*)calloc(size, sizeof(char));
                size = H5Iget_name(classFieldToEvent_[ii->first][jj], dsname, 255);
            }
            status = H5Rcreate(&(buf->data), filehandle_, dsname, H5R_OBJECT, -1);
            free(dsname);
            assert(status >= 0);
        }
        // create memory space
        hid_t memtype = H5Tcreate(H5T_COMPOUND, sizeof(map_type));
        status = H5Tinsert(memtype, "source",
                           HOFFSET(map_type, source), strtype);
        status = H5Tinsert(memtype, "data",
                           HOFFSET(map_type, data), H5T_STD_REF_OBJ);
        status = H5Dwrite(ds, memtype,  H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        free(buf);
        status = H5Tclose(strtype);
        status = H5Tclose(ftype);
        status = H5Tclose(memtype);
        status = H5Dclose(ds);
    }
}

/**
   Create or retrieve a dataset for an event input.  The dataset path
   will be /data/event/{class}/{srcFinfo}/{id}_{dataIndex}_{fieldIndex}.

   path : {source_object_id}.{source_field_name}

   TODO: check the returned hid_t and show appropriate error messages.
*/
hid_t NSDFWriter2::getEventDataset(string srcPath, string srcField)
{
    string eventSrcPath = srcPath + string("/") + srcField;
    map< string, hid_t >::iterator it = eventSrcDataset_.find(eventSrcPath);
    if (it != eventSrcDataset_.end()){
        return it->second;
    }
    ObjId source(srcPath);
    herr_t status;
    htri_t exists = -1;
    string className = Field<string>::get(source, "className");
    string path = EVENTPATH + string("/") + className + string("/") + srcField;
    hid_t container = require_group(filehandle_, path);
    stringstream dsetname;
    dsetname << source.id.value() <<"_" << source.dataIndex << "_" << source.fieldIndex;
    hid_t dataset = createDoubleDataset(container, dsetname.str().c_str());
    classFieldToEvent_[className + "/" + srcField].push_back(dataset);
    classFieldToEventSrc_[className + "/" + srcField].push_back(srcPath);
    status = writeScalarAttr<string>(dataset, "source", srcPath);
    assert(status >= 0);
    status = writeScalarAttr<string>(dataset, "field", srcField);
    assert(status >= 0);
    eventSrcDataset_[eventSrcPath] = dataset;
    return dataset;
}

void NSDFWriter2::flush()
{
    // We need to update the tend on each write since we do not know
    // when the simulation is getting over and when it is just paused.
    writeScalarAttr<string>(filehandle_, "tend", iso_time(NULL));

    // append all uniform data
	for ( vector< Block >::iterator bit = blocks_.begin(); (steps_ > 0) && (bit != blocks_.end()); bit++ ) {
		assert( steps_ == bit->data[0].size() );
        double* buffer = (double*)calloc(bit->data.size() * steps_, sizeof(double));
        for (unsigned int ii = 0; ii < bit->data.size(); ++ii){
            for (unsigned int jj = 0; jj < steps_; ++jj){
                buffer[ii * steps_ + jj] = bit->data[ii][jj];
            }
            bit->data[ii].clear();
        }
        hid_t filespace = H5Dget_space(bit->dataset);
        if (filespace < 0){
			cout << "Error: NSDFWriter2::flush(): Failed to open filespace\n";
            break;
        }
        hsize_t dims[2];
        hsize_t maxdims[2];
        // retrieve current datset dimensions
        herr_t status = H5Sget_simple_extent_dims(filespace, dims, maxdims);
        hsize_t newdims[] = {dims[0], dims[1] + steps_}; // new column count
        status = H5Dset_extent(bit->dataset, newdims); // extend dataset to new column count
        H5Sclose(filespace);
        filespace = H5Dget_space(bit->dataset); // get the updated filespace
        hsize_t start[2] = {0, dims[1]};
        dims[1] = steps_; // change dims for memspace & hyperslab
        hid_t memspace = H5Screate_simple(2, dims, NULL);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, dims, NULL);
        status = H5Dwrite(bit->dataset, H5T_NATIVE_DOUBLE,  memspace, filespace, H5P_DEFAULT, buffer);
        H5Sclose(memspace);
        H5Sclose(filespace);
        free(buffer);
    }
	steps_ = 0;

    // append all event data
    for (unsigned int ii = 0; ii < eventSrc_.size(); ++ii){
        appendToDataset(getEventDataset(eventSrc_[ii], eventSrcFields_[ii]),
                        events_[ii]);
        events_[ii].clear();
    }
    // flush HDF5 nodes.
    HDF5DataWriter::flush();
}

void NSDFWriter2::reinit(const Eref& eref, const ProcPtr proc)
{
    // write environment
    // write model
    // write map
    if (filehandle_ >0){
        close();
    }
    // TODO: what to do when reinit is called? Close the existing file
    // and open a new one in append mode? Or keep adding to the
    // current file?
    if (filename_.empty()){
        filename_ = "moose_data.nsdf.h5";
    }
    openFile();
    writeScalarAttr<string>(filehandle_, "created", iso_time(0));
    writeScalarAttr<string>(filehandle_, "tstart", iso_time(0));
    writeScalarAttr<string>(filehandle_, "nsdf_version", "1.0");
    openUniformData(eref);
	for (vector< Block >::iterator bi = blocks_.begin(); bi != blocks_.end(); ++bi) {
        writeScalarAttr< double >(bi->dataset, "tstart", 0.0);
        // dt is same for all requested data - that of the NSDFWriter2
        writeScalarAttr< double >(bi->dataset, "dt", proc->dt);
	}
    openEventData(eref);
    writeModelFiles();
    writeModelTree();
    createUniformMap();
    createStaticMap();
    createEventMap();
	writeStaticCoords();
    steps_ = 0;
}

void NSDFWriter2::process(const Eref& eref, ProcPtr proc)
{
    if (filehandle_ < 0){
        return;
    }
    vector < double > uniformData;
    const Finfo* tmp = eref.element()->cinfo()->findFinfo("requestOut");
    const SrcFinfo1< vector < double > *>* requestOut = static_cast<const SrcFinfo1< vector < double > * > * >(tmp);
    requestOut->send(eref, &uniformData);
	assert( uniformData.size() == mapMsgIdx_.size() );
	// Note that uniformData is ordered by msg tgt order. We want to store
	// data in block_->objVec order.
	unsigned int ii = 0;
	for (unsigned int blockIdx = 0; blockIdx < blocks_.size(); ++blockIdx) {
		vector< vector< double > >&  bjd = blocks_[blockIdx].data;
		for ( auto jj = bjd.begin(); jj != bjd.end(); ++jj ) {
			jj->push_back( uniformData[ mapMsgIdx_[ii] ] );
			ii++;
		}
	}
    ++steps_;
    if (steps_ < flushLimit_){
        return;
    }
    NSDFWriter2::flush();
 }

NSDFWriter2& NSDFWriter2::operator=( const NSDFWriter2& other)
{
	eventInputs_ = other.eventInputs_;
	for ( vector< InputVariable >::iterator i = eventInputs_.begin(); i != eventInputs_.end(); ++i ) {
		i->setOwner( this );
	}
	for (unsigned int ii = 0; ii < getNumEventInputs(); ++ii){
		events_[ii].clear();
	}
	return *this;
}

void NSDFWriter2::setNumEventInputs(unsigned int num)
{
    unsigned int prevSize = eventInputs_.size();
    eventInputs_.resize(num);
    for (unsigned int ii = prevSize; ii < num; ++ii){
        eventInputs_[ii].setOwner(this);
    }
}

unsigned int NSDFWriter2::getNumEventInputs() const
{
    return eventInputs_.size();
}

void NSDFWriter2::setEnvironment(string key, string value)
{
    env_[key] = value;
}


void NSDFWriter2::setInput(unsigned int index, double value)
{
    events_[index].push_back(value);
}

InputVariable* NSDFWriter2::getEventInput(unsigned int index)
{
    static InputVariable dummy;
    if (index < eventInputs_.size()){
        return &eventInputs_[index];
    }
    cout << "Warning: NSDFWriter2::getEventInput: index: " << index <<
		" is out of range: " << eventInputs_.size() << endl;
   return &dummy;
}

void NSDFWriter2::setModelRoot(string value)
{
    modelRoot_ = value;
}

string NSDFWriter2::getModelRoot() const
{
    return modelRoot_;
}

void NSDFWriter2::setModelFiles(string value)
{
	modelFileNames_.clear();	
    moose::tokenize( value, ", ", modelFileNames_);
}

string NSDFWriter2::getModelFiles() const
{
	string ret = "";
	string spacer = "";
	for( auto s = modelFileNames_.begin(); s!= modelFileNames_.end(); ++s) {
		ret += spacer + *s;
		spacer = ",";
	}
    return ret;
}

bool parseBlockString( const string& val, Block& block )
{
	string s = val;
	auto ff = s.find_last_of(".");
	if (ff == string::npos )
		return false;
	block.hasMsg = false;
	block.hasContainer = false;
	block.field = s.substr( ff + 1 );
	string temp = block.field;
	temp[0] = toupper( temp[0] );
	block.getField = "get" + temp;
	s = s.substr( 0, ff );
	vector< string > svec;
    moose::tokenize(s, "/", svec);
	string path = "";
	unsigned int containerIdx = 0;
	block.containerPath = "";
	block.nsdfContainerPath = "";
	string pct = "";
	for ( unsigned int ii = 0; ii < svec.size(); ii++ ) {
		path += "/" + svec[ii];
		pct = "%";
		Id id( path );
		if ( id != Id() ) {
			if ( id.element()->cinfo()->isA( "Neuron" ) ||
				id.element()->cinfo()->isA( "ChemCompt" ) ) 
			{
				containerIdx = ii;
				block.containerPath = path;
				block.nsdfContainerPath += pct + svec[ii];
			}
		}
	}
	if( block.containerPath == "" )
		return false;
	block.relPath = "";
	block.nsdfRelPath = "";
	string slash = "";
	pct = "";
	for ( auto jj = containerIdx+1; jj < svec.size(); ++jj) {
		block.relPath += slash + svec[jj];
		slash = "/";
		block.nsdfRelPath += pct + svec[jj];
		pct = "%";
	}
	if( block.relPath == "" )
		return false;
	string objWildcardPath = block.containerPath + '/' + block.relPath;
	block.objVec.clear();
	simpleWildcardFind( objWildcardPath, block.objVec );
	if ( block.objVec.size() == 0 ) {
		cout << "Error: NSDFWriter2:parseBlockString: No objects found on path '" << objWildcardPath << "'\n";
		return false;
	}
	block.className = block.objVec[0].element()->cinfo()->name();
	for ( auto obj : block.objVec ) {
		// Nasty workaround for different ways of handling CaConcs in Hsolve
		if (obj.element()->cinfo()->name().find("CaConc") != string::npos &&
			block.className.find( "CaConc" ) != string::npos )
			continue;
		if (obj.element()->cinfo()->name() != block.className ) {
			cout << "Error: NSDFWriter2:parseBlockString: different classes found on path '" << objWildcardPath << "': '" << block.className << "' vs. '" << obj.element()->cinfo()->name() << "'\n";
			return false;
		}
	}

	block.data.resize( block.objVec.size() );
	for( auto i = block.data.begin(); i != block.data.end(); i++ )
		i->clear();
	return true;
}

void NSDFWriter2::setBlocks(vector< string > value)
{
	if ( value.size() == 0 )
		blocks_.clear();
	blocks_.resize( value.size() );
	for ( unsigned int i = 0; i < value.size(); ++i ) {
		if ( !parseBlockString( value[i], blocks_[i] ) ) {
			cout << "Error: NSDFWriter2::setBlocks: block[" << i << "] = '" 
					<< value[i] << "' failed\n";
			return;
		}
	}
    blockStrVec_ = value;
}

vector< string > NSDFWriter2::getBlocks() const
{
    return blockStrVec_;
}

////////////////////////////////////////////////////////////////////////

ObjId findParentElecCompt( ObjId obj )
{
	for (ObjId pa = Field< ObjId >::get( obj, "parent" ); pa != ObjId(); 
		pa = Field< ObjId >::get( pa, "parent" ) ) {
		if ( pa.element()->cinfo()->isA( "CompartmentBase" ) )
			return pa;
	}
	return ObjId();
}

void NSDFWriter2::writeStaticCoords()
{
    hid_t staticObjContainer = require_group(filehandle_, STATICPATH );
	for( auto bit = blocks_.begin(); bit != blocks_.end(); bit++ ) {
		string coordContainer = bit->nsdfContainerPath + "/" + bit->nsdfRelPath;
		string fieldName = "coords"; // pathTokens[1] is not relevant.
        hid_t container = require_group(staticObjContainer, coordContainer);
        double * buffer = 
			(double*)calloc(bit->data.size() * 7, sizeof(double));
		if ( bit->className.find( "Pool" ) != string::npos || 
			 bit->className.find( "Compartment" ) != string::npos ) {
        	for (unsigned int jj = 0; jj < bit->data.size(); ++jj) {
				ObjId obj = bit->objVec[jj];
            	vector< double > coords = Field< vector< double > >::get( obj, fieldName );
				if ( coords.size() == 11 ) { // For SpineMesh
					for ( unsigned int kk = 0; kk < 6; ++kk) {
						buffer[jj * 7 + kk] = coords[kk];
					}
					buffer[jj * 7 + 6] = coords[9]; // head Dia 
				} else if ( coords.size() == 4 ) { // for EndoMesh
					for ( unsigned int kk = 0; kk < 3; ++kk) {
						buffer[jj * 7 + kk] = coords[kk];
						buffer[jj * 7 + kk+3] = coords[kk];
					}
					buffer[jj * 7 + 6] = coords[3];
				} else if ( coords.size() >= 7 ) { // For NeuroMesh
					for ( unsigned int kk = 0; kk < 7; ++kk) {
						buffer[jj * 7 + kk] = coords[kk];
					}
				}
			}
		} else { // Check for things like Ca or chans in an elec compt
        	for (unsigned int jj = 0; jj < bit->objVec.size(); ++jj) {
				ObjId pa = findParentElecCompt( bit->objVec[jj] );
				vector< double > coords( 7, 0.0 ); 
				if (pa != ObjId()) {
            		coords = Field< vector< double > >::get(pa, fieldName);
				}
				for ( unsigned int kk = 0; kk < 7; ++kk) {
					buffer[jj * 7 + kk] = coords[kk];
				}
			}
		}
        hsize_t dims[2];
		dims[0] = bit->data.size();
		dims[1] = 7;
        hid_t memspace = H5Screate_simple(2, dims, NULL);
        hid_t dataspace = H5Screate_simple(2, dims, NULL);
    	hid_t dataset = H5Dcreate2(container, fieldName.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t filespace = H5Dget_space(dataset);
        herr_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE,  memspace, filespace, H5P_DEFAULT, buffer);
		if ( status < 0 ) {
			cout << "Error: Failed to write coords as static entry\n";
		}
	}
}

void NSDFWriter2::writeModelFiles()
{
	// These can be large, exceed 64K limit of attributes. So write as 
	// datasets, not attributes.
	for ( const string& fName : modelFileNames_ ) {
    	// string fPath = MODELFILEPATH + string("/") + fName;
    	string fPath = MODELFILEPATH;
		std::ifstream f( fName );
		auto ss = ostringstream{};
		if ( f.is_open() ) {
			ss << f.rdbuf();
			string fstr = ss.str();
			char* filebuf = (char*)calloc( ss.str().length()+1, sizeof(char)  );
			char** sources = (char**) calloc( 1, sizeof(char* ) );
			sources[0] = filebuf;
			strcpy( filebuf, fstr.c_str() );
    		hid_t fGroup = require_group(filehandle_, fPath);
			hid_t ds = createStringDataset(fGroup, fName, (hsize_t)1, (hsize_t)1 );
			hid_t memtype = H5Tcopy(H5T_C_S1);
			int status = H5Tset_size(memtype, H5T_VARIABLE );
			assert(status >= 0);
			// status = H5Tclose(memtype);
			status = H5Dwrite(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, sources );
			assert(status >= 0);
			free( filebuf );
			free( sources );
		} else {
			cout << "Warning: NSDFWriter2::writeModelFiles Could not open file '" << fName << "'/n";
		}
	}
}

void NSDFWriter2::writeModelTree()
{
	if (modelRoot_ == "")
		return;
    vector< string > tokens;
    ObjId mRoot(modelRoot_);
    string rootPath = MODELTREEPATH + string("/") + mRoot.element()->getName();
    hid_t rootGroup = require_group(filehandle_, rootPath);
    hid_t tmp;
    htri_t exists;
    herr_t status;
    deque<Id> nodeQueue;
    deque<hid_t> h5nodeQueue;
    nodeQueue.push_back(mRoot);
    h5nodeQueue.push_back(rootGroup);
    // TODO: need to clarify what happens with array elements. We can
    // have one node per vec and set a count field for the number of
    // elements
    while (nodeQueue.size() > 0){
        ObjId node = nodeQueue.front();
        nodeQueue.pop_front();
        hid_t prev = h5nodeQueue.front();;
        h5nodeQueue.pop_front();
        vector < Id > children;
        Neutral::children(node.eref(), children);
        for ( unsigned int ii = 0; ii < children.size(); ++ii){
            string name = children[ii].element()->getName();
            // skip the system elements
            if (children[ii].path() == "/Msgs"
                || children[ii].path() == "/clock"
                || children[ii].path() == "/classes"
                || children[ii].path() == "/postmaster"){
                continue;
            }
            exists = H5Lexists(prev, name.c_str(), H5P_DEFAULT);
            if (exists > 0){
                tmp = H5Gopen2(prev, name.c_str(), H5P_DEFAULT);
            } else {
                tmp = H5Gcreate2(prev, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            }
            writeScalarAttr< string >(tmp, "uid", children[ii].path());
            nodeQueue.push_back(children[ii]);
            h5nodeQueue.push_back(tmp);
        }
        status = H5Gclose(prev);
    }
}
#endif // USE_HDF5

//
// NSDFWriter2.cpp ends here
