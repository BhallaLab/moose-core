// NSDFWriter.cpp --- 
// 
// Filename: NSDFWriter.cpp
// Description: 
// Author: subha
// Maintainer: 
// Created: Thu Jun 18 23:16:11 2015 (-0400)
// Version: 
// Last-Updated: 
//           By: 
//     Update #: 0
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
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

#include "header.h"
#include "../utility/utility.h"

#include "HDF5WriterBase.h"
#include "HDF5DataWriter.h"

#include "NSDFWriter.h"


const char* const EVENTPATH ="/data/event";
const char* const UNIFORMPATH ="/data/uniform";

const Cinfo * NSDFWriter::initCinfo()
{    
    static FieldElementFinfo< NSDFWriter, InputVariable > eventInputFinfo(
        "eventInput",
        "Sets up field elements for event inputs",
        InputVariable::initCinfo(),
        &NSDFWriter::getEventInput,
        &NSDFWriter::setNumEventInputs,
        &NSDFWriter::getNumEventInputs);
    
    static DestFinfo process(
        "process",
        "Handle process calls. Collects data in buffer and if number of steps"
        " since last write exceeds flushLimit, writes to file.",
        new ProcOpFunc<NSDFWriter>( &NSDFWriter::process));

    static  DestFinfo reinit(
        "reinit",
        "Reinitialize the object. If the current file handle is valid, it tries"
        " to close that and open the file specified in current filename field.",
        new ProcOpFunc<NSDFWriter>( &NSDFWriter::reinit ));

    static Finfo * processShared[] = {
        &process, &reinit
    };
    
    static SharedFinfo proc(
        "proc",
        "Shared message to receive process and reinit",
        processShared, sizeof( processShared ) / sizeof( Finfo* ));

    static Finfo * finfos[] = {
        &eventInputFinfo,
        &proc,
    };

    static string doc[] = {
        "Name", "NSDFWriter",
        "Author", "Subhasis Ray",
        "Description", "NSDF file writer for saving data."
    };

    static Dinfo< NSDFWriter > dinfo;
    static Cinfo cinfo(
        "NSDFWriter",
        HDF5DataWriter::initCinfo(),
        finfos,
        sizeof(finfos)/sizeof(Finfo*),
        &dinfo,
	doc, sizeof( doc ) / sizeof( string ));

    return &cinfo;
}

static const Cinfo * nsdfWriterCinfo = NSDFWriter::initCinfo();

NSDFWriter::NSDFWriter(): eventGroup_(-1), uniformGroup_(-1), dataGroup_(-1), modelGroup_(-1), mapGroup_(-1)
{
    ;
}

NSDFWriter::~NSDFWriter()
{
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
}

void NSDFWriter::closeUniformData()
{
    for (map < string, hid_t>::iterator ii = classFieldToUniform_.begin();
         ii != classFieldToUniform_.end();
         ++ii){
        if (ii->second >= 0){
            H5Dclose(ii->second);
        }
    }
    vars_.clear();
    data_.clear();
    src_.clear();
    func_.clear();
    datasets_.clear();

}

void NSDFWriter::sortOutUniformSources(const Eref& eref)
{
    vars_.clear();
    classFieldToSrcIndex_.clear();
    objectField_.clear();
    objectFieldToIndex_.clear();
    const SrcFinfo * requestOut = (SrcFinfo*)eref.element()->cinfo()->findFinfo("requestOut");
    unsigned int numTgt = eref.element()->getMsgTargetAndFunctions(eref.dataIndex(),
                                                                requestOut,
                                                                src_,
                                                                func_);
    assert(numTgt ==  src_.size());
    /////////////////////////////////////////////////////////////
    // Go through all the sources and determine the index of the
    // source message in the dataset
    /////////////////////////////////////////////////////////////
    
    for (unsigned int ii = 0; ii < func_.size(); ++ii){
        string varname = func_[ii];
        size_t found = varname.find("get");
        if (found == 0){
            varname = varname.substr(3);
            if (varname.length() == 0){
                varname = func_[ii];
            } else {
                varname[0] = tolower(varname[0]);
            }
        }
        assert(varname.length() > 0);
        string className = Field<string>::get(src_[ii], "className");
        string datasetPath = className + "/"+ varname;
        classFieldToSrcIndex_[datasetPath].push_back(ii);
        vars_.push_back(varname);
    }
    data_.resize(numTgt);
}

/**
   Handle the datasets for the requested fields (connected to
   requestOut). This is is similar to what HDF5DataWriter does. 
 */
void NSDFWriter::openUniformData(const Eref &eref)
{
    sortOutUniformSources(eref);
    htri_t exists;
    herr_t status;
    if (dataGroup_ < 0){
        exists = H5Lexists(filehandle_, "data", H5P_DEFAULT);
        if (exists > 0){
            dataGroup_ = H5Gopen2(filehandle_, "data", H5P_DEFAULT);
        } else if (exists == 0){
            dataGroup_ = H5Gcreate2(filehandle_, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }
    }
    if (uniformGroup_ < 0){
        exists = H5Lexists(dataGroup_, "uniform", H5P_DEFAULT);
        if (exists > 0){
            uniformGroup_ = H5Gopen2(dataGroup_, "uniform", H5P_DEFAULT);
        } else if (exists == 0){
            uniformGroup_ = H5Gcreate2(dataGroup_, "uniform", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }
    }

    // create the datasets and map them to className/field
    for (map< string, vector< unsigned int > >::iterator it = classFieldToSrcIndex_.begin();
         it != classFieldToSrcIndex_.end();
         ++it){
        vector< string > tokens;
        hid_t tmp = -1;
        tokenize(it->first, "/", tokens);
        hid_t prev = uniformGroup_;
        for( unsigned int ii = 0; ii < tokens.size() - 1; ++ii){
            exists = H5Lexists(prev, tokens[ii].c_str(), H5P_DEFAULT);
            if (exists > 0){
                tmp = H5Gopen2(prev, tokens[ii].c_str(), H5P_DEFAULT);
            } else {
                tmp = H5Gcreate2(prev, tokens[ii].c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            }
            if (prev != uniformGroup_){
                status = H5Gclose(prev);
            }
            prev = tmp;
        }
        exists = H5Lexists(prev, tokens.back().c_str(), H5P_DEFAULT);
        if (exists == 0){
            hid_t dataset = createDataset2D(prev, tokens.back().c_str(), it->second.size());
            classFieldToUniform_[it->first] = dataset;
            // TODO: Add the list of sources as an attribute
            writeScalarAttr<string>(dataset, "field", tokens.back());
        }
        H5Gclose(prev);
    }
    
    for (unsigned int ii = 0; ii < src_.size(); ++ii){
        string classField = Field<string>::get(src_[ii], "className") + "/" + vars_[ii];
        map< string, hid_t>::iterator dit = classFieldToUniform_.find(classField);
        if (dit == classFieldToUniform_.end()){
            cerr << "Error: NSDFWriter::openUniformData - could not find dataset " << classField << endl;
            continue;
        }
    }
}

void NSDFWriter::closeEventData()
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
void NSDFWriter::openEventData(const Eref &eref)
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
            cerr << "NSDFWriter::openEventData - only one source can be connected to an eventInput" <<endl;
        } else if (src.size() == 1){
            eventSrcFields_.push_back(srcFields[0]);
            eventSrc_.push_back(src[0].path());
            events_.resize(eventSrc_.size());
            stringstream path;
            path << src[0].path() << "." << srcFields[0];
            hid_t dataSet = getEventDataset(src[0].path(), srcFields[0]);
            eventDatasets_.push_back(dataSet);            
        } else {
            cerr <<"NSDFWriter::openEventData - cannot handle multiple connections at single input." <<endl;
        }
    }
}

herr_t NSDFWriter::writeEnv()
{
    // TODO complete this
    return 0;
}

/**
   Create or retrieve a dataset for an uniform data source.
   The dataset path will be
   /data/uniform/{class}/{field}
 */

// hid_t NSDFWriter::getUniformDataset(unsigned int srcIndex)
// {
//     string path = srcPath + "." +srcField;
//     map < string, pair< hid_t, unsigned int> >::iterator it = uniformSrcDatasetIndex_.find(path);
//     if (it != uniformSrcDatasetIndex_.end()){
//         return it->second.first;
//     }
//     ObjId src = ObjId(srcPath);
//     string className = Field<string>::get(src, "className");
//     vector< string > pathTokens;
//     tokenize(UNIFORMPATH, "/", pathTokens);
//     pathTokens.push_back(className);
//     pathTokens.push_back(srcField);
//     hid_t container = -1;
//     hid_t prev = filehandle_;
//     for (unsigned int ii = 0; ii < pathTokens.size(); ++ii){
//         herr_t exists = H5Lexists(prev, pathTokens[ii].c_str(),
//                                   H5P_DEFAULT);
//         if (exists > 0){
//             // try to open existing group
//             container = H5Gopen2(prev, pathTokens[ii].c_str(), H5P_DEFAULT);
//         } else if (exists == 0) {
//             // If that fails, try to create a group
//             container = H5Gcreate2(prev, pathTokens[ii].c_str(),
//                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//         } 
//         if ((exists < 0) || (container < 0)){
//             // Failed to open/create a group, print the
//             // offending path (for debugging; the error is
//             // perhaps at the level of hdf5 or file system).
//             cerr << "Error: failed to open/create group: ";
//             for (unsigned int jj = 0; jj <= ii; ++jj){
//                 cerr << "/" << pathTokens[jj];
//             }
//             cerr << endl;
//             prev = -1;            
//         }
//         if (prev >= 0  && prev != filehandle_){
//             // Successfully opened/created new group, close the old group
//             herr_t status = H5Gclose(prev);
//             assert( status >= 0 );
//         }
//         prev = container;
//     }    
//     stringstream dsetname;
//     dsetname << src.id.value() <<"_" << src.dataIndex << "_" << src.fieldIndex;
//     hid_t dataset = createDataset(container, dsetname.str().c_str());
//     hid_t attr = writeScalarAttr<string>(dataset, "source", src.path());
//     H5Aclose(attr);
//     attr = writeScalarAttr<string>(dataset, "field", srcField);
//     H5Aclose(attr);
//     eventSrcDataset_[path] = dataset;
//     return dataset;    
    
// }

/**
   Create or retrieve a dataset for an event input.  The dataset path
   will be /data/event/{class}/{srcFinfo}/{id}_{dataIndex}_{fieldIndex}.

   path : {source_object_id}.{source_field_name}

   TODO: check the returned hid_t and show appropriate error messages.
 */
hid_t NSDFWriter::getEventDataset(string srcPath, string srcField)
{
    string path = srcPath + "." + srcField;
            
    map< string, hid_t >::iterator it = eventSrcDataset_.find(path);
    if (it != eventSrcDataset_.end()){
        return it->second;
    }
    ObjId source(srcPath);
    herr_t status;
    htri_t exists = -1;
    string className = Field<string>::get(source, "className");
    vector<string> pathTokens;
    tokenize(EVENTPATH, "/", pathTokens);
    pathTokens.push_back(className);
    pathTokens.push_back(srcField);
    hid_t container = -1;
    hid_t prev = filehandle_;
    for (unsigned int ii = 0; ii < pathTokens.size(); ++ii){
        exists = H5Lexists(prev, pathTokens[ii].c_str(),
                                  H5P_DEFAULT);
        if (exists > 0){
            // try to open existing group
            container = H5Gopen2(prev, pathTokens[ii].c_str(), H5P_DEFAULT);
        } else if (exists == 0) {
            // If that fails, try to create a group
            container = H5Gcreate2(prev, pathTokens[ii].c_str(),
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        } 
        if ((exists < 0) || (container < 0)){
            // Failed to open/create a group, print the
            // offending path (for debugging; the error is
            // perhaps at the level of hdf5 or file system).
            cerr << "Error: failed to open/create group: ";
            for (unsigned int jj = 0; jj <= ii; ++jj){
                cerr << "/" << pathTokens[jj];
            }
            cerr << endl;
            prev = -1;            
        }
        if (prev >= 0  && prev != filehandle_){
            // Successfully opened/created new group, close the old group
            status = H5Gclose(prev);
            assert( status >= 0 );
        }
        prev = container;
    }    
    stringstream dsetname;
    dsetname << source.id.value() <<"_" << source.dataIndex << "_" << source.fieldIndex;
    hid_t dataset = createDataset(container, dsetname.str().c_str());
    status = writeScalarAttr<string>(dataset, "source", source.path());
    assert(status >= 0);
    status = writeScalarAttr<string>(dataset, "field", srcField);
    assert(status >= 0);
    eventSrcDataset_[path] = dataset;
    return dataset;    
}

void NSDFWriter::flush()
{
    // append all uniform data
    for (map< string, hid_t>::iterator it = classFieldToUniform_.begin();
         it != classFieldToUniform_.end(); ++it){
        map< string, vector < unsigned int > >::iterator idxit = classFieldToSrcIndex_.find(it->first);
        if (idxit == classFieldToSrcIndex_.end()){
            cerr << "Error: NSDFWriter::flush - could not find entry for " << it->first <<endl;
            break;
        }
        if (data_.size() == 0 || data_[0].size() == 0){
            break;
        }        
        double buffer[idxit->second.size()][steps_];
        vector< double > values;
        for (unsigned int ii = 0; ii < idxit->second.size(); ++ii){
            for (unsigned int jj = 0; jj < steps_; ++jj){
                buffer[ii][jj] = data_[idxit->second[ii]][jj];
            }
            data_[idxit->second[ii]].clear();
        }
        
        hid_t filespace = H5Dget_space(it->second);
        if (filespace < 0){
            break;
        }
        hsize_t dims[2];
        hsize_t maxdims[2];
        // retrieve current datset dimensions
        herr_t status = H5Sget_simple_extent_dims(filespace, dims, maxdims);        
        hsize_t newdims[] = {dims[0], dims[1] + steps_}; // new column count
        status = H5Dset_extent(it->second, newdims); // extend dataset to new column count
        H5Sclose(filespace);
        filespace = H5Dget_space(it->second); // get the updated filespace 
        hsize_t start[2] = {0, dims[1]};
        dims[1] = steps_; // change dims for memspace & hyperslab
        hid_t memspace = H5Screate_simple(2, dims, NULL);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, dims, NULL);
        status = H5Dwrite(it->second, H5T_NATIVE_DOUBLE,  memspace, filespace, H5P_DEFAULT, buffer);
        H5Sclose(memspace);
        H5Sclose(filespace);
    }
    
    // append all event data
    for (unsigned int ii = 0; ii < eventSrc_.size(); ++ii){
        appendToDataset(getEventDataset(eventSrc_[ii], eventSrcFields_[ii]),
                        events_[ii]);
        events_[ii].clear();
    }
    // flush HDF5 nodes.
    HDF5DataWriter::flush();
}

void NSDFWriter::reinit(const Eref& eref, const ProcPtr proc)
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
    openUniformData(eref);
    openEventData(eref);
    herr_t err = writeEnv();
    steps_ = 0;
}

void NSDFWriter::process(const Eref& eref, ProcPtr proc)
{
    if (filehandle_ < 0){
        return;
    }
    vector < double > uniformData;
    const Finfo* tmp = eref.element()->cinfo()->findFinfo("requestOut");
    const SrcFinfo1< vector < double > *>* requestOut = static_cast<const SrcFinfo1< vector < double > * > * >(tmp);
    requestOut->send(eref, &uniformData);
    for (unsigned int ii = 0; ii < uniformData.size(); ++ii){
        data_[ii].push_back(uniformData[ii]);
    }
    ++steps_;
    if (steps_ < flushLimit_){
        return;
    }
    // // TODO this is place holder. Will convert to 2D datasets to
    // // collect same field from object on same clock
    // for (unsigned int ii = 0; ii < datasets_.size(); ++ii){
    //     herr_t status = appendToDataset(datasets_[ii], data_[ii]);
    //     data_[ii].clear();
    // }
    // for (unsigned int ii = 0; ii < events_.size(); ++ii){
    //     herr_t status = appendToDataset(eventDatasets_[ii], events_[ii]);
    // }
    NSDFWriter::flush();
    steps_ = 0;
}

NSDFWriter& NSDFWriter::operator=( const NSDFWriter& other)
{
	eventInputs_ = other.eventInputs_;
	for ( vector< InputVariable >::iterator 
					i = eventInputs_.begin(); i != eventInputs_.end(); ++i )
			i->setOwner( this );
        for (unsigned int ii = 0; ii < getNumEventInputs(); ++ii){
            events_[ii].clear();
        }
	return *this;
}

void NSDFWriter::setNumEventInputs(unsigned int num)
{
    unsigned int prevSize = eventInputs_.size();
    eventInputs_.resize(num);
    for (unsigned int ii = prevSize; ii < num; ++ii){
        eventInputs_[ii].setOwner(this);
    }
}

unsigned int NSDFWriter::getNumEventInputs() const
{
    return eventInputs_.size();
}

void NSDFWriter::setEnvironment(string key, string value)
{
    env_[key] = value;
}


void NSDFWriter::setInput(unsigned int index, double value)
{
    events_[index].push_back(value);
}

InputVariable* NSDFWriter::getEventInput(unsigned int index)
{
    static InputVariable dummy;
    if (index < eventInputs_.size()){
        return &eventInputs_[index];
    }
    cout << "Warning: NSDFWriter::getEventInput: index: " << index <<
		" is out of range: " << eventInputs_.size() << endl;
   return &dummy;
}

#endif // USE_HDF5

// 
// NSDFWriter.cpp ends here
