// HDF5DataWriter.cpp --- 
// 
// Filename: HDF5DataWriter.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 16:03:59 2012 (+0530)
// Version: 
// Last-Updated: Tue Oct  2 01:58:19 2012 (+0530)
//           By: subha
//     Update #: 681
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
// 2012-02-25 16:04:02 (+0530) Subha - started initial implementation
// 

// Code:

#ifdef USE_HDF5

#include "hdf5.h"

#include "header.h"
#include "../utility/utility.h"

#include "HDF5DataWriter.h"

const hssize_t HDF5DataWriter::CHUNK_SIZE = 1024; // default chunk size

static SrcFinfo1< FuncId > *requestData() {
	static SrcFinfo1< FuncId > requestData(
			"requestData",
			"Sends request for a field to target object"
			);
	return &requestData;
}

static DestFinfo *recvDataBuf() {
    static DestFinfo recvDataBuf(
            "recvData",
            "Handles data sent back following request",
            new EpFunc1< HDF5DataWriter, PrepackedBuffer >( &HDF5DataWriter::recvData ));
    return &recvDataBuf;
}

static SrcFinfo0 * clear(){
    static SrcFinfo0 clear("clear",
                           "Send request to clear a Table vector.");
    return &clear;
}

const Cinfo * HDF5DataWriter::initCinfo()
{
  static DestFinfo process(
      "process",
      "Handle process calls. Write data to file and clear all Table objects"
      " associated with this.",
  		new ProcOpFunc<HDF5DataWriter>( &HDF5DataWriter::process)
	);
  static  DestFinfo reinit(
      "reinit",
      "Reinitialize the object",
  		new ProcOpFunc<HDF5DataWriter>( &HDF5DataWriter::reinit )
    );
  static Finfo * processShared[] = {
    &process, &reinit
  };
  static SharedFinfo proc(
      "proc",
      "Shared message to receive process and reinit",
      processShared, sizeof( processShared ) / sizeof( Finfo* ));
  static Finfo * finfos[] = {
    requestData(),
    clear(),
    recvDataBuf(),        
    &proc,
  };

  static string doc[] = {
    "Name", "HDF5DataWriter",
    "Author", "Subhasis Ray",
    "Description", "HDF5 file writer for saving data tables. It saves the tables added to"
    " it via addObject function into an HDF5 file. At every process call it"
    " writes the contents of the tables to the file and clears the table"
    " vectors. You can explicitly save the data via the flush function."
  };

    static Cinfo cinfo(
        "HDF5DataWriter",
        HDF5WriterBase::initCinfo(),
        finfos,
        sizeof(finfos)/sizeof(Finfo*),
        new Dinfo<HDF5DataWriter>(),
		doc, sizeof( doc ) / sizeof( string )
	   	);
    return &cinfo;
}

static const Cinfo * hdf5dataWriterCinfo = HDF5DataWriter::initCinfo();

HDF5DataWriter::HDF5DataWriter()
{
}

HDF5DataWriter::~HDF5DataWriter()
{
    flush();
    for (map < string, hid_t >::iterator ii = nodemap_.begin(); ii != nodemap_.end(); ++ii){
        if (ii->second >= 0){
            herr_t status = H5Dclose(ii->second);
            if (status < 0){
                cerr << "Warning: closing dataset for " << ii->first << ", returned status = " << status << endl;
            }
        }
    }
}
void HDF5DataWriter::flush()
{
    if (filehandle_ < 0){
        return;
    }
    for (map < string, vector < double > >::iterator ii = datamap_.begin(); ii != datamap_.end(); ++ ii){
        hid_t dataset = nodemap_[ii->first];
        if (dataset < 0){
            dataset = get_dataset(ii->first);
            nodemap_[ii->first] = dataset;
        }
        herr_t status = appendToDataset(dataset, ii->second);
        if (status < 0){
            cerr << "Warning: appending data for object " << ii->first << " returned status " << status << endl;                
        }
        ii->second.clear();        
    }
    H5Fflush(filehandle_, H5F_SCOPE_LOCAL);
}
        
/**
   Write data to datasets in HDF5 file. Clear all data in the table
   objects associated with this object. */
void HDF5DataWriter::process(const Eref & e, ProcPtr p)
{
    if (filehandle_ < 0){
        return;
    }
    requestData()->send(e, p->threadIndexInGroup, recvDataBuf()->getFid());
    for (map<string, vector < double > >:: iterator data_it = datamap_.begin(); data_it != datamap_.end(); ++data_it){        
        string path = data_it->first;
        // if (data_it->second.size() >= flushLimit_){
        map < string, hid_t >::iterator node_it = nodemap_.find(path);
        assert (node_it != nodemap_.end());
        if (node_it->second < 0){
            nodemap_[path] = get_dataset(path);
        }
        herr_t status = appendToDataset(nodemap_[path], data_it->second);
        if (status < 0){
            cerr << "Warning: appending data for object " << data_it->first << " returned status " << status << endl;                
        }
        data_it->second.clear();
    }    
    clear()->send(e, p->threadIndexInGroup);
}

void HDF5DataWriter::reinit(const Eref & e, ProcPtr p)
{
    if (filename_.empty()){
        filename_ = "moose_output.h5";
    }
    if (filehandle_ < 0){
        openFile();
    }
    cout << "Openfile:" << filehandle_ << endl;
}

/**
   Traverse the path of an object in HDF5 file, checking existence of
   groups in the path and creating them if required.  */
hid_t HDF5DataWriter::get_dataset(string path)
{
    if (filehandle_<0){
        return -1;
    }
    herr_t status;
    // Create the groups corresponding to this path We are not
    // taking care of Table object containing Table
    // objects. That's an unusual possibility.
    vector<string> path_tokens;
    tokenize(path, "/", path_tokens);
    hid_t prev_id = filehandle_;
    for ( unsigned int ii = 0; ii < path_tokens.size()-1; ++ii ){
        // First try to open existing group
        hid_t id = H5Gcreate2(prev_id, path_tokens[ii].c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (id < 0){
            // If that fails, try to create a group
            id = H5Gopen2(prev_id, path_tokens[ii].c_str(), H5P_DEFAULT);
            if (id < 0){
                // Failed to craete a group also, print the
                // offending path (for debugging; the error is
                // perhaps at the level of hdf5 or file system).
                cerr << "Error: failed to open/create group: ";
                for (unsigned int jj = 0; jj <= ii; ++jj){
                    cerr << "/" << path_tokens[jj];
                }
                cerr << endl;
                prev_id = -1;
                break;
            } else if (prev_id != filehandle_){
                // Successfully created new group, close the old group
                status = H5Gclose(prev_id);
				assert( status >= 0 );
                prev_id = id;
            }
        } else if (prev_id != filehandle_){
            // Successfully opened new group, close the old group
            status = H5Gclose(prev_id);
			assert( status >= 0 );
        }
        prev_id = id;
    }
    if (prev_id < 0){
        nodemap_[path] = prev_id;
        return prev_id;
    }
    // first try to create, then try to open dataset
    string name = path_tokens[path_tokens.size()-1];
    hid_t dataset_id = create_dataset(prev_id, name);
    if (dataset_id < 0){
        dataset_id = H5Dopen2(prev_id, name.c_str(), H5P_DEFAULT);
    }
    return dataset_id;
}

/**
   Create a new 1D dataset. Make it extensible.
*/
hid_t HDF5DataWriter::create_dataset(hid_t parent_id, string name)
{
    herr_t status;
    hsize_t dims[1] = {0};
    hsize_t maxdims[] = {H5S_UNLIMITED};
    hsize_t chunk_dims[] = {CHUNK_SIZE}; // 1 K
    hid_t chunk_params = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_chunk(chunk_params, 1, chunk_dims);
	assert( status >= 0 );
    hid_t dataspace = H5Screate_simple(1, dims, maxdims);            
    hid_t dataset_id = H5Dcreate2(parent_id, name.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, chunk_params, H5P_DEFAULT);
    return dataset_id;
}

/**
   Append a vector to a specified dataset and return the error status
   of the write operation. */
herr_t HDF5DataWriter::appendToDataset(hid_t dataset_id, const vector< double >& data)
{
    herr_t status;
    if (dataset_id < 0){
        return -1;
    }
    hid_t filespace = H5Dget_space(dataset_id);
    if (filespace < 0){
        return -1;
    }
    if (data.size() == 0){
        return 0;
    }
    hsize_t size = H5Sget_simple_extent_npoints(filespace) + data.size();
    status = H5Dset_extent(dataset_id, &size);
    if (status < 0){
        return status;
    }
    filespace = H5Dget_space(dataset_id);
    hsize_t size_increment = data.size();
    hid_t memspace = H5Screate_simple(1, &size_increment, NULL);
    hsize_t start = size - data.size();
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &start, NULL, &size_increment, NULL);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, &data[0]);
    return status;
}

void HDF5DataWriter::recvData(const Eref&e, const Qinfo* q, PrepackedBuffer pb)
{
    string path = q->src().path();
    if (nodemap_.find(path) == nodemap_.end()){
        // first time call, initialize entries in map
        hid_t dataid =  get_dataset(path);
        if (dataid < 0){
            cerr << "Warning: could not create data set for " << path << endl;
        }
        nodemap_[path] = dataid;
        datamap_[path] = vector<double>();
    }
    // for vectors, the PrepackedBuffer has format:
    // [0] = dataSize (total length)
    // [1] = 0 - this seems always 0
    // [2] = vector size
    // [3 ... dataSize-1] - vector content
    unsigned vec_size = pb.data()[0];
    // This is a hack - based on emprical look at PrepackedBuffer
    // contents for table vectors. I believe when populating
    // prepackedbuffer with vector<double>, the numEntries, data(),
    // size() are confusing.
    const double * start = pb.data() + 1;
    const double * end = start + vec_size;
    // append only the new data. old_size is guaranteed to be 0 on
    // write and the table vecs will also be cleared.
    datamap_[path].insert(datamap_[path].end(), start, end);        
}
        
#endif // USE_HDF5
// 
// HDF5DataWriter.cpp ends here
