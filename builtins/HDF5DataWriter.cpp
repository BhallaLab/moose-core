// HDF5DataWriter.cpp --- 
// 
// Filename: HDF5DataWriter.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 16:03:59 2012 (+0530)
// Version: 
// Last-Updated: Wed Nov 14 18:47:02 2012 (+0530)
//           By: subha
//     Update #: 735
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

static SrcFinfo1< vector < double > * > *requestOut() {
    static SrcFinfo1< vector < double > * > requestOut(
        "requestOut",
        "Sends request for a field to target object"
                                                       );
    return &requestOut;
}

const Cinfo * HDF5DataWriter::initCinfo()
{
    static DestFinfo process(
        "process",
        "Handle process calls. Write data to file and clear all Table objects"
        " associated with this. Hence you want to keep it on a slow clock"
        " 1000 times or more slower than that for the tables.",
        new ProcOpFunc<HDF5DataWriter>( &HDF5DataWriter::process)
                             );
    static  DestFinfo reinit(
        "reinit",
        "Reinitialize the object. If the current file handle is valid, it tries"
        " to close that and open the file specified in current filename field.",
        new ProcOpFunc<HDF5DataWriter>( &HDF5DataWriter::reinit )
                             );
    static Finfo * processShared[] = {
        &process, &reinit
    };
    
    static SharedFinfo proc(
        "proc",
        "Shared message to receive process and reinit",
        processShared, sizeof( processShared ) / sizeof( Finfo* ));

    static ValueFinfo< HDF5DataWriter, unsigned int> flushLimit(
      "flushLimit",
      "Buffer size limit for flushing the data from memory to file. Default"
      " is 4M doubles.",
      &HDF5DataWriter::setFlushLimit,
      &HDF5DataWriter::getFlushLimit);

    static Finfo * finfos[] = {
        requestOut(),
        &flushLimit,
        &proc,
    };

    

    static string doc[] = {
        "Name", "HDF5DataWriter",
        "Author", "Subhasis Ray",
        "Description", "HDF5 file writer for saving data tables. It saves the tables connected"
        " to it via `requestOut` field into an HDF5 file.  The path of the"
        " table is maintained in the HDF5 file, with a HDF5 group for each"
        " element above the table."
        "\n"
        "Thus, if you have a table `/data/VmTable` in MOOSE, then it will be"
        " written as an HDF5 table called `VmTable` inside an HDF5 Group called"
        " `data`."
        "\n"
        "However Table inside Table is considered a pathological case and is"
        " not handled.\n"
        "At every process call it writes the contents of the tables to the file"
        " and clears the table vectors. You can explicitly force writing of the"
        " data via the `flush` function."
    };

    static Dinfo< HDF5DataWriter > dinfo;
    static Cinfo cinfo(
        "HDF5DataWriter",
        HDF5WriterBase::initCinfo(),
        finfos,
        sizeof(finfos)/sizeof(Finfo*),
        &dinfo,
	doc, sizeof( doc ) / sizeof( string ));
    return &cinfo;
}

static const Cinfo * hdf5dataWriterCinfo = HDF5DataWriter::initCinfo();

HDF5DataWriter::HDF5DataWriter(): flushLimit_(4*1024*1024), steps_(0)
{
}

HDF5DataWriter::~HDF5DataWriter()
{
    if (filehandle_ < 0){
        return;
    }
    this->flush();
    for (map < string, hid_t >::iterator ii = nodemap_.begin();
         ii != nodemap_.end(); ++ii){
        if (ii->second >= 0){
            herr_t status = H5Dclose(ii->second);
            if (status < 0){
  	        cerr << "Warning: closing dataset for "
                     << ii->first << ", returned status = "
                     << status << endl;
            }
        }
    }
    filehandle_ = -1;
}

void HDF5DataWriter::flush()
{
    if (filehandle_ < 0){
        cerr << "HDF5DataWriter::flush() - "
                "Filehandle invalid. Cannot write data." << endl;
        return;
    }
    
    for (unsigned int ii = 0; ii < datasets_.size(); ++ii){
        herr_t status = appendToDataset(datasets_[ii], data_[ii]);
        data_[ii].clear();
        if (status < 0){
            cerr << "Warning: appending data for object " << src_[ii]
                 << " returned status " << status << endl;                
        }        
    }
    HDF5WriterBase::flush();
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
    
    vector <double> dataBuf;
        requestOut()->send(e, &dataBuf);
    for (unsigned int ii = 0; ii < dataBuf.size(); ++ii){
        data_[ii].push_back(dataBuf[ii]);
    }
    ++steps_;
    if (steps_ >= flushLimit_){
        steps_ = 0;
        for (unsigned int ii = 0; ii < datasets_.size(); ++ii){
            herr_t status = appendToDataset(datasets_[ii], data_[ii]);
            data_[ii].clear();
            if (status < 0){
                cerr << "Warning: appending data for object " << src_[ii]
                     << " returned status " << status << endl;                
            }
        }
    }
}

void HDF5DataWriter::reinit(const Eref & e, ProcPtr p)
{
    steps_ = 0;
    for (unsigned int ii = 0; ii < data_.size(); ++ii){
        H5Dclose(datasets_[ii]);
    }
    data_.clear();
    src_.clear();
    func_.clear();
    datasets_.clear();
    unsigned int numTgt = e.element()->getMsgTargetAndFunctions(e.dataIndex(),
                                                                requestOut(),
                                                                src_,
                                                                func_);
    assert(numTgt ==  src_.size());
    // TODO: what to do when reinit is called? Close the existing file
    // and open a new one in append mode? Or keep adding to the
    // current file?
    if (filename_.empty()){
        filename_ = "moose_data.h5";
    }
    if (filehandle_ > 0 ){
        close();
    }
    if (numTgt == 0){
        return;
    }
    openFile();
    for (unsigned int ii = 0; ii < src_.size(); ++ii){
        string varname = func_[ii];
        size_t found = varname.find("get");
        if (found == 0){
            varname = varname.substr(3);
            if (varname.length() == 0){
                varname = func_[ii];
            } else {
                // TODO: there is no way we can get back the original
                // field-name case. tolower will get the right name in
                // most cases as field names start with lower case by
                // convention in MOOSE.
                varname[0] = tolower(varname[0]);
            }
        }
        assert(varname.length() > 0);
        string path = src_[ii].path() + "/" + varname;
        hid_t dataset_id = get_dataset(path);
        datasets_.push_back(dataset_id);
    }
    data_.resize(src_.size());
}

/**
   Traverse the path of an object in HDF5 file, checking existence of
   groups in the path and creating them if required.  */
hid_t HDF5DataWriter::get_dataset(string path)
{
    if (filehandle_ < 0){
        return -1;
    }
    herr_t status = H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    // Create the groups corresponding to this path
    vector<string> path_tokens;
    tokenize(path, "/", path_tokens);
    hid_t prev_id = filehandle_;
    hid_t id = -1;
    for ( unsigned int ii = 0; ii < path_tokens.size()-1; ++ii ){
        // check if object exists
        htri_t exists = H5Lexists(prev_id, path_tokens[ii].c_str(),
                                  H5P_DEFAULT);
        if (exists > 0){
            // try to open existing group
            id = H5Gopen2(prev_id, path_tokens[ii].c_str(), H5P_DEFAULT);
        } else if (exists == 0) {
            // If that fails, try to create a group
            id = H5Gcreate2(prev_id, path_tokens[ii].c_str(),
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        } 
        if ((exists < 0) || (id < 0)){
            // Failed to open/create a group, print the
            // offending path (for debugging; the error is
            // perhaps at the level of hdf5 or file system).
            cerr << "Error: failed to open/create group: ";
            for (unsigned int jj = 0; jj <= ii; ++jj){
                cerr << "/" << path_tokens[jj];
            }
            cerr << endl;
            prev_id = -1;            
        }
        if (prev_id >= 0  && prev_id != filehandle_){
            // Successfully opened/created new group, close the old group
            status = H5Gclose(prev_id);
            assert( status >= 0 );
        }
        prev_id = id;
    }
    string name = path_tokens[path_tokens.size()-1];
    htri_t exists = H5Lexists(prev_id, name.c_str(), H5P_DEFAULT);
    hid_t dataset_id = -1;
    if (exists > 0){
        dataset_id = H5Dopen2(prev_id, name.c_str(), H5P_DEFAULT);
    } else if (exists == 0){
        dataset_id = create_dataset(prev_id, name);
    } else {
        cerr << "Error: H5Lexists returned "
             << exists << " for path \""
             << path << "\"" << endl;
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
    hsize_t chunk_dims[] = {chunkSize_};
    hid_t chunk_params = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_chunk(chunk_params, 1, chunk_dims);
    assert( status >= 0 );
    if (compressor_ == "zlib"){
        status = H5Pset_deflate(chunk_params, compression_);
    } else if (compressor_ == "szip"){
        // this needs more study
        unsigned sz_opt_mask = H5_SZIP_NN_OPTION_MASK;
        status = H5Pset_szip(chunk_params, sz_opt_mask,
                             HDF5WriterBase::CHUNK_SIZE);
    }
    hid_t dataspace = H5Screate_simple(1, dims, maxdims);            
    hid_t dataset_id = H5Dcreate2(parent_id, name.c_str(),
                                  H5T_NATIVE_DOUBLE, dataspace,
                                  H5P_DEFAULT, chunk_params, H5P_DEFAULT);
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
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &start, NULL,
                        &size_increment, NULL);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
                      H5P_DEFAULT, &data[0]);
    return status;
}

void HDF5DataWriter::setFlushLimit(unsigned int value)
{
    flushLimit_ = value;
}

unsigned int HDF5DataWriter::getFlushLimit() const
{
    return flushLimit_;
}
        
#endif // USE_HDF5
// 
// HDF5DataWriter.cpp ends here
