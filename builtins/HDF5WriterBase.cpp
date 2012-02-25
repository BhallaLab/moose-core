// HDF5WriterBase.cpp --- 
// 
// Filename: HDF5WriterBase.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 14:42:03 2012 (+0530)
// Version: 
// Last-Updated: Sun Feb 26 01:16:11 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 143
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

// Code:

#ifdef USE_HDF5

#include "hdf5.h"

#include "header.h"

#include "HDF5WriterBase.h"

const Cinfo* HDF5WriterBase::initCinfo()
{

    static Finfo * finfos[] = {
    //////////////////////////////////////////////////////////////
    // Field Definitions
    //////////////////////////////////////////////////////////////
        new ValueFinfo< HDF5WriterBase, string > (
                "filename",
                "Name of the file associated with this HDF5 writer object.",
                &HDF5WriterBase::setFilename,
                &HDF5WriterBase::getFilename),
        new ReadOnlyValueFinfo < HDF5WriterBase, bool > (
                "isOpen",
                "True if this object has an open file handle.",
                &HDF5WriterBase::isOpen),
        new DestFinfo(
                "addObject",
                "Add an object for writing to file.",
                new OpFunc1 < HDF5WriterBase, string > ( &HDF5WriterBase::addObject )),        
    };
    
    static Cinfo hdf5Cinfo(
            "HDF5WriterBase",
            Neutral::initCinfo(),
            finfos,
            sizeof(finfos)/sizeof(Finfo*),
            new Dinfo<HDF5WriterBase>());
    return &hdf5Cinfo;                
}

HDF5WriterBase::HDF5WriterBase():
        filehandle_(-1)
{
}

HDF5WriterBase::~HDF5WriterBase()
{
    herr_t err = H5Fclose(filehandle_);
    if (err < 0){
        cerr << "Error: Error occurred when closing file. Error code: " << err << endl;
    }
}

void HDF5WriterBase::setFilename(string filename)
{
    herr_t status;
    if (filename_ == filename){
        return;
    }
     
    // TODO check if file is open. If not check if it exists. If it
    // exists, open R/W else create a new one. If file is open, close
    // it and open one with the new name.
    if (filehandle_ >= 0){
        cout << "Warning: closing " << filename_ << " and opening " << filename << ". Error code: " << status  << endl;
        status = H5Fclose(filehandle_);
        if (status < 0){
            cerr << "Error: failed to close HDF5 file handle for " << filename_ << ". Error code: " << status << endl;
            return;
        }
    }
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    // Ensure that all open objects are closed before the file is closed    
    H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
    filehandle_ = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, fapl_id);
    if (filehandle_ < 0){
        cout << "Warning: writing to existing file: " << filename << endl;
        filehandle_ = H5Fopen(filename.c_str(), H5F_ACC_RDWR, fapl_id);
    } else {
        openmode_ = H5F_ACC_RDWR;
    }
    if (filehandle_ < 0){
        cerr << "Error: Could not open file for writing: " << filename_ << ". Error code: " << status << endl;
        return;
    } else {
        openmode_ = H5F_ACC_EXCL;
    }
    filename_ = filename;
}

string HDF5WriterBase::getFilename() const
{
    return filename_;
}

bool HDF5WriterBase::isOpen() const
{
    return filehandle_ >= 0;
}

/**
   Add an object to the list of objects to be saved in this file. The
   ObjId corresponding to the specified path is saved as the key to a
   map from ObjId to the corresponding HDF5 node id (hid_t).  */
void HDF5WriterBase::addObject(string path)
{
    ObjId oid = ObjId(path);
    if (oid == ObjId::bad){
        cerr << "Error: no object exists at this path: " << path << endl;
        return;
    }
    if (object_node_map_.find(path) == object_node_map_.end()){
        object_node_map_[path] = -1;
    }
}

#endif // USE_HDF5
// 
// HDF5WriterBase.cpp ends here
