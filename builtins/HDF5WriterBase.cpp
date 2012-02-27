// HDF5WriterBase.cpp --- 
// 
// Filename: HDF5WriterBase.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 14:42:03 2012 (+0530)
// Version: 
// Last-Updated: Tue Feb 28 00:25:44 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 246
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
        new ValueFinfo <HDF5WriterBase, unsigned int > (
                "mode",
                "Depending on mode, if file already exists, if mode=1, data will be"
                " appended to existing file, if mode=2, file will be truncated, if "
                " mode=4, no writing will happen.",
                &HDF5WriterBase::setMode,
                &HDF5WriterBase::getMode),
        new DestFinfo(
                "flush",
                "Write all buffer contents to file and clear the buffers.",
                new OpFunc0 < HDF5WriterBase > ( &HDF5WriterBase::flush )),        
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
        filehandle_(-1),
        filename_("moose_output.h5"),
        openmode_(H5F_ACC_EXCL)
{
}

HDF5WriterBase::~HDF5WriterBase()
{
    // derived classes should flush data in their own destructors
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
    if (filehandle_ > 0){
        status = H5Fclose(filehandle_);
        if (status < 0){
            cerr << "Error: failed to close HDF5 file handle for " << filename_ << ". Error code: " << status << endl;
        }
    }
    filehandle_ = -1;
    filename_ = filename;
    // status = openFile(filename);
}

string HDF5WriterBase::getFilename() const
{
    return filename_;
}

bool HDF5WriterBase::isOpen() const
{
    return filehandle_ >= 0;
}

herr_t HDF5WriterBase::openFile()
{
    herr_t status = 0;
    if (filehandle_ >= 0){
        cout << "Warning: closing already open file and opening " << filename_ <<  endl;
        status = H5Fclose(filehandle_);
        if (status < 0){
            cerr << "Error: failed to close currently open HDF5 file. Error code: " << status << endl;
            return status;
        }
    }
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    // Ensure that all open objects are closed before the file is closed    
    H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
    if (openmode_ == H5F_ACC_EXCL || openmode_ == H5F_ACC_TRUNC){
        cout << "Excl mode? " << (openmode_ == H5F_ACC_EXCL) << endl;
        filehandle_ = H5Fcreate(filename_.c_str(), openmode_, H5P_DEFAULT, fapl_id);
        printf("File id: %d\n", filehandle_);
    } else {
        filehandle_ = H5Fopen(filename_.c_str(), openmode_, fapl_id);
    }
    if (filehandle_ < 0){
        cerr << "Error: Could not open file for writing: " << filename_ << ". Error code: " << status << endl;
        status = -1;
    }
    return status;
}

void HDF5WriterBase::setMode(unsigned int mode)
{
    if (mode == H5F_ACC_RDWR || mode == H5F_ACC_TRUNC || mode == H5F_ACC_EXCL){
        openmode_ = mode;
    }
}

unsigned HDF5WriterBase::getMode() const
{
    return openmode_;
}
// Subclasses should reimplement this for flushing data content to
// file.
void HDF5WriterBase::flush()
{
    cout << "HDF5WriterBase:: flush() " << endl;// do nothing
}

#endif // USE_HDF5
// 
// HDF5WriterBase.cpp ends here
