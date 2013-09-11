// HDF5WriterBase.cpp --- 
// 
// Filename: HDF5WriterBase.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 14:42:03 2012 (+0530)
// Version: 
// Last-Updated: Wed Nov 14 18:39:19 2012 (+0530)
//           By: subha
//     Update #: 282
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

#include <algorithm>
#include <string>
#include <fstream>

#include "hdf5.h"

#include "header.h"

#include "HDF5WriterBase.h"

using namespace std;
                
const Cinfo* HDF5WriterBase::initCinfo()
{

  //////////////////////////////////////////////////////////////
  // Field Definitions
  //////////////////////////////////////////////////////////////
  static ValueFinfo< HDF5WriterBase, string > fileName(
      "filename",
      "Name of the file associated with this HDF5 writer object.",
      &HDF5WriterBase::setFilename,
      &HDF5WriterBase::getFilename);
  
  static ReadOnlyValueFinfo < HDF5WriterBase, bool > isOpen(
      "isOpen",
      "True if this object has an open file handle.",
      &HDF5WriterBase::isOpen);

  static ValueFinfo <HDF5WriterBase, unsigned int > mode(
      "mode",
      "Depending on mode, if file already exists, if mode=1, data will be"
      " appended to existing file, if mode=2, file will be truncated, if "
      " mode=4, no writing will happen.",
      &HDF5WriterBase::setMode,
      &HDF5WriterBase::getMode);

  static ValueFinfo< HDF5WriterBase, unsigned int> chunkSize(
      "chunkSize",
      "Chunksize for writing array data. Defaults to 100.",
      &HDF5WriterBase::setChunkSize,
      &HDF5WriterBase::getChunkSize);

  static ValueFinfo< HDF5WriterBase, string> compressor(
      "compressor",
      "Compression type for array data. zlib and szip are supported. Defaults to zlib.",
      &HDF5WriterBase::setCompressor,
      &HDF5WriterBase::getCompressor);
  
  static ValueFinfo< HDF5WriterBase, unsigned int> compression(
      "compression",
      "Compression level for array data. Defaults to 6.",
      &HDF5WriterBase::setCompression,
      &HDF5WriterBase::getCompression);

  static LookupValueFinfo< HDF5WriterBase, string, string  > sattr(
      "sattr",
      "String attributes. The key is attribute name, value is attribute value (string).",
      &HDF5WriterBase::setSAttr,
      &HDF5WriterBase::getSAttr);
  
  static LookupValueFinfo< HDF5WriterBase, string, double > fattr(
      "fattr",
      "Float attributes. The key is attribute name, value is attribute value (double).",
      &HDF5WriterBase::setFAttr,
      &HDF5WriterBase::getFAttr);
  
  static LookupValueFinfo< HDF5WriterBase, string, long > iattr(
      "iattr",
      "Integer attributes. The key is attribute name, value is attribute value (long).",
      &HDF5WriterBase::setIAttr,
      &HDF5WriterBase::getIAttr);
  
  static DestFinfo flush(
      "flush",
      "Write all buffer contents to file and clear the buffers.",
      new OpFunc0 < HDF5WriterBase > ( &HDF5WriterBase::flush ));

  static DestFinfo close(
      "close",
      "Close the underlying file. This is a safety measure so that file is not in an invalid state even if a crash happens at exit.",
      new OpFunc0< HDF5WriterBase > ( & HDF5WriterBase::close ));
      

  static Finfo * finfos[] = {
    &fileName,
    &isOpen,
    &mode,
    &chunkSize,
    &compressor,
    &compression,
    &sattr,
    &fattr,
    &iattr,
    &flush,
    &close,
  };
  static string doc[] = {
    "Name", "HDF5WriterBase",
    "Author", "Subhasis Ray",
    "Description", "HDF5 file writer base class. This is not to be used directly. Instead,"
    " it should be subclassed to provide specific data writing functions."
    " This class provides most basic properties like filename, file opening"
    " mode, file open status."
  };


  static Cinfo hdf5Cinfo(
      "HDF5WriterBase",
      Neutral::initCinfo(),
      finfos,
      sizeof(finfos)/sizeof(Finfo*),
      new Dinfo<HDF5WriterBase>(),
      doc, sizeof(doc)/sizeof(string));
  return &hdf5Cinfo;                
}

const hssize_t HDF5WriterBase::CHUNK_SIZE = 1024; // default chunk size


HDF5WriterBase::HDF5WriterBase():
        filehandle_(-1),
        filename_("moose_output.h5"),
        openmode_(H5F_ACC_EXCL),
        chunkSize_(CHUNK_SIZE),
        compressor_("zlib"),
        compression_(6)
{
}

HDF5WriterBase::~HDF5WriterBase()
{
    // derived classes should flush data in their own destructors
    if (filehandle_ < 0){
        return;
    }
    flush();
    herr_t err = H5Fclose(filehandle_);
    filehandle_ = -1;
    if (err < 0){
        cerr << "Error: Error occurred when closing file. Error code: " << err << endl;
    }
}

void HDF5WriterBase::setFilename(string filename)
{
    if (filename_ == filename){
        return;
    }
     
    // // If file is open, close it before changing filename
    // if (filehandle_ >= 0){
    //     status = H5Fclose(filehandle_);
    //     if (status < 0){
    //         cerr << "Error: failed to close HDF5 file handle for " << filename_ << ". Error code: " << status << endl;
    //     }
    // }
    // filehandle_ = -1;
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
        filehandle_ = -1;
        if (status < 0){
            cerr << "Error: failed to close currently open HDF5 file. Error code: " << status << endl;
            return status;
        }
    }
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    // Ensure that all open objects are closed before the file is closed    
    H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG);
    ifstream infile(filename_.c_str());
    bool fexists = infile.good();
    infile.close();
    if (!fexists || openmode_ == H5F_ACC_TRUNC){
        filehandle_ = H5Fcreate(filename_.c_str(), openmode_, H5P_DEFAULT, fapl_id);
    } else if (openmode_ == H5F_ACC_RDWR) {
        filehandle_ = H5Fopen(filename_.c_str(), openmode_, fapl_id);
    } else {
        cerr << "Error: File \"" << filename_ << "\" already exists. Specify mode=" << H5F_ACC_RDWR
             << " for appending to it, mode=" << H5F_ACC_TRUNC
             << " for overwriting it. mode=" << H5F_ACC_EXCL
             << " requires the file does not exist." << endl;
        return -1;
    }
    if (filehandle_ < 0){
        cerr << "Error: Could not open file for writing: " << filename_ << endl;
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

void HDF5WriterBase::setChunkSize(unsigned int size)
{
    chunkSize_ = size;
}

unsigned int HDF5WriterBase::getChunkSize() const
{
    return chunkSize_;
}

void HDF5WriterBase::setCompressor(string name)
{
    compressor_ = name;
    std::transform(compressor_.begin(), compressor_.end(), compressor_.begin(), ::tolower);
}

string HDF5WriterBase::getCompressor() const
{
    return compressor_;
}

void HDF5WriterBase::setCompression(unsigned int level)
{
    compression_ = level;
}

unsigned int HDF5WriterBase::getCompression() const
{
    return compression_;
}

        
// Subclasses should reimplement this for flushing data content to
// file.
void HDF5WriterBase::flush()
{
    cout << "Warning: HDF5WriterBase:: flush() should never be called. Subclasses should reimplement this." << endl;// do nothing
}

void HDF5WriterBase::close()
{
    if (filehandle_ < 0){
        return;
    }
    // First write all attributes of root node.
    writeRootAttr<string>(filehandle_, sattr_);
    writeRootAttr<double>(filehandle_, fattr_);
    writeRootAttr<long>(filehandle_, iattr_);    
    herr_t err = H5Fclose(filehandle_);
    filehandle_ = -1;
    if (err < 0){
        cerr << "Error: closing file. Status code=" << err << endl;
    }
}

void HDF5WriterBase::setSAttr(string name, string value)
{
    sattr_[name] = value;
}

void HDF5WriterBase::setFAttr(string name, double value)
{
    fattr_[name] = value;
}

void HDF5WriterBase::setIAttr(string name, long value)
{
    iattr_[name] = value;
}

string HDF5WriterBase::getSAttr(string name) const
{
    map <string, string>::const_iterator ii = sattr_.find(name);
    if (ii != sattr_.end()){
        return ii->second;
    }
    cerr << "Error: no attribute named " << name << endl;
    return "";
}

double HDF5WriterBase::getFAttr(string name) const
{
    map <string, double>::const_iterator ii = fattr_.find(name);
    if (ii != fattr_.end()){
        return ii->second;
    }
    cerr << "Error: no attribute named " << name << endl;
    return 0.0;
}

long HDF5WriterBase::getIAttr(string name) const
{
    map <string, long>::const_iterator ii = iattr_.find(name);
    if (ii != iattr_.end()){
        return ii->second;
    }
    cerr << "Error: no attribute named " << name << endl;
    return 0.0;
}




#endif // USE_HDF5
// 
// HDF5WriterBase.cpp ends here
