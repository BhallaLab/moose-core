// HDF5WriterBase.h --- 
// 
// Filename: HDF5WriterBase.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sat Feb 25 14:39:19 2012 (+0530)
// Version: 
// Last-Updated: Wed Nov 14 17:58:53 2012 (+0530)
//           By: subha
//     Update #: 53
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// HDF5WriterBase provides a common interface for writing data/model to file.
// 
// 

// Change log:
// 
// 2012-02-25 14:39:36 (+0530) subha - started initial implementation
// 

// Code:

#ifdef USE_HDF5
#ifndef _HDF5IO_H
#define _HDF5IO_H
#include <typeinfo>
class HDF5WriterBase
{
  public:
    HDF5WriterBase();
    virtual ~HDF5WriterBase();
    void setFilename(string filename);
    string getFilename() const;
    bool isOpen() const;
    void setMode(unsigned int mode);
    unsigned int getMode() const;
    void setSAttr(string name, string value);
    void setFAttr(string name, double value);
    void setIAttr(string name, long value);
    string getSAttr(string name) const;
    double getFAttr(string name) const;
    long getIAttr(string name) const;            
    virtual void flush();
    void close();
    
    static const Cinfo* initCinfo();
    
  protected:
    herr_t openFile();
    
    /// map from element path to nodes in hdf5file.  Multiple MOOSE
    /// tables can be written to the single file corresponding to a
    /// HDF5Writer. Each one will be represented by a specific data
    /// node in the file.
    map <string, hid_t> nodemap_;
    /// File handle for the HDF5 file opened by this object
    hid_t filehandle_;
    string filename_;
    unsigned int openmode_;
    // We also allow attributes of string, double or long int type on / node
    map<string, string> sattr_;
    map<string, double> fattr_;
    map<string, long> iattr_;
};

template <typename A>
void writeRootAttr(hid_t fileid, const map< string, A> & attr)
{
    hid_t dataid;
    hid_t attrid;
    hid_t dtype;
    for (typename map<string, A>::const_iterator ii = attr.begin(); ii != attr.end(); ++ii){
        dataid = H5Screate(H5S_SCALAR);
        const void * data = (void*)(&ii->second);
        if (typeid(A) == typeid(std::string)){
            dtype = H5Tcopy(H5T_C_S1);
            string * s = reinterpret_cast<string*>(const_cast<void*>(data));
            H5Tset_size(dtype, s->length());
            data = s->c_str();
        } else if (typeid(A) == typeid(double)){
            dtype = H5T_NATIVE_DOUBLE;
        } else if (typeid(A) == typeid(long)){
            dtype = H5T_NATIVE_LONG;
        } else {
            cerr << "Error: handling of " << typeid(A).name() << " not handled." << endl;
            return;
        }
        if (H5Aexists(fileid, ii->first.c_str())){
            attrid = H5Aopen(fileid, ii->first.c_str(), H5P_DEFAULT);
        } else {
            attrid = H5Acreate2(fileid, ii->first.c_str(), dtype, dataid, H5P_DEFAULT, H5P_DEFAULT);
        }
        if (attrid < 0){
            cerr <<  "Error: failed to open/create attribute " << ii->first << ". Return value: " << attrid << endl;
            continue;
        }        
        herr_t status = H5Awrite(attrid, dtype, data);
        if (status < 0){
            cerr << "Error: failed to write attribute " << ii->first << ". Status code=" << status << endl;
        }
    }    
}

#endif // _HDF5IO_H
#endif // USE_HDF5



// 
// HDF5WriterBase.h ends here
