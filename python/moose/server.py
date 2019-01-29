# -*- coding: utf-8 -*-
"""server.py: 
TCP socket server to handle incoming requests to simulate model.
"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2019, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import socket
import tarfile
import tempfile
import subprocess32 as subprocess

try:
    import socketserver
except ImportError as e:
    import SocketServer as socketserver

__all__ = [ 'serve' ]

def split_data( data ):
    prefixLenght = 10
    return data[:prefixLenght].strip(), data[prefixLenght:]

def find_files_to_run( files ):
    """Any file name starting with __main is to be run."""
    toRun = []
    for f in files:
        if '__main' in os.path.basename(f):
            toRun.append(f)

    if toRun:
        return toRun

    # Then guess.
    if len(files) == 1:
        return files

    for f in files:
        with open(f, 'r' ) as fh:
            txt = fh.read()
            if re.search(r'def\s+main\(', txt):
                if re.search('^\s+main\(\S+?\)', txt):
                    toRun.append(f)
    return toRun

class MooseServerHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for moose server.

    """
    def handle( self ):
        print( "[INFO ] Handing request" )
        data = ''
        bufsize = 2048
        while 1:
            d = self.request.recv(bufsize)
            if len(d) < 1:
                break
            data += d
        prefix, rest = split_data(data)
        if prefix ==  '>TARFILE':
            dataFile = os.path.join( tempfile.mkdtemp(), 'data.tar.bz2' )
            with open(dataFile, 'wb' ) as f:
                f.write(rest)
            if tarfile.is_tarfile(dataFile):
                self.request.sendall( "[INFO ] Successfully recieved data.")
                self.simulate(dataFile)
            else:
                self.request.sendall( "[ERROR] Data was corrupted. Please retry..." )

    def simulate(self, tfile ):
        """Simulate a given tar file.
        """
        tdir = os.path.dirname( tfile )
        os.chdir( tdir )
        userFiles = None
        with tarfile.open(tfile, 'r' ) as f:
            userFiles = f.getnames( )
            f.extractall()

        # Now simulate.
        toRun = find_files_to_run(userFiles)
        if len(toRun) < 1:
            self.request.sendall( "MOOSE could not determine which file to execute." )
        [ self.run_file(f) for f in toRun ]

    def run_file( self, filename ):
        print( "[INFO ] Running file %s" % filename )
        subprocess.run( [ sys.executable, filename ] )
        print( 'Done' )
        

def serve(host='localhost', port=31417):
    s = socketserver.TCPServer( (host, port), MooseServerHandler )
    s.serve_forever()

def test():
    serve()

if __name__ == '__main__':
    test()
