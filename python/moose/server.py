# -*- coding: utf-8 -*-
from __future__ import print_function, division

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
import re
import os 
import time
import socket 
import tarfile 
import tempfile 
import threading 
import subprocess32 as subprocess

__all__ = [ 'serve' ]

stop_all_ = False

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

def recv_input(conn, size=1024):
    data = conn.recv(size)
    return data

def writeTarfile( data ):
    tfile = os.path.join(tempfile.mkdtemp(), 'data.tar.bz2')
    with open(tfile, 'wb' ) as f:
        print( "[INFO ] Writing %d bytes to %s" % (len(data), tfile))
        f.write(data)
    time.sleep(0.1)
    assert tarfile.is_tarfile(tfile), "Not a valid tarfile %s" % tfile
    return tfile

def run_file(filename):
    print( '[INFO] Running %s' % filename )
    subprocess.run( [ sys.executable, filename] )
    print( '.... DONE' )

def simulate( tfile ):
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
            return 1
        [ run_file( _file ) for _file in toRun ]


def savePayload( conn ):
    tarData = b''
    tarFileStart, tarfileName = False, None
    data = recv_input(conn)
    if b'<TARFILE>' in data:
        print( "[INFO ] GETTING PAYLOAD." )
        tarFileStart = True
        tarData += data.split( '<TARFILE>' )[1]

    while tarFileStart and (b'</TARFILE>' not in data):
        tarData += data
        data = recv_input(conn)

    tarData += data.split( '</TARFILE>' )[0]
    tarfileName = writeTarfile( tarData )
    return tarfileName

def handle_client(conn, ip, port):
    isActive = True
    tarData = b''
    while isActive:
        tarfileName = savePayload(conn)
        print( "[INFO ] PAYLOAD RECIEVED." )
        if os.path.isfile(tarfileName):
            simulate(tarfileName)
            conn.sendall( '>DONE' )
            isActive = False

def start_server( host, port, max_requests = 10 ):
    global stop_all_
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print( "[INFO ] Server created." )
    try:
        soc.bind( (host, port))
    except Exception as e:
        print( "[ERROR] Failed to bind: %s" % str(sys.exec_info()))
        quit(1)

    # listen upto 100 of requests
    soc.listen(max_requests)
    while True:
        conn, (ip, port) = soc.accept()
        print( "[INFO ] Connected with %s:%s" % (ip, port) )
        try:
            t = threading.Thread(target=handle_client, args=(conn, ip, port)) 
            t.start()
        except Exception as e:
            print(e)
        if stop_all_:
            break
    soc.close()

def serve(host, port):
    start_server(host, port)

def main():
    global stop_all_
    host, port = 'localhost', 31417
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = sys.argv[2]
    try:
        serve(host, port)
    except KeyboardInterrupt as e:
        stop_all_ = True
        quit(1)

if __name__ == '__main__':
    main()
