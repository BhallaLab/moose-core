# -*- coding: utf-8 -*-
from __future__ import print_function, division

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2016, Dilawar Singh"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import os
import sys
sys.path.append(os.path.dirname(__file__))
import time
import socket
import numpy as np
import multiprocessing as mp
import moose
import moose.utils as mu
import models
from collections import defaultdict

print( '[INFO] Using moose form %s' % moose.__file__ )

def socket_client(q, done, port=31416):
    # This function waits for socket to be available.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while 1:
        if 1 == done.value:
            print( '[INFO] MOOSE is done before I could connect' )
            return
        try:
            s.connect( ('', port) )
            break
        except Exception:
            continue

    print( 'Py: Connected' )

    # This is client reponsibility to read the data.
    s.settimeout(0.1)
    data = b''
    while True:
        try:
            data += s.recv(1024)
        except socket.timeout:
            print(end='x')
        if 1 == done.value:
            break
    if data:
        res = mu.decode_data(data)
        q.put(res)
    s.close()
    return

def test():
    global finish_all_
    os.environ['MOOSE_STREAMER_ADDRESS'] = 'http://127.0.0.1:31416'
    done = mp.Value('d', 0)
    q = mp.Queue()
    client = mp.Process(target=socket_client, args=(q, done))
    client.start()

    print( '[INFO] Socket client is running now' )
    ts = models.simple_model_a()
    moose.reinit()
    # If TCP socket is created, some delay is often neccessary before start. Don't
    # know why. probably some latency in a fresh TCP socket. A TCP guru can
    # tell.
    time.sleep(0.0)
    moose.start(50)
    time.sleep(1)
    done.value = 1
    print( 'MOOSE is done' )
    res = q.get()
    for k in res:
        a = res[k][1::2]
        b = moose.element(k).vector
        print(k, len(a), len(b))
        assert( (a==b).all())
    # sleep for some time so data can be read.
    client.join()
    print( 'Test 1 passed' )

def main( ):
    test( )
    print( '[INFO] All tests passed' )

if __name__ == '__main__':
    main()
