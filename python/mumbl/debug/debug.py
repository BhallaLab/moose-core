import inspect
import sys

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[90m'
ERR = '\033[31m'
ENDC = '\033[0m'
RED = ERR
WARN = WARNING
INFO = OKGREEN
TODO = OKBLUE
DEBUG = HEADER
ERROR = ERR

prefix = dict(
    ERR = ERR
    , ERROR = ERR
    , WARN = WARN
    , FATAL = ERR
    , INFO = INFO
    , TODO = TODO
    , NOTE = HEADER
    , DEBUG = DEBUG
    )

def colored(msg, label="INFO") :
    """
    Return a colored string. Formatting is optional.

    At each ` we toggle the color.
    
    """
    global prefix
    if label in prefix :
        color = prefix[label]
    else :
        color = ""
    txt = ''
    newMsg = msg.split('`')
    i = 0
    for m in newMsg:
        if i % 2 == 0:
            txt += color + m
        else:
            txt += ENDC + m
        i += 1
    return "{0} {1}".format(txt, ENDC)

def cl(msg, label="INFO"):
    return colored(msg, label)

def printDebug(label, msg, frame=None, exception=None):
    ''' If msg is a list then first msg in list is the main message. Rest are
    sub message which should be printed prefixed by \n\t.
    '''

    prefix = '[{0}] '.format(label)

    ''' Enable it if you want indented messages 
    stackLength = len(inspect.stack()) - 1
    if stackLength == 1:
        prefix = '\n[{}] '.format(label)
    else:
        prefix = ' '.join(['' for x in range(stackLength)])
    '''

    if type(msg) == list:
        if len(msg) > 1:
            msg = [msg[0]] + ["`|- {0}`".format(x) for x in msg[1:]] 
        msg = (prefix+"\n\t").join(msg)


    if not frame :
        print(prefix+"{0}".format(colored(msg,label)))
    else :
        filename = frame.f_code.co_filename
        filename = "/".join(filename.split("/")[-2:])
        print(prefix+"@{0}:{1} {2}".format(filename, frame.f_lineno, colored(msg, label)))
    if exception:
        print(" [Expcetion] {0}".format(exception))

