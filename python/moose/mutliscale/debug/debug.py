# This file defines function which can be used to print in colors. Very useful
# in debugging and printing on console.

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
ERR = '\033[91m'
ENDC = '\033[0m'
RED = ERR
WARN = WARNING
INFO = OKGREEN 
TODO = OKBLUE
DEBUG = HEADER

prefix = dict(
    ERR = ERR
    , WARN = WARN
    , FATAL = ERR
    , INFO = INFO
    , TODO = TODO 
    , NOTE = HEADER 
    , DEBUG = DEBUG
    )

def colored(msg, label) :
  """
  Return a colored string. Formatting is optional.
  """
  global prefix
  if label in prefix :
    color = prefix[label]
  else :
    color = ""
  return "[{0}] {1} {2}".format(label, color+msg, ENDC)

def printDebug(label, msg):
  print(colored(msg, label))
