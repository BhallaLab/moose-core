import moose
import time

def profile_creation():
    print('Only creation')
    moose.Neutral('x')
    t = time.time()
    for i in range(10000):
        moose.Neutral('x/x%d'%i)
    print('Time taken %g s' % time.time()-t)
    moose.delete('x')

def profile_creation_destruction():
    print('Create and delete')
    t = time.time()
    for i in range(10000):
        moose.Neutral('a')
        moose.delete('a')
    print('Time taken %g s' % time.time()-t)

if __name__ == "__main__":
    profile_creation()
    profile_creation_destruction()
