# Author: Subhasis Ray
# Date: Sun Jul  7

kinetics_src = ['PoolBase.cpp',
                'Reac.cpp',
                'EnzBase.cpp',
                'Enz.cpp',
                'MMenz.cpp',
                'Species.cpp',
                'ConcChan.cpp',
                'ReadKkit.cpp',
                'WriteKkit.cpp',
                'ReadCspace.cpp',
                'lookupVolumeFromMesh.cpp',
                'testKinetics.cpp']

kinetics_lib = static_library('kinetics', kinetics_src, include_directories: gsl_dep.get_variable(pkgconfig:'includedir'))
