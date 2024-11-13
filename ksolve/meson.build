# Author: Subhasis Ray
# Date: Sun Jul  7

ksolve_src = ['KinSparseMatrix.cpp',
               'VoxelPoolsBase.cpp',
               'VoxelPools.cpp',
               'GssaVoxelPools.cpp',
               'RateTerm.cpp',
               'FuncTerm.cpp',
               'Stoich.cpp',
               'Ksolve.cpp',
               'Gsolve.cpp',
               'KsolveBase.cpp',
               'SteadyStateGsl.cpp',
               'testKsolve.cpp',
               # '../utility/utility.cpp'
             ]

ksolve_lib = static_library('ksolve', ksolve_src, dependencies: [gsl_dep], include_directories: gsl_dep.get_variable(pkgconfig:'includedir'))

