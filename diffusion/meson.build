# Author: Subhasis Ray
# Date: Sun Jul  7

diffusion_src = ['FastMatrixElim.cpp',
                 'DiffPoolVec.cpp',
                 'Dsolve.cpp',
                 'testDiffusion.cpp']

diffusion_lib = static_library('diffusion', diffusion_src, include_directories: gsl_dep.get_variable(pkgconfig:'includedir'))
