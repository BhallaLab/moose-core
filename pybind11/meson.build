# Author: Subhasis Ray
# Date: Sun Jul  7

pybind11_src = ['Finfo.cpp',
                'helper.cpp',
                'MooseVec.cpp',
                # 'pymoose.cpp',
                'PyRun.cpp']

python_res = run_command('python', '-c', 'from sysconfig import get_paths as gp; print(gp()["include"])')
if python_res.returncode() == 0
  inc_dirs = python_res.stdout().strip()
else
  message('ERROR: could not find Python include directory')
  inc_dirs = []
endif
                                
pybind11_lib = static_library('pybind11', pybind11_src, dependencies: pybind_dep, include_directories: inc_dirs)

