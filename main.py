#%%
# petsc4py.init(["-log_view",  "log.txt:ascii_xml"])
# petsc4py.init(["-log_view",  "/home/userone/PycharmProjects/scientificProject/log.txt:ascii_xml"])

# import sys
# import numpy as np
# import time
# from petsc4py.typing import KSPMonitorFunction


# download from https://gitlab.com/petsc/petsc
# Configurate for complex numbers https://abhigupta.io/2021/12/08/installing-petsc-complex.html

# https://www.pism.io/docs/installation/petsc.html

# conda create -n env3-10-complex -c conda-forge python=3.10 fenics-dolfinx 'petsc=*=complex*' mpich
# conda env export
#%%
import os
import json
from crypt import methods
from datetime import datetime

from constants import SEED
import analyse
import petsc4py
petsc4py.init()
from petsc4py import PETSc

test_matrix_types= [
    2,
    1054,
    "random",
    "hilbert"
]


# N=[10,100,200,500]#matrix size
N=[10]#matrix size
# N=[10,100,250,500,1000,2000,3000,5000]#matrix size

# https://petsc.org/release/overview/linear_solve_table/

#FOR INDIVIDUAL METHOD TEST
PARAMS= [
    # {"method":"BCGS","preconditioner":"GAMG"},
    # {"method":"RICHARDSON","preconditioner":"NONE"},
    # {"method":"RICHARDSON","preconditioner":"GAMG"},
    # {"method":"GMRES","preconditioner":"GAMG"},
    {"method":"GMRES","preconditioner":"LMVM"},
    # {"method": "BCGS", "preconditioner": "ILU"},
    # {"method":"GMRES","preconditioner":"NONE"},
    # {"method":"BCGS","preconditioner":"NONE"},
    # {"method":"CG","preconditioner":"GAMG"}
]


########################3#FOR TESTING WITH ALL PRECONDITIONERS
preconditioners=[el for el in PETSc.PC.Type.__dict__.keys() if el[:1] != '_']
# preconditioners=["GAMG","NONE","ILU"]
methods=[
    "GMRES",
    "BCGS",
    "RICHARDSON"
]
# PARAMS=[{"method":met,"preconditioner":prec} for met in methods for prec in preconditioners]

exception_list=[
#     {"method":"GMRES","preconditioner":"CP"}
]


# Number is ID in the SuiteSparse Matrix Collection
# see https://github.com/drdarshan/ssgetpy?tab=readme-ov-file
# and https://sparse.tamu.edu/

# starting_point={"method":"GMRES","preconditioner":"REDISTRIBUTE"}
starting_point=0
start_test=False
for test_matrix_type in test_matrix_types:
    for param in PARAMS:
        if (starting_point==0) or (param==starting_point):
            start_test=True
        if start_test:
            if param not in exception_list:
                for n in N:
                    print(param,f"{f'on suite sparse matrix with id {test_matrix_type}' if type(test_matrix_type)==int else f' on {test_matrix_type} matrix of size n={n}'} begins at {datetime.now()}")
                    os.system(f"python slae_testing.py '{json.dumps(n)}' '{json.dumps(SEED)}' '{json.dumps(param)}' '{json.dumps(test_matrix_type)}'")
                    if type(test_matrix_type)==int:
                        break
# mpiexec -n 2 python main.py

analyse.main()
print("Successfully displayed")

# PARAMS= [{"method":PETSc.KSP.Type.GMRES,"preconditioner": PETSc.PC.Type.ILU},
# {"method": PETSc.KSP.Type.BCGS, "preconditioner": PETSc.PC.Type.ILU},
#                    ]


# PETSc.Log.begin()


# from slae_testing import test_random_slae_solution
# results=test_random_slae_solution(ns=N,seed=SEED,params= PARAMS)


# PETSc.Log.view()

# print(results)
# for n in results:
#     print("n=",n)
#     for param in results[n]:
#         # param["KSP"].getType().view()
#         ksp=param["KSP"]
#         print("method=",ksp.getType(),", preconditioner=", ksp.getPC().type,", time=", param["time"],
#               " seconds, residual norm=",  ksp.getResidualNorm(),  ksp.its, ksp.history, ksp.getMonitor())
#         # param["KSP"].getSolution().view()

# v=PETSc.Viewer()
# v.create()
# v.setFileName("log.txt")
# v.setFileName("/home/userone/PycharmProjects/scientificProject/log.txt")
# v.setType(PETSc.Viewer.Type.ASCII)
# PETSc.Log.view(v)
# PETSc.Log.view()
# v.flush()
# v.view()

# print(results[10][0]["KSP"].history)
# setConvergenceHistory
# setErrorIfNotConverged
# setInitialGuessKnoll
# setMonitor
# KSPSetTolerances

# https://petsc.org/release/manual/ksp/
# https://petsc.org/release/manual/profiling/#ch-profiling
# https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.html#petsc4py.PETSc.KSP.setUp
# https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.Type.html
