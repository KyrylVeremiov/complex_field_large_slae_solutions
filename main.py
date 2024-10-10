#%%
# petsc4py.init(["-log_view",  "log.txt:ascii_xml"])
# petsc4py.init(["-log_view",  "/home/userone/PycharmProjects/scientificProject/log.txt:ascii_xml"])

# import sys
# import numpy as np
# import time
# from petsc4py.typing import KSPMonitorFunction




#%%
import os
import json
from datetime import datetime

N=[10,100,1000,3000,5000]#matrix size
SEED=10
PARAMS= [
        {"method":"CG","preconditioner":"GAMG"},
        {"method":"GMRES","preconditioner":"ILU"},
        {"method": "BCGS", "preconditioner": "ILU"}
         ]
for n in N:
    for param in PARAMS:
        print(param,f"of size n={n} begins at {datetime.now()}")
        os.system(f"python slae_testing.py '{json.dumps(n)}' '{json.dumps(SEED)}' '{json.dumps(param)}'")

# mpiexec -n 2 python main.py

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