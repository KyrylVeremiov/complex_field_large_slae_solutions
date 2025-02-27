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

from pandas.core.arrays import DatetimeArray

from analyse import check_directory
from constants import *
import analyse

# https://petsc.org/release/overview/linear_solve_table/

import time
from contextlib import redirect_stdout
# import subprocess

from multiprocessing import Process
import psutil
import signal

import petsc4py
petsc4py.init()
from petsc4py import PETSc

def get_best_results(filename):
    with open(filename, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object




# Number is ID in the SuiteSparse Matrix Collection
# See https://github.com/drdarshan/ssgetpy?tab=readme-ov-file
# and https://sparse.tamu.edu/
test_matrix_types= [
    # 2,
    # "random",
    # "hilbert"

    # 281,
    # 1621,
    # 1620,
    # 1054,
    # 326,
    # 1595,
    # 1597,


    # 378,
    # 540,
    # 1366,

    # 435,
    # 312,
    # 39,
    # 443,
    811
]
import scipy as sp
from ssgetpy import search, fetch
import numpy as np


N=[100]#matrix size
# N=[10,100,250,500,1000,2000,3000,5000]#matrix size
# N=[10,100,250,500,1000]#matrix size





#########################    FOR INDIVIDUAL METHOD TEST    #########################
# PARAMS= [
#     {"method":"QMRCGS","preconditioner": "SOR"},
    # {"method":"BCGS","preconditioner":"GAMG"},
    # {"method":"RICHARDSON","preconditioner":"NONE"},
    # {"method":"RICHARDSON","preconditioner":"GAMG"},
    # {"method":"GMRES","preconditioner":"GAMG"},
    # {"method":"GMRES","preconditioner":"LMVM"},
    # {"method":"BCGS","preconditioner":"LMVM"},
    # {"method": "BCGS", "preconditioner": "ILU"},
#     {"method":"GMRES","preconditioner":"NONE"},
#     {"method":"BCGS","preconditioner":"NONE"},
#     {"method":"CG","preconditioner":"GAMG"}
# ]


#########################   FOR TESTING WITH ALL PRECONDITIONERS OR GRID TEST   #########################
#
# https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.PC.Type.html
preconditioners=[el for el in PETSc.PC.Type.__dict__.keys() if el[:1] != '_']
# https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.Type.html
methods=[el for el in PETSc.KSP.Type.__dict__.keys() if el[:1] != '_']
#########################FOR GRID TEST
# preconditioners=["GAMG","NONE","ILU"]
# methods=[
#     "GMRES",
#     "BCGS",
#     "RICHARDSON"
# ]



#########################    TESTING FORM FILE    #########################
# PARAMS=get_best_results(BEST_RES_REL_THR_RESULT_FILENAME)

# PARAMS=get_best_results(BEST_RES_REL_THR_TIME_THR_RESULT_FILENAME)

PARAMS=get_best_results(BEST_RES_REL_THR_TIME_THR_RESULT_FILENAME)

methods_set=set()
preconditioners_set=set()

for item in PARAMS:
    methods_set.add(item["method"])
    preconditioners_set.add(item["preconditioner"])

print(len(methods_set), "methods")
print(methods_set)
print(len(preconditioners_set), "preconditioners")
print(preconditioners_set)

# print(len(methods), "methods")
# print(len(preconditioners), "preconditioners")

print(len(PARAMS), "configurations")
print("Configurations: ",PARAMS)

starting_point=0
# starting_point={"method":"FBCGSR","preconditioner":"DEFLATION"}

exception_list=[
    {"method":"FGMRES","preconditioner":"REDISTRIBUTE"}
]



# start_test=False
# for test_matrix_type in test_matrix_types:
#     for param in PARAMS:
#         if (starting_point==0) or (param==starting_point):
#             start_test=True
#         if start_test:
#             if param not in exception_list:
#                 for n in N:
#                     iterator= PARAMS.index(param)
#                     print(f"method {iterator} of {len(PARAMS)} on this matrix type")
#                     # iterator+=1
#                     print(param,f"{f'on suite sparse matrix with id {test_matrix_type}' if type(test_matrix_type)==int else f' on {test_matrix_type} matrix of size n={n}'} begins at {datetime.now()}")
#
#
#
#                     # argument = f"'{json.dumps(n)}' '{json.dumps(SEED)}' '{json.dumps(param)}' '{json.dumps(test_matrix_type)}'"
#                     # # proc = subprocess.Popen(['python', 'slae_testing.py', f"'{json.dumps(n)}'",f"'{json.dumps(SEED)}'",f"'{json.dumps(param)}'",
#                     # #                          f"'{json.dumps(test_matrix_type)}'"], shell=True)
#                     # # proc = subprocess.Popen(['python', f" slae_testing.py '{json.dumps(n)}' '{json.dumps(SEED)}' '{json.dumps(param)}' '{json.dumps(test_matrix_type)}'"], shell=True)
#                     # proc = subprocess.Popen(['python', 'slae_testing.py', argument], shell=True)
#                     # # time.sleep(3)  # <-- There's no time.wait, but time.sleep.
#                     # # pid = proc.pid
#                     # # proc.kill()
#                     #
#                     # start_time = time.time()
#                     # check_interval_s = 5  # regularly check what the process is doing
#                     #
#                     # kill_process = False
#                     # finished_work = False
#                     #
#                     # while not kill_process and not finished_work:
#                     #     time.sleep(check_interval_s)
#                     #     runtime = time.time() - start_time
#                     #     # print("Doing, ", runtime)
#                     #
#                     #     if not proc.poll() is None:
#                     #         print("Finished")
#                     #         finished_work = True
#                     #
#                     #     elif runtime > TIMEOUT:
#                     #         print("EXCEEDED TIMEOUT LIMIT")
#                     #         kill_process = True
#                     # proc.kill()
#                     #
#                     # if kill_process:
#                     #     check_directory(RESULTS_DIRECTORY)
#                     #     file_name = f"./{RESULTS_DIRECTORY}/{f'n{n}_s{SEED}_' if type(test_matrix_type) != int else ''}met{param['method']}_pc{param['preconditioner']}_mat_{test_matrix_type}"
#                     #
#                     #     with (open(file_name + ".txt", 'w') as f, redirect_stdout(f)):
#                     #         print("EXCEEDED TIMEOUT LIMIT")
#
#                     # cmd_command=f"python slae_testing.py '{json.dumps(n)}' '{json.dumps(SEED)}' '{json.dumps(param)}' '{json.dumps(test_matrix_type)}'"
#                     cmd_command=f"/home/userone/miniconda3/envs/env3-10-complex/bin/python slae_testing.py '{json.dumps(n)}' '{json.dumps(SEED)}' '{json.dumps(param)}' '{json.dumps(test_matrix_type)}'"
#                     p = Process(target=lambda: os.system(cmd_command))
#                     p.start()
#
#                     start_time = time.time()
#                     check_interval_s = 5  # regularly check what the process is doing
#
#                     kill_process = False
#                     finished_work = False
#
#                     while not kill_process and not finished_work:
#                         time.sleep(check_interval_s)
#                         runtime = time.time() - start_time
#                         # print("Doing, ", runtime)
#
#                         if not p.is_alive():
#                             print("Finished")
#                             finished_work = True
#
#                         if runtime > TIMEOUT and not finished_work:
#                             print("EXCEEDED TIMEOUT LIMIT")
#                             kill_process = True
#
#                     if kill_process:
#                         check_directory(RESULTS_DIRECTORY)
#                         file_name = f"./{RESULTS_DIRECTORY}/{f'n{n}_s{SEED}_' if type(test_matrix_type) != int else ''}met{param['method']}_pc{param['preconditioner']}_mat_{test_matrix_type}"
#
#                         with (open(file_name + ".txt", 'w') as f, redirect_stdout(f)):
#                             print("EXCEEDED TIMEOUT LIMIT")
#                         # while p.is_alive():
#                         #     # forcefully kill the process, because often (during heavvy computations) a graceful termination
#                         #     # can be ignored by a process.terminate
#                         #     # print(f"send SIGKILL signal to process because exceeding {TIMEOUT} seconds.")
#                         #     os.system(f"kill -9 {p.pid}")
#                         #
#                         #
#                         #     if p.is_alive():
#                         #         time.sleep(5)
#
#
#                     for pr in psutil.process_iter():
#                         if cmd_command in pr.name() or cmd_command in ' '.join(pr.cmdline()):
#                             # pr.terminate()
#                             # pr.wait()
#                             # pr.kill()
#                             pid=pr.pid
#                             try:
#                                 os.kill(pid+1, signal.SIGTERM)
#                                 # print(f"Sent SIGTERM signal to process {pid}")
#                             except OSError:
#                                 print(f"Failed to send SIGTERM signal to process {pid}")
#                             # print(pr.pid)
#                             # print(pr.cmdline())
#
#
#
#                     while p.is_alive():
#                         # forcefully kill the process, because often (during heavvy computations) a graceful termination
#                         # can be ignored by a process.
#                         # print(f"send SIGKILL signal to process because exceeding {TIMEOUT} seconds.")
#                         p.kill()
#                         # os.system(f"kill -9 {p.pid}")
#
#                         if p.is_alive():
#                             time.sleep(5)
#
#                     try:
#                         p.join(30)  # wait 30 seconds to join the process
#                         # print("Processes are successfully joined")
#
#                     except Exception:
#                         # This can happen if a process was killed for other reasons (such as out of memory)
#                         print("Joining the process and receiving results failed, results are set as invalid.")
#
#
#
#                     if type(test_matrix_type)==int:
#                         break
#
# analyse.main()
# print("Successfully displayed")

# mpiexec -n 2 python main.py








# PARAMS= [{"method":PETSc.KSP.Type.PIPEBCGS,"preconditioner": PETSc.PC.Type.ILU},
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
