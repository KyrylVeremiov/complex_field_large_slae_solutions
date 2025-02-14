import json
import sys
import scipy as sp
from contextlib import redirect_stdout

from ssgetpy import search, fetch

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import os

from analyse import check_directory
from constants import *
from get_convergence_reason import get_convergence_reason


def run_test(n, seed, parameters, mat_test_type):
    check_directory(RESULTS_DIRECTORY)
    file_name=f"./{RESULTS_DIRECTORY}/{f'n{n}_s{seed}_' if type(mat_test_type) != int else ''}met{parameters['method']}_pc{parameters['preconditioner']}_mat_{mat_test_type}"
    # print(file_name)

    with (open(file_name+".txt", 'w') as f, redirect_stdout(f)):
        import petsc4py
        # petsc4py.init(arch="complex")
        petsc4py.init()

        # petsc4py.init(arch="linux-gnu-complex-64")
        from petsc4py import PETSc
        # petsc4py.init(arch="linux-gnu-complex-64")
        # petsc4py.init(arch="/usr/lib/petscdir/petsc3.10/x86_64-linux-gnu-complex/")

        def get_random_b(b: PETSc.Vec,data_dir, seed: int = 10):

            b_filename=data_dir+"/"+"b.dat"
            if os.path.isfile(b_filename):
                viewer_b = PETSc.Viewer().createBinary(b_filename, mode=PETSc.Viewer.Mode.READ)
                b = PETSc.Vec().load(viewer_b)
            else:
                np.random.seed(seed=seed)
                rstart, rend = b.getOwnershipRange()
                for row in range(rstart, rend):
                    b[row] = np.random.rand() * 10 + 0.1+  np.random.rand() * 10j
                    # b[row] = np.random.rand() * 10
                b.assemblyBegin()
                b.assemblyEnd()

                viewer_b = PETSc.Viewer().createBinary(b_filename, mode=PETSc.Viewer.Mode.WRITE)
                viewer_b(b)
            return b


        def get_random_matrix(n: int, seed: int = 10):
            np.random.seed(seed=seed)


            # This demo is structured as a script to be executed using:
            #   $ python main.py
            #
            # potentially with additional options passed at the end of the command.
            #
            # At the start of your script, call `petsc4py.init` passing `sys.argv` so that
            # command-line arguments to the script are passed through to PETSc.

            # opts=PETSc.Options()
            # opts["history"]="hist.txt"
            # opts["log_view"]="log1.txt"
            # print(opts.getAll())

            # PETSc is extensively programmable using the `PETSc.Options` database. For
            # more information see `working with PETSc Options <petsc_options>`.
            # OptDB = PETSc.Options()

            # Grid size and spacing using a default value of ``5``. The user can specify a
            # different number of points in each direction by passing the ``-n`` option to
            # the script.

            # n = OptDB.getInt('n', N)

            # Matrices are instances of the `PETSc.Mat` class.
            A = PETSc.Mat()
            #
            # Create the underlying PETSc C Mat object.
            # You can omit the ``comm`` argument if your objects live on
            # `PETSc.COMM_WORLD` but it is a dangerous choice to rely on default values
            # for such important arguments.

            A.create(comm=PETSc.COMM_WORLD)

            # Specify global matrix shape with a tuple.

            # A.setSizes(((n, n), (n , n)))
            A.setSizes((n, n))
            # A.setSizes(n)

            # The call above implicitly assumes that we leave the parallel decomposition of
            # the matrix rows to PETSc by using `PETSc.DECIDE` for local sizes.
            # It is equivalent to:
            # A.setSizes(((PETSc.DECIDE, n), (PETSc.DECIDE, n ))

            # Here we use a sparse matrix of AIJ type
            # Various `matrix formats <petsc4py.PETSc.Mat.Type>` can be selected:
            A.setType(PETSc.Mat.Type.AIJ)

            # Finally we allow the user to set any options they want to on the matrix from
            # the command line:
            A.setFromOptions()

            # Insertion into some matrix types is vastly more efficient if we preallocate
            # space rather than allow this to happen dynamically. Here we hint the number
            # of nonzeros to be expected on each row.
            A.setPreallocationNNZ(n)  # ??????????????????????????????????????????????????????????????????? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # We can now write out our finite difference matrix assembly using conventional
            # Python syntax. `Mat.getOwnershipRange` is used to retrieve the range of rows
            # local to this processor.


            rstart, rend = A.getOwnershipRange()
            for row in range(rstart, rend):
                for col in range(rstart, rend):
                    A[row, col] = np.random.rand() * 10 + 0.1+ np.random.rand() * 10j

            # At this stage, any exchange of information required in the matrix assembly
            # process has not occurred. We achieve this by calling `Mat.assemblyBegin` and
            # then `Mat.assemblyEnd`.

            A.assemblyBegin()
            A.assemblyEnd()

            # We set up an additional option so that the user can print the matrix by
            # passing ``-view_mat`` to the script.

            # A.viewFromOptions('-view_mat')

            # A.view()
            A.convert(PETSc.Mat.Type.SEQAIJ)
            return (A,"random")

        def get_special_matrix(mat_type):
            if mat_type=="hilbert":
                # A_numpy =np.array(sp.linalg.hilbert(n) , dtype=np.int32)+0j
                # A_numpy =sp.linalg.hilbert(n).astype(np.float64)*1000
                A_numpy =sp.linalg.hilbert(n).astype(np.float64)
            A = PETSc.Mat().create()
            A.setSizes([n, n])
            # A.setType("aij")
            A.setUp()

            # First arg is list of row indices, second list of column indices
            # A.setValues([1, 2, 3], [0, 5, 9], np.ones((3, 3)))
            # A.setValues(np.arange(n), np.arange(n), A_numpy)
            A.setValues(np.arange(n).astype(np.int32), np.arange(n).astype(np.int32), A_numpy)
            A.assemble()
            # print(sp.linalg.hilbert(n))
            return (A,mat_type)

        def get_suite_sparse_matrix(mat_type):

            A_mm=fetch(mat_type)[0]
            data_dir =A_mm.download(destpath=DATA_DIRECTORY, extract=True)[0]
            #
            # print(data_dir+"/"+os.listdir(data_dir)[0])
            # print(A_path[0])

            A_mm_tuple=A_mm.to_tuple()

            matrix_filename=data_dir+"/"+"A.dat"
            if os.path.isfile(matrix_filename):
                A = PETSc.Mat().create()
                viewer_A = PETSc.Viewer().createBinary(matrix_filename, mode=PETSc.Viewer.Mode.READ, comm=A.getComm())
                A = PETSc.Mat().load(viewer_A)
            else:
                A_mat=sp.io.mmread(data_dir+"/"+os.listdir(data_dir)[0]).toarray()
                # if mat_type=="hilbert":
                #     # A_numpy =np.array(sp.linalg.hilbert(n) , dtype=np.int32)+0j
                #     A_numpy =sp.linalg.hilbert(n).astype(np.float64)
                A = PETSc.Mat().create()
                # A_mat.shape[0]
                A.setSizes([A_mat.shape[0], A_mat.shape[1]])
                # A.setType("aij")
                A.setUp()
                # print(A_mat.shape[0])

                # First arg is list of row indices, second list of column indices
                # A.setValues([1, 2, 3], [0, 5, 9], np.ones((3, 3)))
                # A.setValues(np.arange(n), np.arange(n), A_numpy)
                A.setValues(np.arange(A_mat.shape[0]).astype(np.int32), np.arange(A_mat.shape[1]).astype(np.int32), A_mat)
                A.assemble()
                # print(sp.linalg.hilbert(n))

                viewer_A = PETSc.Viewer().createBinary(matrix_filename, mode=PETSc.Viewer.Mode.WRITE, comm=A.getComm())
                viewer_A(A)

            return (A,(A_mm_tuple[11]+"_"+A_mm_tuple[1]+"_"+A_mm_tuple[2]).replace('/','_or_').replace('-','_'),data_dir)


        def test_slae_solution(ns, seed, alg_params):
            # The full PETSc4py API is to be found in the `petsc4py.PETSc` module.

            np.random.seed(seed=seed)
            results = {}
            for n in ns:
                if mat_test_type== "random":
                    A,mat_test_name = get_random_matrix(n, seed)
                elif mat_test_type== "hilbert":
                    A,mat_test_name=get_special_matrix(mat_type=mat_test_type)
                elif type(mat_test_type)==int:
                    A,mat_test_name,A_directory=get_suite_sparse_matrix(mat_type=mat_test_type)
                    n=A.size[0]
                else:
                    raise Exception("UNKNOWN TYPE OF TEST MATRIX")

                # print(type(mat_test_type))
                results[n] = []

                # plot_portrait_matrix(A=A, n=n, mat_test_type=mat_test_name, norm="LogNorm")

                x, b = A.createVecs()
                # x.view()
                # b.view()
                b=get_random_b(b,A_directory,seed)

                for param in alg_params:
                    # PETSc represents all linear solvers as preconditioned Krylov subspace methods
                    # of type `PETSc.KSP`. Here we create a KSP object for a conjugate gradient
                    # solver preconditioned with an algebraic multigrid method.

                    # L=PETSc.Log.Event("log1")
                    # L.activate()
                    # L.push()

                    # del PETSc
                    # del petsc4py
                    PETSc.Log.begin()

                    # PETSc.LogEvent.begin(1)
                    # PETSc.LogEvent.activate(1)

                    ksp = PETSc.KSP()
                    ksp.create(comm=A.getComm())

                    # We set the matrix in our linear solver and allow the user to program the
                    # solver with options.
                    # ksp.setMonitor(petsc4py.typing.KSPMonitorFunction)
                    ksp.setOperators(A)
                    ksp.setFromOptions()

                    # Since the matrix knows its size and parallel distribution, we can retrieve
                    # appropriately-scaled vectors using `Mat.createVecs`. PETSc vectors are
                    # objects of type `PETSc.Vec`. Here we set the right-hand side of our system to
                    # a vector of ones, and then solve.

                    ksp.setType(param["method"])
                    # ksp.setType(PETSc.KSP.Type.GMRES)
                    # ksp.getPC().setType(PETSc.PC.Type.LU)
                    ksp.getPC().setType(param["preconditioner"])

                    # ksp.setTolerances(rtol=RTOL,atol=RESIDUAL_NORM_TRESHOLD,divtol=DIVTOL,max_it=MAXIT)
                    # ksp.getPC().setFactorLevels(1)

                    # PETSc.Log.begin()
                    # start_time = time.time()
                    ksp.setUp()
                    # end_time = time.time()

                    # ksp.logConvergenceHistory(ksp.getResidualNorm())
                    start_time = time.time()
                    ksp.solve(b, x)
                    end_time = time.time()
                    results[n].append({"KSP": ksp, "time": end_time - start_time,"mat_name":mat_test_name})


                    # PETSc.LogEvent.end(1)
                    # PETSc.LogEvent.deactivate(1)

                    # petsc4py.PETSc.garbage_cleanup()

                    # L.view()
                    # L.deactivate()
                    # L.pop()
                    # L.view()
            return results

        # Finally, allow the user to print the solution by passing ``-view_sol`` to the script.

        # x.viewFromOptions('-view_sol')
        # x.view()

        # y = A.createVecLeft()
        # A.mult(x, y)
        # y.view()
        # b.view()
        # ksp.
        # print("Residual= ", (y - b).sum())
        # ksp.buildResidual().view()
        # ksp.logConvergenceHistory(ksp.getResidualNorm())
        # ksp.monitor()
        # PETSc.Log.begin()
        # return PETSc.Log.getTime()
        # return PETSc.Log.getCPUTime()
        # return PETSc.Log.getFlops()

        # return PETSc.Log.isActive()
        # return ksp.getConvergenceHistory()
        # Things to try
        # -------------
        # )
        # - Show the solution with ``-view_sol``.
        # - Show the matrix with ``-view_mat``u

        # - Change the resolution with ``-n``.
        # - Use a direct solver by passing ``-ksp_type preonly -pc_type lu``.
        # - Run in parallel on two processors using:
        #
        #   .. code-block:: console
        #
        #       mpiexec -n 2 python main.py
        PARAM={}
        for arg in parameters:
            if arg=="method":
                PARAM[arg]=eval("PETSc.KSP.Type."+parameters[arg])
            if arg=="preconditioner":
                PARAM[arg]=eval("PETSc.PC.Type."+parameters[arg])
        results=test_slae_solution([n],seed,[PARAM])
        # print(results)
        for n in results:
            print(f"n={n}")
            for param in results[n]:
                # param["KSP"].getType().view()
                ksp=param["KSP"]
                # param["KSP"].getSolution().view()

                print(f'matrix type={param["mat_name"]}\n'
                      f'method={ksp.getType()}\npreconditioner={ksp.getPC().type}\ntime={param["time"]} seconds\n'
                      f'computed residual norm={ksp.norm}\nnumber of iterations={ksp.its}\n'
                      f'rtol={ksp.rtol}\natol={ksp.atol}\ndivtol={ksp.divtol}\nmaxit={ksp.max_it}\n'
                      f'is converged={ksp.is_converged}\nis diverged={ksp.is_diverged}\nis iterating={ksp.is_iterating}\n'
                      f'norm type={ksp.norm_type}\nguess nonzero={ksp.guess_nonzero}\n'
                      f'guess knoll={ksp.guess_knoll}\nconverged reason={ksp.reason, get_convergence_reason(ksp.reason)},')

                A=ksp.getOperators()[0]
                x=ksp.getSolution()
                b=ksp.getRhs()
                y = A.createVecLeft()
                A.mult(x, y)
                # A.view()
                # x.view()
                # y.view()
                # b.view()
                # As I understand ksp.norm is computing during iterations, so it uses a preconditioned equation
                # So to now real residual we can compute original matrix equation
                print("Real residual norm= ", np.sqrt(((y - b)*(y - b)).sum()))
                r_c=y-b
                r_c.conjugate()
                print("Real residual norm abs= ", abs(np.sqrt(((y - b)*r_c).sum())))
                print("\n\n\n\n")

                name = ''.join(list(filter(str.isalnum, list(param["mat_name"]))))
                directory_to_save = ANALYSE_DIRECTORY + "/" + param["mat_name"] + "_matrix/"
                filename_to_save_plot = directory_to_save + name + f"_matrix_n_{n}"
                if not os.path.isfile(filename_to_save_plot):
                    plot_portrait_matrix(A=A, n=n, mat_test_type=param["mat_name"], norm="LogNorm")
                # PETSc.MatView(A)
                # PETSc.Viewer.createDraw(A)
                # A.assemble()
                # A.view()


                # ksp.history, ksp.getMonitor())
        PETSc.Log.view()

                # param["KSP"].getSolution().view()
def plot_portrait_matrix(A, n, mat_test_type,norm="NoNorm"):
    fig, ax = plt.subplots()
    # name=mat_test_type
    name=''.join(list(filter(str.isalnum, list(mat_test_type))))

    A_plot=abs(A.getValues(range(0, A.getSize()[0]), range(0, A.getSize()[1])))
    title =(name + "\nmatrix"+f" n={n} absolute values. "
            # +"cond(A)="+str(np.linalg.cond(A_plot,p=2))
            # +"cond(A)="+str(np.linalg.norm(A_plot,2)*np.linalg.norm(np.linalg.inv(A_plot),2))
            )
    # ax.title=title
    # mp=plt.cm.ScalarMappable()
    # mp.set_array(A_plot)
    # fig.colorbar(mappable=plt.cm.ScalarMappable(norm=matplotlib.colors.LogNorm()), ax=ax)
    vmin=A_plot.min()
    if norm== "LogNorm":
        if vmin == 0:
            A_plot+=0.001
        norm = matplotlib.colors.LogNorm(vmin=float(A_plot.min()),vmax=float(A_plot.max()))
    elif norm=="NoNorm":
        norm=matplotlib.colors.Normalize(vmin=vmin,vmax=A_plot.max())
    fig.colorbar(mappable=plt.cm.ScalarMappable(norm=norm), ax=ax)
    # fig.colorbar(mappable=plt.cm.ScalarMappable(), ax=ax)
    # , norm = LogNorm(vmin=0.01, vmax=1)
    ax.matshow(A_plot, norm=norm)
    # ax.pcolormesh(A_plot)
    plt.title(title, fontsize="9")

    directory_to_save=ANALYSE_DIRECTORY + "/" + mat_test_type + "_matrix/"
    check_directory(directory_to_save)
    filename_to_save_plot= directory_to_save + name + f"_matrix_n_{n}"
    plt.savefig(filename_to_save_plot)
    # plt.show()

def main():
    try:
        n= json.loads(sys.argv[1])
        seed= json.loads(sys.argv[2])
        param= json.loads(sys.argv[3])
        test_type =  json.loads(sys.argv[4])
        # print(type(test_type))
        # if test_type.isdigit():
        #     test_type=int(test_type)
    except:
        print("Incorrect arguments")

    # try:
    # print(f"n={n}, seed={seed}, Parameters={param}")
    run_test(n, seed, param, mat_test_type=test_type)

    # except:
    #     print("Error")

    return "Success"

if __name__ == "__main__":
    main()
