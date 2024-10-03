#%%
import sys
import petsc4py
from petsc4py import PETSc
import numpy as np
import time

#%%
def get_random_matrix(n: int, seed: int = 10):
    np.random.seed(seed=seed)
    # PETSc is extensively programmable using the `PETSc.Options` database. For
    # more information see `working with PETSc Options <petsc_options>`.
    OptDB = PETSc.Options()

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

    def index_to_grid(r):
        """Convert a row number into a grid point."""
        return (r // n, r % n)

    rstart, rend = A.getOwnershipRange()
    for row in range(rstart, rend):
        for col in range(rstart, rend):
            A[row, col] = np.random.rand() * 10 + 0.1

    # At this stage, any exchange of information required in the matrix assembly
    # process has not occurred. We achieve this by calling `Mat.assemblyBegin` and
    # then `Mat.assemblyEnd`.

    A.assemblyBegin()
    A.assemblyEnd()

    # We set up an additional option so that the user can print the matrix by
    # passing ``-view_mat`` to the script.

    # A.viewFromOptions('-view_mat')

    # A.view()
    return A

def get_random_b(b: PETSc.Vec , seed: int = 10):
    np.random.seed(seed=seed)
    rstart, rend = b.getOwnershipRange()
    for row in range(rstart, rend):
        b[row]=np.random.rand()*10+0.1


def test_random_slae_solution(ns, seed, params):
    # The full PETSc4py API is to be found in the `petsc4py.PETSc` module.
    np.random.seed(seed=seed)
    results = {}
    for n in ns:
        results[n]=[]
        A = get_random_matrix(n, SEED)
        x, b = A.createVecs()
        # x.view()
        # b.view()
        get_random_b(b, SEED)

        for param in params:
            # PETSc represents all linear solvers as preconditioned Krylov subspace methods
            # of type `PETSc.KSP`. Here we create a KSP object for a conjugate gradient
            # solver preconditioned with an algebraic multigrid method.
            ksp = PETSc.KSP()
            ksp.create(comm=A.getComm())

            # We set the matrix in our linear solver and allow the user to program the
            # solver with options.
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

            # ksp.logConvergenceHistory(ksp.getResidualNorm())
            start_time = time.time()
            ksp.solve(b, x)
            end_time = time.time()
            results[n].append({"KSP":ksp,"time":end_time-start_time})

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
    #)
    # - Show the solution with ``-view_sol``.
    # - Show the matrix with ``-view_mat``.
    # - Change the resolution with ``-n``.
    # - Use a direct solver by passing ``-ksp_type preonly -pc_type lu``.
    # - Run in parallel on two processors using:
    #
    #   .. code-block:: console
    #
    #       mpiexec -n 2 python poisson2d.py


#%%
N=[10,100,1000]#matrix size
SEED=10
PARAMS= [{"method":PETSc.KSP.Type.GMRES,"preconditioner":PETSc.PC.Type.ILU},
         {"method": PETSc.KSP.Type.BCGS, "preconditioner": PETSc.PC.Type.ILU},
                   ]
# This demo is structured as a script to be executed using:
#   $ python poisson2d.py
#
# potentially with additional options passed at the end of the command.
#
# At the start of your script, call `petsc4py.init` passing `sys.argv` so that
# command-line arguments to the script are passed through to PETSc.
petsc4py.init(sys.argv)

results=test_random_slae_solution(ns=N,seed=SEED,params= PARAMS)

# print(results)
for n in results:
    print("n=",n)
    for param in results[n]:
        # param["KSP"].getType().view()
        ksp=param["KSP"]
        print("method=",ksp.getType(),", preconditioner=", ksp.getPC().type,", time=", param["time"],
              " seconds, residual norm=",  ksp.getResidualNorm(),  ksp.its, ksp.history)
        # param["KSP"].getSolution().view()

# setConvergenceHistory
# setErrorIfNotConverged
# setInitialGuessKnoll