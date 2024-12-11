# See https://petsc.org/release/manualpages/KSP/KSPConvergedReason/
def get_convergence_reason(x):
    if x == 1:
        return 'KSP_CONVERGED_RTOL_NORMAL - requested decrease in the residual for the normal equations'
    elif x == 9:
        return 'KSP_CONVERGED_ATOL_NORMAL - requested absolute value in the residual for the normal equations'
    elif x == 2:
        return 'KSP_CONVERGED_RTOL - requested decrease in the residual'
    elif x == 3:
        return "KSP_CONVERGED_ATOL - requested absolute value in the residual"
    elif x == 4:
        return 'KSP_CONVERGED_ITS - requested number of iterations'
    elif x == 5:
        return 'KSP_CONVERGED_NEG_CURVE - The values KSP_CONVERGED_NEG_CURVE, and KSP_CONVERGED_STEP_LENGTH are returned only by KSPCG, KSPMINRES and by the special KSPNASH, KSPSTCG, and KSPGLTR solvers which are used by the SNESNEWTONTR (trust region) solver.'
    elif x == 6:
        return 'KSP_CONVERGED_CG_CONSTRAINED_DEPRECATED -The values KSP_CONVERGED_NEG_CURVE, and KSP_CONVERGED_STEP_LENGTH are returned only by KSPCG, KSPMINRES and by the special KSPNASH, KSPSTCG, and KSPGLTR solvers which are used by the SNESNEWTONTR (trust region) solver.'
    elif x == 7:
        return 'KSP_CONVERGED_HAPPY_BREAKDOWN - happy breakdown (meaning early convergence of the KSPType occurred).'
    elif x == -2:
        return 'KSP_DIVERGED_NULL - breakdown when solving the Hessenberg system within GMRES'
    elif x == -3:
        return 'KSP_DIVERGED_ITS - requested number of iterations'
    elif x == -4:
        return 'KSP_DIVERGED_DTOL - large increase in the residual norm'
    elif x == -5:
        return 'KSP_DIVERGED_BREAKDOWN - breakdown in the Krylov method'
    elif x == -6:
        return 'KSP_DIVERGED_BREAKDOWN_BICG - breakdown in the KSPBGCS Krylov method'
    elif x == -7:
        return 'KSP_DIVERGED_NONSYMMETRIC - the operator or preonditioner was not symmetric for a KSPType that requires symmetry'
    elif x == -8:
        return 'KSP_DIVERGED_INDEFINITE_PC - the preconditioner was indefinite for a KSPType that requires it be definite'
    elif x == -9:
        return 'KSP_DIVERGED_NANORINF - a not a number of infinity was detected in a vector during the computation'
    elif x == -10:
        return 'KSP_DIVERGED_INDEFINITE_MAT - the operator was indefinite for a KSPType that requires it be definite'
    elif x == -11:
        return 'KSP_DIVERGED_PC_FAILED - the action of the preconditioner failed for some reason'
    elif x == 0:
        return '  KSP_CONVERGED_ITERATING '
