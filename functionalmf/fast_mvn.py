import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.linalg import solve_triangular
from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
from warnings import warn
from numpy.linalg import LinAlgError


def sample_mvn_from_precision(Q, mu=None, mu_part=None, sparse=True, chol_factor=False, Q_shape=None,
                              force_psd=False, min_force_psd=1e-3, psd_attempts=4):
    '''Fast sampling from a multivariate normal with precision parameterization.
    Supports sparse arrays. Params:
        - mu: If provided, assumes the model is N(mu, Q^-1)
        - mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
        - sparse: If true, assumes we are working with a sparse Q
        - chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the precision matrix
        - force_psd (bool): If true, attempts to force the precision matrix to
                            be positive definite adding a diagonal term.
        - min_force_psd (float): If force_psd is true, min_force_psd is the frist value added to the diagonal
                            to force the precision matrix to be positive definite.
        - psd_attempts (int): If force_psd is true, this is the number of attempts to force
                              the precision matrix to be positive definite. Each attempt a diagonal term that
                              is 10 times larger than the previous one is added.

    '''
    assert np.any([Q_shape is not None, not chol_factor, not sparse])

    attempt = 0
    eps = min_force_psd

    while True:
        try:
            if sparse:
                # Cholesky factor LL' = PQP' of the prior precision Q
                # where P is the permuation that reorders Q, the ordering of resulting L follows P
                factor = cholesky(Q) if not chol_factor else Q

                # Solve L'h = z ==> L'^-1 z = h, this is a sample from the prior.
                z = np.random.normal(size=Q.shape[0] if not chol_factor else Q_shape[0])
                
                # Reorder h by the permutation used in cholesky(Q).
                result = factor.solve_Lt(z, False)[np.argsort(factor.P())]
                if mu_part is not None:
                    # no need to reorder here since solve_A use the original Q
                    result += factor.solve_A(mu_part)
            else:
                # Q is the precision matrix. Q_inv would be the covariance.
                # We care about Q_inv, not Q. It turns out you can sample from a MVN
                # using the precision matrix by doing LL' = Cholesky(Precision)
                # then the covariance part of the draw is just inv(L')z where z is
                # a standard normal.
                # Ordering should be good here since linalg.cholesky solves LL'=Q
                Lt = np.linalg.cholesky(Q).T if not chol_factor else Q.T
                z = np.random.normal(size=Q.shape[0])
                result = solve_triangular(Lt, z, lower=False)
                if mu_part is not None:
                    result += sp.linalg.cho_solve((Lt, False), mu_part)
                elif mu is not None:
                    result += mu
        except (CholmodNotPositiveDefiniteError, LinAlgError):
            if force_psd and attempt < psd_attempts:
                Q = Q.copy()
                Q[np.diag_indices_from(Q)] += eps
                warn(f"Cholesky factorization failed, adding shrinkage {eps}.")
                attempt += 1
                eps *= 10
            else:
                warn(f'Cholesky factorization failed, try setting force_psd=True or increasing attempts')
                if attempt > psd_attempts:
                    raise Exception("Max attempts reached. Could not force matrix to be positive definite.")
        else:
            return result


def sample_mvn_from_covariance(Q, mu=None, mu_part=None, sparse=True, chol_factor=False,
                               force_psd=False, min_force_psd=1e-3, psd_attempts=4):
    '''Fast sampling from a multivariate normal with covariance parameterization.
    Supports sparse arrays. Params:
        - mu: If provided, assumes the model is N(mu, Q)
        - mu_part: If provided, assumes the model is N(Q mu_part, Q)
        - sparse: If true, assumes we are working with a sparse Q
        - chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the covariance matrix
        - force_psd (bool): If true, attempts to force the precision matrix to
                            be positive definite adding a diagonal term
        - min_force_psd (float): If force_psd is true, min_force_psd is the frist value added to the diagonal
                            to force the precision matrix to be positive definite.l
        - psd_attempts (int): If force_psd is true, this is the number of attempts to force
                              the precision matrix to be positive definite. Each attempt a diagonal term that
                              is 10 times larger than the previous one is added.
    '''

    attempt = 0
    eps = min_force_psd

    while True:
        try:
            if sparse:
                # Cholesky factor LL' = Q of the covariance matrix Q
                if chol_factor:
                    factor = Q
                    Q = factor.L().dot(factor.L().T)
                else:
                    factor = cholesky(Q)

                # Get the sample as mu + Lz for z ~ N(0, I)
                z = np.random.normal(size=Q.shape[0])
                result = factor.L().dot(z)[np.argsort(factor.P())]
                if mu_part is not None:
                    result += Q.dot(mu_part)
                elif mu is not None:
                    result += mu
            else:
                # Cholesky factor LL' = Q of the covariance matrix Q
                if chol_factor:
                    Lt = Q
                    Q = Lt.dot(Lt.T)
                else:
                    Lt = np.linalg.cholesky(Q)

                # Get the sample as mu + Lz for z ~ N(0, I)
                z = np.random.normal(size=Q.shape[0])
                result = Lt.dot(z)
                if mu_part is not None:
                    result += Q.dot(mu_part)
                elif mu is not None:
                    result += mu
        except (CholmodNotPositiveDefiniteError, LinAlgError):
            if force_psd and attempt < psd_attempts:
                Q = Q.copy()
                Q[np.diag_indices_from(Q)] += eps
                warn(f"Cholesky factorization failed, adding shrinkage {eps}.")
                attempt += 1
                eps *= 10
            else:
                warn(f'Cholesky factorization failed, try setting force_psd=True or increasing attempts')
                if attempt > psd_attempts:
                    raise Exception("Max attempts reached. Could not force matrix to be positive definite.")
        else:
            return result


def sample_mvn(Q, mu=None, mu_part=None, sparse=True, precision=False, chol_factor=False, Q_shape=None, **kwargs):
    '''Fast sampling from a multivariate normal with covariance or precision
    parameterization. Supports sparse arrays. Params:
        - mu: If provided, assumes the model is N(mu, Q)
        - mu_part: If provided, assumes the model is N(Q mu_part, Q)
        - sparse: If true, assumes we are working with a sparse Q
        - precision: If true, assumes Q is a precision matrix (inverse covariance)
        - chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the covariance matrix
                        (or of the precision matrix if precision=True).

    '''
    assert np.any((mu is None, mu_part is None)) # The mean and mean-part are mutually exclusive
    
    # If Q is a scalar or vector, consider it Q*I
    if not chol_factor:
        if np.isscalar(Q) or len(Q.shape) == 1:
            dim = len(mu) if mu is not None else len(mu_part)
            Q = np.eye(dim) * Q
            if sparse:
                Q = csc_matrix(Q)

    # Sample from the appropriate precision or covariance version
    if precision:
        return sample_mvn_from_precision(Q,
                                         mu=mu, mu_part=mu_part,
                                         sparse=sparse,
                                         chol_factor=chol_factor,
                                         Q_shape=Q_shape, 
                                         **kwargs)
    return sample_mvn_from_covariance(Q,
                                      mu=mu, mu_part=mu_part,
                                      sparse=sparse,
                                      chol_factor=chol_factor, 
                                      **kwargs)


####################### TESTS FOR MVN SAMPLERS ABOVE #######################
if __name__ == '__main__':
    print('Running MVN sampler tests.')
    Q = np.array([[1,0.4],[0.4,1]])
    Lt = np.linalg.cholesky(Q)
    Q_inv = np.linalg.inv(Q)
    Lt_inv = np.linalg.cholesky(Q_inv)
    sp_Q = csc_matrix(Q)
    sp_Lt = cholesky(sp_Q)
    sp_Q_inv = csc_matrix(Q_inv)
    sp_Lt_inv = cholesky(sp_Q_inv)

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axarr = plt.subplots(3,4, figsize=(20,10), sharex=True, sharey=True)

    # Covariance, dense, no factor
    X = np.array([sample_mvn(Q, sparse=False, chol_factor=False, precision=False) for _ in range(1000)])
    axarr[0,0].scatter(X[:,0], X[:,1])
    axarr[0,0].set_title('Covariance, dense, no factor')

    # Covariance, dense, with factor
    X = np.array([sample_mvn(Lt, sparse=False, chol_factor=True, precision=False) for _ in range(1000)])
    axarr[0,1].scatter(X[:,0], X[:,1])
    axarr[0,1].set_title('Covariance, dense, with factor')

    # Covariance, sparse, no factor
    X = np.array([sample_mvn(sp_Q, sparse=True, chol_factor=False, precision=False) for _ in range(1000)])
    axarr[0,2].scatter(X[:,0], X[:,1])
    axarr[0,2].set_title('Covariance, sparse, no factor')

    # Covariance, sparse, with factor
    X = np.array([sample_mvn(sp_Lt, sparse=True, chol_factor=True, precision=False) for _ in range(1000)])
    axarr[0,3].scatter(X[:,0], X[:,1])
    axarr[0,3].set_title('Covariance, sparse, with factor')

    # Precision, dense, no factor
    X = np.array([sample_mvn(Q_inv, sparse=False, chol_factor=False, precision=True) for _ in range(1000)])
    axarr[1,0].scatter(X[:,0], X[:,1])
    axarr[1,0].set_title('Precision, dense, no factor')

    # Precision, dense, with factor
    X = np.array([sample_mvn(Lt_inv, sparse=False, chol_factor=True, precision=True) for _ in range(1000)])
    axarr[1,1].scatter(X[:,0], X[:,1])
    axarr[1,1].set_title('Precision, dense, with factor')

    # Precision, sparse, no factor
    X = np.array([sample_mvn(sp_Q_inv, sparse=True, chol_factor=False, precision=True) for _ in range(1000)])
    axarr[1,2].scatter(X[:,0], X[:,1])
    axarr[1,2].set_title('Precision, sparse, no factor')

    # Precision, sparse, with factor
    X = np.array([sample_mvn(sp_Lt_inv, sparse=True, chol_factor=True, precision=True, Q_shape=(2,2)) for _ in range(1000)])
    axarr[1,3].scatter(X[:,0], X[:,1])
    axarr[1,3].set_title('Precision, sparse, with factor')

    # Vector covariance, dense
    X = np.array([sample_mvn(Q[0], mu=np.array([0.5,-0.5]), sparse=False, chol_factor=False, precision=False) for _ in range(1000)])
    axarr[2,0].scatter(X[:,0], X[:,1])
    axarr[2,0].set_title('Vector covariance, dense')

    # Scalar covariance, dense, no factor
    X = np.array([sample_mvn(Q[0,1], mu=np.array([0.5,-0.5]), sparse=False, chol_factor=False, precision=False) for _ in range(1000)])
    axarr[2,1].scatter(X[:,0], X[:,1])
    axarr[2,1].set_title('Scalar covariance, dense, no factor')
    xlim, ylim = axarr[2,1].get_xlim(), axarr[2,1].get_ylim()

    # Non invertible case Covariance
    Q = np.array([[1,1.0],[1.0,1]])
    X = np.array([sample_mvn(Q, sparse=False, chol_factor=False, precision=False, force_psd=True) for _ in range(1000)])
    axarr[2,2].scatter(X[:,0], X[:,1])
    axarr[2,2].set_title('Non invertible, dense covariance')
    axarr[2,2].set_xlim(xlim)
    axarr[2,2].set_ylim(ylim)

    # Non invertible case Precision
    sp_Q = csc_matrix(Q)
    X = np.array([sample_mvn(sp_Q, sparse=True, chol_factor=False, precision=True, force_psd=True) for _ in range(1000)])
    axarr[2,3].scatter(X[:,0], X[:,1])
    axarr[2,3].set_title('Non invertible, sparse precision')
    axarr[2,3].set_xlim(xlim)
    axarr[2,3].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig('plots/mvn-tests.pdf', bbox_inches='tight')
    plt.close()
