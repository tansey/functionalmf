import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.linalg import solve_triangular
from sksparse.cholmod import cholesky

def sample_mvn_from_precision(Q, mu=None, mu_part=None, sparse=True, chol_factor=False, Q_shape=None):
    '''Fast sampling from a multivariate normal with precision parameterization.
    Supports sparse arrays. Params:
        - mu: If provided, assumes the model is N(mu, Q^-1)
        - mu_part: If provided, assumes the model is N(Q^-1 mu_part, Q^-1)
        - sparse: If true, assumes we are working with a sparse Q
        - chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the precision matrix
    '''
    assert np.any([Q_shape is not None, not chol_factor, not sparse])
    if sparse:
        # Cholesky factor LL' = Q of the prior precision Q
        factor = cholesky(Q) if not chol_factor else Q

        # Solve L'h = z ==> L'^-1 z = h, this is a sample from the prior.
        z = np.random.normal(size=Q.shape[0] if not chol_factor else Q_shape[0])
        result = factor.solve_Lt(z, False)
        if mu_part is not None:
            result += factor.solve_A(mu_part)
        return result

    # Q is the precision matrix. Q_inv would be the covariance.
    # We care about Q_inv, not Q. It turns out you can sample from a MVN
    # using the precision matrix by doing LL' = Cholesky(Precision)
    # then the covariance part of the draw is just inv(L')z where z is
    # a standard normal.
    Lt = np.linalg.cholesky(Q).T if not chol_factor else Q.T
    z = np.random.normal(size=Q.shape[0])
    result = solve_triangular(Lt, z, lower=False)
    if mu_part is not None:
        result += sp.linalg.cho_solve((Lt, False), mu_part)
    elif mu is not None:
        result += mu
    return result

def sample_mvn_from_covariance(Q, mu=None, mu_part=None, sparse=True, chol_factor=False):
    '''Fast sampling from a multivariate normal with covariance parameterization.
    Supports sparse arrays. Params:
        - mu: If provided, assumes the model is N(mu, Q)
        - mu_part: If provided, assumes the model is N(Q mu_part, Q)
        - sparse: If true, assumes we are working with a sparse Q
        - chol_factor: If true, assumes Q is a (lower triangular) Cholesky
                        decomposition of the covariance matrix
    '''
    if sparse:
        # Cholesky factor LL' = Q of the covariance matrix Q
        if chol_factor:
            factor = Q
            Q = factor.L().dot(factor.L().T)
        else:
            factor = cholesky(Q)

        # Get the sample as mu + Lz for z ~ N(0, I)
        z = np.random.normal(size=Q.shape[0])
        result = factor.L().dot(z)
        if mu_part is not None:
            result += Q.dot(mu_part)
        elif mu is not None:
            result += mu
        return result

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
    return result


def sample_mvn(Q, mu=None, mu_part=None, sparse=True, precision=False, chol_factor=False, Q_shape=None):
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
                                         Q_shape=Q_shape)
    return sample_mvn_from_covariance(Q,
                                      mu=mu, mu_part=mu_part,
                                      sparse=sparse,
                                      chol_factor=chol_factor)


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

    plt.tight_layout()
    plt.savefig('plots/mvn-tests.pdf', bbox_inches='tight')
    plt.close()


