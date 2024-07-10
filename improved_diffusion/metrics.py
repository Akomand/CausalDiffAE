import numpy as np
import scipy
from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
import torch
from munkres import Munkres
from scipy.optimize import linear_sum_assignment


import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import numpy as np




def generate_batch_factor_code(ground_truth_data, representation_function, num_points, random_state, batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    num_points: Number of points to sample.
    random_state: Numpy random state used for randomness.
    batch_size: Batchsize to sample points.

    Returns:
    representations: Codes (num_codes, num_points)-np array.
    factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = \
            ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations,
                                       representation_function(
                                           current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


# def make_discretizer(target, num_bins=gin.REQUIRED, discretizer_fn=gin.REQUIRED):

#     return discretizer_fn(target, num_bins)



def compute_irs(rep, y, diff_quantile=0.99):
    """Computes the Interventional Robustness Score.

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

    Returns:
    Dict with IRS and number of active dimensions.
    """

    # mus, ys = generate_batch_factor_code(ground_truth_data,
    #                                          representation_function, num_train,
    #                                          random_state, batch_size)
    # assert mus.shape[1] == num_train


    if not rep.any():
        irs_score = 0.0
    else:
        irs_score = scalable_disentanglement_score(y.T, rep.T, diff_quantile)["avg_score"]

    score_dict = {}
    score_dict["IRS"] = irs_score
    score_dict["num_active_dims"] = np.sum(rep)
    
    return score_dict


def _drop_constant_dims(ys):
    """Returns a view of the matrix `ys` with dropped constant rows."""
    ys = np.asarray(ys)
    if ys.ndim != 2:
        raise ValueError("Expecting a matrix.")

    variances = ys.var(axis=1)
    active_mask = variances > 0.
    
    return ys[active_mask, :]


def scalable_disentanglement_score(gen_factors, latents, diff_quantile=0.99):
    """Computes IRS scores of a dataset.

    Assumes no noise in X and crossed generative factors (i.e. one sample per
    combination of gen_factors). Assumes each g_i is an equally probable
    realization of g_i and all g_i are independent.

    Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

    Returns:
    Dictionary with IRS scores.
    """
    num_gen = gen_factors.shape[1]
    num_lat = latents.shape[1]

    # Compute normalizer.
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
    cum_deviations = np.zeros([num_lat, num_gen])
    for i in range(num_gen):
        unique_factors = np.unique(gen_factors[:, i], axis=0)
        assert unique_factors.ndim == 1
        num_distinct_factors = unique_factors.shape[0]
        for k in range(num_distinct_factors):
            # Compute E[Z | g_i].
            match = gen_factors[:, i] == unique_factors[k]
            e_loc = np.mean(latents[match, :], axis=0)

            # Difference of each value within that group of constant g_i to its mean.
            diffs = np.abs(latents[match, :] - e_loc)
            max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
            cum_deviations[:, i] += max_diffs
        cum_deviations[:, i] /= num_distinct_factors
    # Normalize value of each latent dimension with its maximal deviation.
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations
    disentanglement_scores = irs_matrix.max(axis=1)
    if np.sum(max_deviations) > 0.0:
        avg_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        avg_score = np.mean(disentanglement_scores)

    parents = irs_matrix.argmax(axis=1)
    score_dict = {}
    score_dict["disentanglement_scores"] = disentanglement_scores
    score_dict["avg_score"] = avg_score
    score_dict["parents"] = parents
    score_dict["IRS_matrix"] = irs_matrix
    score_dict["max_deviations"] = max_deviations
    
    return score_dict


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    disent, code_importance = disentanglement(importance_matrix)
    scores["disentanglement"] = disent
    scores["completeness"] = completeness(importance_matrix)
    return scores, importance_matrix, code_importance


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                 dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        # from xgboost import XGBClassifier
        # model = XGBClassifier()
        # model = ensemble.GradientBoostingClassifier()
        model = ensemble.GradientBoostingRegressor()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    
  return np.sum(per_code*code_importance), code_importance


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)


def MCC(Z, Zp):
    n = np.shape(Z)[1]
    #     print (n)
    rho_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho_matrix[i, j] = np.abs(np.corrcoef(Z[:, i], Zp[:, j])[0, 1])

    r, c = linear_sum_assignment(-rho_matrix)

    return np.mean(rho_matrix[r, c])


def r2_disentanglement(z, hz, mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "adjusted_r2", "pearson", "spearman")

    if mode == "r2":
        # print(z[0].shape)
        # print(hz[0].shape)
        # exit(0)
        r2_i = []
        for i in range(z.shape[0]):
            r2_i.append(metrics.r2_score(z[i], hz[i]))
            print(metrics.r2_score(z[i], hz[i]))

        return sum(r2_i) / len(r2_i)
    elif mode == "adjusted_r2":
        r2 = metrics.r2_score(z, hz)
        # number of data samples
        n = z.shape[0]
        # number of predictors, i.e. features
        p = z.shape[1]
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2, None
    elif mode in ("spearman", "pearson"):
        dim = z.shape[-1]

        if mode == "spearman":
            raw_corr, pvalue = scipy.stats.spearmanr(z, hz)
        else:
            raw_corr = np.corrcoef(z.T, hz.T)
        corr = raw_corr[:dim, dim:]

        if reorder:
            # effectively computes MCC
            munk = Munkres()
            indexes = munk.compute(-np.absolute(corr))

            sort_idx = np.zeros(dim)
            hz_sort = np.zeros(z.shape)
            for i in range(dim):
                sort_idx[i] = indexes[i][1]
                hz_sort[:, i] = hz[:, indexes[i][1]]

            if mode == "spearman":
                raw_corr, pvalue = scipy.stats.spearmanr(z, hz_sort)
            else:
                raw_corr = np.corrcoef(z.T, hz_sort.T)

            corr = raw_corr[:dim, dim:]

        return np.diag(np.abs(corr)).mean(), corr

    
    
def linear_disentanglement(z, hz, mode="r2", train_test_split=None):
    """Calculate disentanglement up to linear transformations.
    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        train_test_split: Use first half to train linear model, second half to test.
            Is only relevant if there are less samples then latent dimensions.
    """

    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    # assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    # assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    # split z, hz to get train and test set for linear model
    if train_test_split:
        n_train = len(z) // 2
        z_1 = z[:n_train]
        hz_1 = hz[:n_train]
        z_2 = z[n_train:]
        hz_2 = hz[n_train:]
    else:
        z_1 = z
        hz_1 = hz
        z_2 = z
        hz_2 = hz

    model = linear_model.LinearRegression()
    model.fit(hz_1, z_1)

    hz_2 = model.predict(hz_2)

    inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    return inner_result, (z_2, hz_2)


def _disentanglement(z, hz, mode="r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    # assert mode in ("r2", "adjusted_r2", "pearson", "spearman")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "adjusted_r2":
        r2 = metrics.r2_score(z, hz)
        # number of data samples
        n = z.shape[0]
        # number of predictors, i.e. features
        p = z.shape[1]
        adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2, None
    elif mode in ("spearman", "pearson"):
        dim = z.shape[-1]

        if mode == "spearman":
            raw_corr, pvalue = sp.stats.spearmanr(z, hz)
        else:
            raw_corr = np.corrcoef(z.T, hz.T)
        corr = raw_corr[:dim, dim:]

        if reorder:
            # effectively computes MCC
            munk = Munkres()
            indexes = munk.compute(-np.absolute(corr))

            sort_idx = np.zeros(dim)
            hz_sort = np.zeros(z.shape)
            for i in range(dim):
                sort_idx[i] = indexes[i][1]
                hz_sort[:, i] = hz[:, indexes[i][1]]

            if mode == "spearman":
                raw_corr, pvalue = sp.stats.spearmanr(z, hz_sort)
            else:
                raw_corr = np.corrcoef(z.T, hz_sort.T)

            corr = raw_corr[:dim, dim:]

        return np.diag(np.abs(corr)).mean(), corr
  
    
def permutation_disentanglement(
    z,
    hz,
    mode="r2",
    rescaling=True,
    solver="naive",
    sign_flips=True,
    cache_permutations=None,
):
    """Measure disentanglement up to permutations by either using the Munkres solver
    or naively trying out every possible permutation.
    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        rescaling: Rescale every individual latent to maximize the agreement
            with the ground-truth.
        solver: How to find best possible permutation. Either use Munkres algorithm
            or naively test every possible permutation.
        sign_flips: Only relevant for `naive` solver. Also include sign-flips in
            set of possible permutations to test.
        cache_permutations: Only relevant for `naive` solver. Cache permutation matrices
            to allow faster access if called multiple times.
    """

    assert solver in ("naive", "munkres")
    if mode == "r2" or mode == "adjusted_r2":
        assert solver == "naive", "R2 coefficient is only supported with naive solver"

    if cache_permutations and not hasattr(
        permutation_disentanglement, "permutation_matrices"
    ):
        permutation_disentanglement.permutation_matrices = dict()

    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    def test_transformation(T, reorder):
        # measure the r2 score for one transformation

        Thz = hz @ T
        if rescaling:
            assert z.shape == hz.shape
            # find beta_j that solve Y_ij = X_ij beta_j
            Y = z
            X = hz

            beta = np.diag((Y * X).sum(0) / (X ** 2).sum(0))

            Thz = X @ beta

        return _disentanglement(z, Thz, mode=mode, reorder=reorder), Thz

    def gen_permutations(n):
        # generate all possible permutations w/ or w/o sign flips

        def gen_permutation_single_row(basis, row, sign_flips=False):
            # generate all possible permutations w/ or w/o sign flips for one row
            # assuming the previous rows are already fixed
            basis = basis.clone()
            basis[row] = 0
            for i in range(basis.shape[-1]):
                # skip possible columns if there is already an entry in one of
                # the previous rows
                if torch.sum(torch.abs(basis[:row, i])) > 0:
                    continue
                signs = [1]
                if sign_flips:
                    signs += [-1]

                for sign in signs:
                    T = basis.clone()
                    T[row, i] = sign

                    yield T

        def gen_permutations_all_rows(basis, current_row=0, sign_flips=False):
            # get all possible permutations for all rows

            for T in gen_permutation_single_row(basis, current_row, sign_flips):
                if current_row == len(basis) - 1:
                    yield T.numpy()
                else:
                    # generate all possible permutations of all other rows
                    yield from gen_permutations_all_rows(T, current_row + 1, sign_flips)

        basis = torch.zeros((n, n))

        yield from gen_permutations_all_rows(basis, sign_flips=sign_flips)

    n = z.shape[-1]
    # use cache to speed up repeated calls to the function
    if cache_permutations and not solver == "munkres":
        key = (rescaling, n)
        if not key in permutation_disentanglement.permutation_matrices:
            permutation_disentanglement.permutation_matrices[key] = list(
                gen_permutations(n)
            )
        permutations = permutation_disentanglement.permutation_matrices[key]
    else:
        if solver == "naive":
            permutations = list(gen_permutations(n))
        elif solver == "munkres":
            permutations = [np.eye(n, dtype=z.dtype)]

    scores = []

    # go through all possible permutations and check r2 score
    for T in permutations:
        scores.append(test_transformation(T, solver == "munkres"))

    return max(scores, key=lambda x: x[0][0])