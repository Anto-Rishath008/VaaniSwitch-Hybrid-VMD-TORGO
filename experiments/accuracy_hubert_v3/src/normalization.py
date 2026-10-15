"""Weighted train-only scaling, including an explicitly defined robust alternative."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .common import digest, require


def l2_rows(x):
    require(np.isfinite(x).all(), 'Non-finite normalization input')
    norms = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
    return np.asarray(x / np.where(norms > 1e-12, norms, 1.), dtype=np.float32)


def weighted_quantiles(x, weights, quantiles=(.25, .5, .75)):
    out = np.empty((len(quantiles), x.shape[1]), dtype=np.float64)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind='stable')
        w = weights[order]
        coordinates = (np.cumsum(w) - .5*w) / w.sum()
        out[:, j] = np.interp(quantiles, coordinates, x[order, j])
    return out


def fit_normalizer(partition, weights, method):
    partition.check(fitting=True)
    require(method in ('N0', 'N1', 'N2'), 'Unknown normalization')
    require(len(weights) == len(partition.records) and np.isfinite(weights).all()
            and (weights > 0).all(), 'Invalid scaler weights')
    x = l2_rows(partition.features) if method == 'N2' else partition.features
    spec = {'method': method, 'fit_identity': partition.identity,
            'weight_sha256': digest(weights.tolist()), 'fit_on_originals_only': True}
    if method in ('N0', 'N2'):
        spec['scaler'] = StandardScaler().fit(x, sample_weight=weights)
        require(np.isfinite(spec['scaler'].mean_).all() and np.isfinite(spec['scaler'].scale_).all(),
                'Non-finite StandardScaler state')
    else:
        q25, median, q75 = weighted_quantiles(x, weights)
        scale = q75-q25
        fallback = scale <= 1e-12
        spec.update(center=median, scale=np.where(fallback, 1., scale),
                    zero_iqr_dimensions=int(fallback.sum()))
    return spec


def transform(x, spec):
    require(x.ndim == 2 and np.isfinite(x).all(), 'Non-finite/invalid raw features')
    values = l2_rows(x) if spec['method'] == 'N2' else x
    if spec['method'] in ('N0', 'N2'):
        s = spec['scaler']
        require(np.isfinite(s.mean_).all() and np.isfinite(s.scale_).all() and (s.scale_ > 0).all(),
                'Invalid scaler state')
        result = s.transform(values)  # preserve V2 float32 sklearn transform semantics exactly
    else:
        require(np.isfinite(spec['center']).all() and np.isfinite(spec['scale']).all()
                and (spec['scale'] > 0).all(), 'Invalid robust scaler state')
        result = np.asarray((values-spec['center'])/spec['scale'], dtype=np.float32)
    require(np.isfinite(result).all(), 'Non-finite standardized features')
    return result
