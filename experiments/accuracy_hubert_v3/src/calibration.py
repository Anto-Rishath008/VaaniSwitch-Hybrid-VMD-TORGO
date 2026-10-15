"""Optional, separate fixed-recipe calibration diagnostic with three disjoint roles."""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .common import digest, publish, require
from .data import partition, separated
from .selection import candidate
from .training import fit, load_checkpoint, predict, prediction_rows, metrics
from .weighting import weights


def calibration_roles(corpus, split):
    labels = {r.speaker_id: r.label for r in corpus.select(split.development)}
    calibration, assessment = [], []
    for label in (0, 1):
        ordered = sorted((s for s in split.development if labels[s] == label),
                         key=lambda s: digest(['v3-calibration', 42, split.fold, s]))
        calibration.append(ordered[0])
        assessment.append(ordered[1])
    training = sorted(set(split.development)-set(calibration)-set(assessment))
    separated(*(corpus.select(s) for s in (training, calibration, assessment, split.test)))
    return tuple(training), tuple(sorted(calibration)), tuple(sorted(assessment))


def diagnostic(corpus, split, cache, cfg, output, identity):
    train_s, cal_s, assess_s = calibration_roles(corpus, split)
    train = partition(corpus, train_s, split, 'train', cache, 9)
    cal = partition(corpus, cal_s, split, 'calibration', cache, 9)
    assess = partition(corpus, assess_s, split, 'assessment', cache, 9)
    result = fit(train, None, candidate(9, .1), cfg, output/'base_model', identity)
    payload = load_checkpoint(result['checkpoint'], result['identity'])
    p_cal, p_assess = predict(payload, cal.features), predict(payload, assess.features)
    def logits(p):
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p/(1-p))[:, None]
    # Equal speaker mass for calibration, fitted only on its own separated predictions.
    counts = {s: sum(r.speaker_id == s for r in cal.records) for s in cal_s}
    w = np.array([1/counts[r.speaker_id] for r in cal.records])
    w /= w.mean()
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=3000, random_state=42)
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(logits(p_cal), cal.labels, sample_weight=w)
    require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), 'Calibration numerical corruption')
    calibrated = model.predict_proba(logits(p_assess))[:, 1]
    report = {'label': 'Development-only fixed-recipe calibration diagnostic; excluded from selection and outer models',
              'fixed_recipe': candidate(9, .1), 'base_train': train_s, 'calibration': cal_s, 'assessment': assess_s,
              'calibrator': {'C': 1., 'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist()},
              'before': metrics(assess.labels, p_assess), 'after': metrics(assess.labels, calibrated),
              'uncalibrated_predictions': prediction_rows(assess.records, p_assess, assess.evidence_kind),
              'calibrated_predictions': prediction_rows(assess.records, calibrated, assess.evidence_kind), 'threshold': .5,
              'limitation': 'Two assessment speakers; exploratory, no calibration benefit presumed'}
    import json
    report = json.loads(json.dumps(report))
    publish(output/'diagnostic.json', report)
    return report
