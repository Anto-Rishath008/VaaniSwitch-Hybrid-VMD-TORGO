"""Finite, resumable logistic fits. No neural classifier is justified or implemented."""
import hashlib
import time
import warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .common import digest, publish, read, require, within_v3, write_json
from .data import separated
from .weighting import weights, expanded_weights, weight_report
from .normalization import fit_normalizer, transform
from experiments.accuracy_hubert_v2.src.training import (
    rng_state, restore_rng, seed_all, save_checkpoint, load_checkpoint, binary_loss)


def metrics(labels, scores):
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    y, p = np.asarray(labels, int), np.asarray(scores, float)
    require(y.ndim == p.ndim == 1 and len(y) == len(p) > 0 and set(y) <= {0, 1}
            and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), 'Invalid metric arrays')
    predicted = p >= .5
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    sensitivity = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    recalls = [v for v in (sensitivity, specificity) if v is not None]
    bins, ece = [], 0.
    indices = np.minimum((p*10).astype(int), 9)
    for b in range(10):
        mask = indices == b
        if mask.any():
            average, observed = float(p[mask].mean()), float(y[mask].mean())
            bins.append({'bin': b, 'count': int(mask.sum()), 'mean_probability': average,
                         'positive_fraction': observed})
            ece += float(mask.mean())*abs(average-observed)
    return {'accuracy': float((predicted == y).mean()), 'balanced_accuracy': float(np.mean(recalls)),
            'macro_f1': float(f1_score(y, predicted, labels=[0, 1], average='macro', zero_division=0)),
            'precision': tp/(tp+fp) if tp+fp else 0., 'sensitivity': sensitivity, 'specificity': specificity,
            'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'confusion_matrix': [[tn, fp], [fn, tp]], 'record_count': len(y), 'threshold': .5,
            'brier': float(np.mean((p-y)**2)), 'log_loss': binary_loss(y, p),
            'ece_10_equal_width': ece, 'reliability_bins': bins}


def prediction_rows(records, probabilities, evidence_kind='REAL_DEVELOPMENT'):
    return [{'logical_id': r.logical_id, 'speaker_id': r.speaker_id, 'label': r.label,
             'probability': float(p), 'prediction': int(p >= .5), 'microphone': r.primary_microphone,
             'evidence_kind': evidence_kind}
            for r, p in zip(records, probabilities, strict=True)]


def predict(payload, features):
    model = payload['model']
    require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), 'Non-finite LR parameters')
    values = transform(features, payload['normalizer'])
    logits = model.decision_function(values)
    require(np.isfinite(logits).all(), 'Non-finite LR logits')
    probabilities = model.predict_proba(values)[:, 1]
    require(np.isfinite(probabilities).all(), 'Non-finite probabilities')
    return probabilities


def feature_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def fit(train, validation, candidate, cfg, root, run_identity, augmented=None, interrupt=None):
    train.check(fitting=True)
    if validation is not None:
        validation.check()
        require(validation.role in {'validation', 'assessment'}, 'Unexpected selection role')
        separated(train.records, validation.records)
    require(candidate['weighting'] in {'B0', 'B2'} and candidate['normalization'] in {'N0', 'N1', 'N2'},
            'Unknown or duplicated candidate policy')
    require(bool(candidate['augmentation']) == (augmented is not None), 'Augmentation candidate/input mismatch')
    root = within_v3(root)
    base_weights = weights(train, candidate['weighting'])
    parent_ids = [r.logical_id for r in train.records]
    features, records = train.features, list(train.records)
    if augmented is not None:
        aug_records, aug_features = augmented
        originals = {r.logical_id: r for r in train.records}
        require(len(aug_records) == len(aug_features) > 0 and len({r.logical_id for r in aug_records}) == len(aug_records),
                'Invalid variant count')
        require(all(originals.get(r.logical_id) == r for r in aug_records), 'Augmented recording leakage/label mismatch')
        require(np.isfinite(aug_features).all(), 'Non-finite augmented input')
        records += list(aug_records)
        features = np.concatenate([features, aug_features])
        parent_ids += [r.logical_id for r in aug_records]
    objective_weights = expanded_weights(train, base_weights, parent_ids)
    identity = digest({'run': run_identity, 'candidate': candidate, 'train': train.identity,
                       'train_features': feature_hash(features), 'parents': parent_ids,
                       'validation': validation.identity if validation is not None else None,
                       'validation_features': feature_hash(validation.features) if validation is not None else None,
                       'max_iter': cfg['logistic_max_iter']})
    complete = root/'complete.json'
    if complete.exists():
        result = read(complete)
        payload = load_checkpoint(result['checkpoint'], identity)
        require({k: v for k, v in result.items() if k != 'checkpoint'} == payload['result'],
                'Completed fit metrics/metadata differ from checkpoint')
        predict(payload, train.features[:1])
        return result
    # If only publication was interrupted, use the finished checkpoint without re-fitting.
    latest = root/'latest.json'
    if latest.exists():
        reference = read(latest)
        payload = load_checkpoint(reference, identity)
        if payload['phase'] == 'FIT_COMPLETE':
            result = {**payload['result'], 'checkpoint': reference}
            predict(payload, train.features[:1])
            publish(complete, result)
            return result
        require(payload['phase'] == 'START', 'Unknown recovery phase')
        restore_rng(payload['rng'])
        write_json(root/('restart-'+str(time.time_ns())+'.json'),
                   {'policy': 'Interrupted LBFGS restarts from deterministic beginning', 'identity': identity})
    else:
        seed_all()
        save_checkpoint(root, {'identity': identity, 'epoch': -1, 'phase': 'START', 'rng': rng_state()})
    if interrupt == 'after_start':
        raise InterruptedError('Deliberate bounded validation interruption after START checkpoint')
    started = time.perf_counter()
    normalizer = fit_normalizer(train, base_weights, candidate['normalization'])
    x = transform(features, normalizer)
    labels = np.array([r.label for r in records], dtype=np.float32)
    require(set(labels) == {0, 1}, 'Training requires both classes')
    model = LogisticRegression(C=candidate['C'], penalty='l2', solver='lbfgs', class_weight=None,
                               max_iter=cfg['logistic_max_iter'], random_state=42)
    with warnings.catch_warnings(), threadpool_limits(limits=cfg['cpu_threads']):
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, labels, sample_weight=objective_weights)
    payload = {'identity': identity, 'epoch': 0, 'phase': 'FIT_COMPLETE', 'model': model,
               'normalizer': normalizer, 'candidate': candidate, 'rng': rng_state()}
    probabilities = predict(payload, features)
    residual = objective_weights*(probabilities-labels)
    gradient = x.astype(np.float64).T @ residual + model.coef_[0]/candidate['C']
    intercept_gradient = float(residual.sum())
    require(np.isfinite(gradient).all() and np.isfinite(intercept_gradient)
            and np.isfinite(binary_loss(labels, probabilities)), 'Non-finite objective/gradient')
    result = {'candidate': candidate, 'identity': identity, 'parameter_count': x.shape[1]+1,
              'evidence_kind': train.evidence_kind,
              'training_seconds': time.perf_counter()-started, 'iterations': model.n_iter_.tolist(),
              'normalizer_fit': normalizer['fit_identity'], 'original_weights': weight_report(train.records, base_weights),
              'effective_weights': weight_report(records, objective_weights), 'original_count': len(train.records),
              'augmented_count': len(records)-len(train.records), 'finite_state': 'PASS',
              'normalized_gradient_norm': float(np.sqrt(np.sum(gradient**2)+intercept_gradient**2)/objective_weights.sum()),
              'rng_checkpointed': True, 'iterative_loss_history': None}
    if validation is not None:
        p = predict(payload, validation.features)
        result['validation_metrics'] = metrics(validation.labels, p)
        result['predictions'] = prediction_rows(validation.records, p, validation.evidence_kind)
    payload['result'] = result
    reference = save_checkpoint(root, payload)
    if interrupt == 'after_checkpoint':
        raise InterruptedError('Deliberate interruption after completed model checkpoint')
    result = {**result, 'checkpoint': reference}
    publish(complete, result)
    return result


def selection_key(result):
    m = result['metrics']
    return (-m['macro_f1'], -m['balanced_accuracy'], -(m['roc_auc'] if m['roc_auc'] is not None else -1),
            m['log_loss'], result['candidate']['complexity'], result['candidate']['id'])
