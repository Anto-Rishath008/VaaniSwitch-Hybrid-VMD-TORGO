"""Speaker-weighted LR and resumable FP32 upper-layer fine-tuning."""
import copy
import math
import time
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .common import checkpoint, digest, event, load_checkpoint, publish, read, require, sha
from .data import audio, separated
from .model import Classifier, inputs, load_backbone
from .numerics import configure_numerics, finite, state_check, step
from experiments.accuracy_hubert_v2.src.training import rng_state, restore_rng
from experiments.accuracy_hubert_v3.src.training import metrics, prediction_rows


def validate_roles(train, validation):
    train.check(fitting=True)
    require(set(train.labels) == {0, 1}, 'Training requires both classes')
    if validation is not None:
        validation.check()
        require(validation.role == 'validation' and train.context == validation.context
            and train.forbidden == validation.forbidden, 'Validation role/context mismatch')
        separated(train.records, validation.records)
        require(set(validation.labels) == {0, 1}, 'Validation requires both classes')


def predict_lr(payload, x):
    finite(x, 'LR_features')
    z = finite(payload['scaler'].transform(x), 'LR_scaled_features')
    model = payload['model']
    finite([model.coef_, model.intercept_], 'LR_parameters')
    finite(model.decision_function(z), 'LR_logits')
    return finite(model.predict_proba(z)[:, 1], 'LR_probabilities')


def fit_lr(train, validation, x, vx, candidate, cfg, root, run_id, interrupt=None):
    validate_roles(train, validation)
    require(len(x) == len(train.records) and x.ndim == 2, 'Training features shape')
    if validation is not None:
        require(vx.ndim == 2 and len(vx) == len(validation.records) and x.shape[1] == vx.shape[1], 'Validation feature shape')
    finite(x, 'training_features')
    if vx is not None:
        finite(vx, 'validation_features')
    identity = digest({'run': run_id, 'candidate': candidate, 'train': train.identity,
        'validation': validation.identity if validation else None,
        'x': digest(x.tolist()), 'vx': digest(vx.tolist()) if vx is not None else None, 'cfg': cfg})
    if (root/'complete.json').exists():
        result = read(root/'complete.json')
        payload = load_checkpoint(result['checkpoint'], identity)
        require(result['result_digest'] == digest(payload['result']), 'LR seal metadata differs')
        return result
    if (root/'latest.json').exists():
        payload = load_checkpoint(read(root/'latest.json'), identity)
        if payload['phase'] == 'COMPLETE':
            result = {**payload['result'], 'result_digest': digest(payload['result']), 'checkpoint': read(root/'latest.json')}
            publish(root/'complete.json', result)
            return result
        restore_rng(payload['rng'])
        event(root, 'LBFGS_RESTART_FROM_BEGINNING', identity=identity)
    else:
        configure_numerics(cfg['seed'])
        checkpoint(root, {'identity': identity, 'phase': 'START', 'rng': rng_state()})
    if interrupt == 'after_start':
        raise InterruptedError('Controlled LR interruption')
    started = time.perf_counter()
    weights = train.weights()
    scaler = StandardScaler().fit(x, sample_weight=weights)
    z = finite(scaler.transform(x), 'training_normalization')
    finite([scaler.mean_, scaler.scale_], 'normalizer_parameters')
    model = LogisticRegression(C=candidate['C'], solver='lbfgs', max_iter=cfg['logistic_max_iter'],
        class_weight=None, random_state=cfg['seed'])
    with threadpool_limits(2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(z, train.labels, sample_weight=weights)
    payload = {'identity': identity, 'phase': 'COMPLETE', 'model': model, 'scaler': scaler,
        'normalizer_fit_speakers': list(train.allowed), 'normalizer_record_ids': [r.logical_id for r in train.records]}
    p = predict_lr(payload, x)
    logits = model.decision_function(z)
    loss = np.mean(weights*(np.logaddexp(0, logits)-train.labels*logits))
    residual = weights*(p-train.labels)/len(p)
    finite([loss, z.T@residual+model.coef_[0]/(candidate['C']*weights.sum()), residual.sum()], 'LR_loss_gradients')
    rows = prediction_rows(validation.records, predict_lr(payload, vx)) if validation else []
    result = {'identity': identity, 'candidate': candidate, 'training_seconds': time.perf_counter()-started,
        'validation_metrics': metrics([r['label'] for r in rows], [r['probability'] for r in rows]) if rows else None,
        'validation_predictions': rows, 'parameter_count': int(model.coef_.size+model.intercept_.size),
        'normalizer_fit_speakers': list(train.allowed), 'best_epoch': None,
        'weight_class_mass': {str(y): float(weights[train.labels == y].sum()) for y in (0, 1)},
        'speaker_mass': {s: float(sum(w for r, w in zip(train.records, weights) if r.speaker_id == s)) for s in train.allowed}}
    payload.update(result=result, rng=rng_state())
    reference = checkpoint(root, payload)
    if interrupt == 'after_checkpoint':
        raise InterruptedError('Controlled LR publication interruption')
    result = {**result, 'result_digest': digest(result), 'checkpoint': reference}
    publish(root/'complete.json', result)
    return result


def delta_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: p.detach().cpu().clone() for n, p in model.state_dict().items() if n in names}


def restore_delta(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    require(set(state) == expected, 'Trainable checkpoint key mismatch')
    finite(state, 'checkpoint_parameters')
    model.load_state_dict(state, strict=False)


def memory():
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
        'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved()}


def build(directory, candidate, cfg):
    backbone, processor, info = load_backbone(directory)
    model = Classifier(backbone, candidate['depth'], candidate['pooling'], cfg['gradient_checkpointing']).cuda()
    return model, processor, info


def optimizer_for(model, cfg, total_steps):
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': cfg['backbone_lr']},
        {'params': head, 'lr': cfg['head_lr']}], weight_decay=cfg['weight_decay'])
    # Linear decay, with schedule bound to planned maximum training updates, including on resume.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: max(0., 1-n/max(1, total_steps)))
    return optimizer, scheduler


def probabilities(model, processor, records, seconds):
    model.eval()
    values = []
    with torch.no_grad():
        for record in records:
            waveform, _ = audio(record, seconds)
            x, mask = inputs(processor, waveform, 'cuda')
            values.append(float(torch.sigmoid(model(x, mask))[0]))
    return finite(np.asarray(values), 'evaluation_probabilities')


def finite_training_group(model, processor, records, labels, weights, cfg, optimizer, scheduler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for record, label, weight in zip(records, labels, weights, strict=True):
        waveform, _ = audio(record, cfg['fine_tune_seconds'])
        x, mask = inputs(processor, waveform, 'cuda')
        logits = model(x, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([label], device='cuda'))
        finite(loss, 'binary_loss')
        (loss*float(weight)/len(records)).backward()
        losses.append(float(loss.detach())*float(weight))
    norm = step(model, optimizer, cfg['gradient_clip'])
    scheduler.step()
    finite(scheduler.state_dict(), 'scheduler')
    return float(np.mean(losses)), norm


def fit_neural(train, validation, candidate, cfg, directory, root, run_id, epochs=None,
               interrupt_after_step=None, fault=None):
    """Resume only at complete accumulation updates; replay unsaved work from serialized state."""
    validate_roles(train, validation)
    planned_epochs = int(epochs or cfg['max_epochs'])
    require(1 <= planned_epochs <= cfg['max_epochs'], 'Epoch budget')
    identity = digest({'run': run_id, 'candidate': candidate, 'config': cfg, 'epochs': planned_epochs,
        'train': train.identity, 'validation': validation.identity if validation else None})
    complete = root/'complete.json'
    if complete.exists():
        result = read(complete)
        payload = load_checkpoint(result['checkpoint'], identity)
        require(payload['phase'] == 'COMPLETE' and payload['result'] == {k: v for k, v in result.items() if k != 'checkpoint'},
            'Completed neural seal differs')
        return result
    configure_numerics(cfg['seed'])
    model, processor, loading = build(directory, candidate, cfg)
    torch.cuda.reset_peak_memory_stats()
    groups_per_epoch = math.ceil(len(train.records)/cfg['accumulation_steps'])
    optimizer, scheduler = optimizer_for(model, cfg, planned_epochs*groups_per_epoch)
    weights, labels = train.weights(), train.labels
    state = {'identity': identity, 'phase': 'TRAINING', 'epoch': 0, 'step': 0, 'offset': 0,
        'order': None, 'best_metric': -1., 'best_epoch': 0, 'best_reference': None, 'bad_epochs': 0,
        'curve': [], 'epoch_loss_sum': 0., 'epoch_record_count': 0, 'training_seconds': 0.,
        'configuration': cfg, 'candidate': candidate, 'partition_identity': train.identity,
        'memory': memory(), 'gradient_norm_max': 0.}

    def save():
        state['model'] = delta_state(model)
        state['optimizer'] = optimizer.state_dict()
        state['scheduler'] = scheduler.state_dict()
        state['rng'] = rng_state()
        state['memory'] = {k: max(state['memory'][k], v) for k, v in memory().items()}
        return checkpoint(root, state)

    try:
        if (root/'latest.json').exists():
            state = load_checkpoint(read(root/'latest.json'), identity)
            if state['phase'] == 'COMPLETE':
                result = {**state['result'], 'checkpoint': read(root/'latest.json')}
                publish(complete, result)
                return result
            restore_delta(model, state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            restore_rng(state['rng'])
            event(root, 'RESUMED', identity=identity, epoch=state['epoch'], step=state['step'], offset=state['offset'])
        else:
            save()
        state_check(model, optimizer)
        while state['epoch'] < planned_epochs and (validation is None or state['bad_epochs'] < cfg['patience']):
            if state['order'] is None:
                state['order'] = np.random.permutation(len(train.records)).tolist()
            while state['offset'] < len(state['order']):
                indexes = state['order'][state['offset']:state['offset']+cfg['accumulation_steps']]
                records = [train.records[i] for i in indexes]
                started = time.perf_counter()
                if fault is not None:
                    fault(state['step'])
                loss, norm = finite_training_group(model, processor, records, labels[indexes], weights[indexes], cfg, optimizer, scheduler)
                torch.cuda.synchronize()
                state['training_seconds'] += time.perf_counter()-started
                state['gradient_norm_max'] = max(state['gradient_norm_max'], norm)
                state['epoch_loss_sum'] += loss*len(indexes)
                state['epoch_record_count'] += len(indexes)
                state['offset'] += len(indexes)
                state['step'] += 1
                if state['step'] % cfg['checkpoint_every_updates'] == 0 or state['step'] == interrupt_after_step:
                    save()
                if state['step'] == interrupt_after_step:
                    raise InterruptedError('Controlled neural interruption after complete accumulation update')
            validation_started = time.perf_counter()
            rows = prediction_rows(validation.records, probabilities(model, processor, validation.records, cfg['fine_tune_seconds'])) if validation else []
            metric = metrics([r['label'] for r in rows], [r['probability'] for r in rows]) if rows else None
            improved = validation is None or metric['macro_f1'] > state['best_metric']
            state['epoch'] += 1
            state['curve'].append({'epoch': state['epoch'], 'train_weighted_loss': state['epoch_loss_sum']/state['epoch_record_count'],
                'validation_macro_f1': metric['macro_f1'] if metric else None, 'validation_metrics': metric,
                'validation_seconds': time.perf_counter()-validation_started, 'memory': memory(),
                'learning_rates': scheduler.get_last_lr()})
            if improved:
                state['best_metric'] = metric['macro_f1'] if metric else -1.
                state['best_epoch'] = state['epoch']
                state['bad_epochs'] = 0
                best = checkpoint(root/'best', {'identity': identity, 'model': delta_state(model),
                    'epoch': state['epoch'], 'step': state['step'], 'validation_metrics': metric, 'validation_predictions': rows})
                state['best_reference'] = best
            else:
                state['bad_epochs'] += 1
            state.update(order=None, offset=0, epoch_loss_sum=0., epoch_record_count=0)
            save()
        best_payload = load_checkpoint(state['best_reference'], identity)
        result = {'identity': identity, 'candidate': candidate, 'best_epoch': state['best_epoch'],
            'completed_epochs': state['epoch'], 'completed_steps': state['step'],
            'validation_metrics': best_payload['validation_metrics'], 'validation_predictions': best_payload['validation_predictions'],
            'training_seconds': state['training_seconds'], 'training_time_scope': 'audio decode plus forward/backward/update, excluding model loading/checkpoint I/O/validation',
            'parameter_count': model.parameter_report()['trainable'], 'parameters': model.parameter_report(),
            'best_checkpoint': state['best_reference'], 'curve': state['curve'],
            'memory': state['memory'], 'gradient_norm_max': state['gradient_norm_max'],
            'physical_batch_size': 1, 'accumulation_steps': cfg['accumulation_steps'],
            'sequence_seconds': cfg['fine_tune_seconds'],
            'training_records_per_second': len(train.records)*state['epoch']/max(state['training_seconds'], 1e-9)}
        state.update(phase='COMPLETE', result=result)
        reference = save()
        result = {**result, 'checkpoint': reference}
        publish(complete, result)
        return result
    except BaseException as exc:
        event(root, 'FAILED', identity=identity, type=type(exc).__name__, error=str(exc),
            epoch=state['epoch'], step=state['step'], offset=state['offset'],
            logical_ids=[train.records[i].logical_id for i in (state['order'] or [])[state['offset']:state['offset']+cfg['accumulation_steps']]],
            memory=memory(), recovery='No automatic retry. Physical batch one: OOM stops. Resume replays from last complete checkpoint.')
        raise
    finally:
        del model, optimizer, scheduler
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def predict_neural(result, directory, candidate, cfg, records):
    model, processor, _ = build(directory, candidate, cfg)
    try:
        payload = load_checkpoint(result['best_checkpoint'], result['identity'])
        restore_delta(model, payload['model'])
        return probabilities(model, processor, records, cfg['fine_tune_seconds'])
    finally:
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
