import numpy as np
import innvestigate
import innvestigate.utils as iutils


def isi(model, indicator, x, y=None, norm=0, batch_size=100, **kwargs):
    """
    Input Significance Indicator based Attack, two indicators: sensitivity and relevance are included.

    Relevance-based attack supports l0,l2 or linf norm constraints.
    Our sensitivity-based attack only supports the l0 norm, since attack in
    other norms using sensitivity is very similar to the 'Basic Iterative Method'.

    :param y: required for targeted attack.
    :param indicator: choose sensitivity or relevance as the indicator.
    :param kwargs: when norm=0, 'gamma'=[0,1] is needed to show the maximum percentage of changeable features.
                    when norm=2, step size 'eps' and  changed features 'n' is needed,
                    when norm=np.inf, step size 'eps' and 'clip_values'=(min,max) is needed.
    :return: adversarial batch
    """
    if indicator == 'sensitivity' and norm != 0:
        raise ValueError('input sensitivity based attack only supports L0 norm, for other norms try the Basic '
                         'Iterative Method/(or Projected Gradient Descent/)')
    indicator = 'gradient' if indicator == 'sensitivity' else 'lrp.epsilon'
    dims = list(x[0].shape)
    nb_features = np.product(dims)
    adv_x = np.reshape(x.astype(np.float32), (-1, nb_features))
    model = iutils.model_wo_softmax(model)
    preds = np.argmax(model.predict(x), axis=1)
    if y is None:
        analyzer = innvestigate.create_analyzer(indicator, model)
    else:
        analyzer = innvestigate.create_analyzer(indicator, model, neuron_selection_mode='index')

    for batch_id in range(int(np.ceil(adv_x.shape[0] / float(batch_size)))):
        batch_index_1, batch_index_2 = batch_id * batch_size, (batch_id + 1) * batch_size
        batch = adv_x[batch_index_1:batch_index_2]
        current_pred = preds[batch_index_1:batch_index_2]
        if y is None:
            active_indices = np.where(current_pred == preds[batch_index_1:batch_index_2])[0]
        else:
            target = np.zeros_like(current_pred) + y
            active_indices = np.where(current_pred != target)[0]
        i = 0
        used_features = np.zeros_like(batch)
        while len(active_indices) != 0 and i < np.floor(kwargs['gamma'] * nb_features):
            r = analyzer.analyze(np.reshape(batch, [batch.shape[0]] + dims)[active_indices], neuron_selection=y)
            r = np.reshape(r, (-1, nb_features))

            if norm == 0:
                batch, used_features = _apply_l0_perturbation(batch, r, indicator, y, active_indices, used_features)
            elif norm == 2:
                batch = _apply_l2_perturbation(batch, r, y, active_indices, kwargs['n'], kwargs['eps'])
            elif norm == np.inf:
                batch = _apply_linf_perturbation(batch, r, y, active_indices, kwargs['eps'], kwargs['clip_values'])
            current_pred = np.argmax(model.predict(np.reshape(batch, [batch.shape[0]] + dims)), axis=1)
            if y is None:
                active_indices = np.where(current_pred == preds[batch_index_1:batch_index_2])[0]
            else:
                active_indices = np.where(current_pred != target)[0]
            i += 1
        adv_x[batch_index_1:batch_index_2] = batch
    adv_x = np.reshape(adv_x, x.shape)
    return adv_x


def _apply_l0_perturbation(batch, score, indicator, y, active_indices, used_features):
    """
    Add perturbations to data batch, and record the features that have been used.
    """
    act_used_features = used_features[active_indices]
    if indicator == 'gradient':
        score[act_used_features == 1] = 0  # set sensitivity of already used features to zero
        ind = np.argpartition(np.abs(score), -1, axis=1)[:, -1:]  # find feature with the largest abs(sensitivity)
        tmp_batch = batch[active_indices]
        if y is None:
            tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = -1 * np.sign(
                score[np.arange(len(active_indices)), ind[:, 0]])

        else:
            tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = np.sign(
                score[np.arange(len(active_indices)), ind[:, 0]])

        batch[active_indices] = tmp_batch
    else:
        if y is None:
            score[act_used_features == 1] = -np.inf  # set relevance of already used features to -inf
            ind = np.argpartition(score, -1, axis=1)[:, -1:]  # find feature with the largest relevance
        else:
            score[act_used_features == 1] = np.inf  # set relevance of already used features to inf
            ind = np.argpartition(score, 0, axis=1)[:, 0:]  # find feature with the least relevance
        tmp_batch = batch[active_indices]
        # tmp_batch[np.arange(len(active_indices)), ind[:, 0]] *= -1
        tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = -np.sign(
            tmp_batch[np.arange(len(active_indices)), ind[:, 0]])

        batch[active_indices] = tmp_batch
    used_features[active_indices, ind[:, 0]] = 1
    return batch, used_features


def _apply_l2_perturbation(batch, r, y, active_indices, n, eps):
    if y is None:
        ind = np.argpartition(r, -n, axis=1)[:, (-n):]  # find n features with the largest relevance
    else:
        ind = np.argpartition(r, n - 1, axis=1)[:, :n]  # find n features with the least relevance
    tmp_batch = batch[active_indices]
    for i in range(n):
        tmp_batch[np.arange(len(active_indices)), ind[:, i]] -= eps * np.sign(
            tmp_batch[np.arange(len(active_indices)), ind[:, i]])
    batch[active_indices] = tmp_batch
    return batch


def _apply_linf_perturbation(batch, r, y, active_indices, eps, clip_values):
    tmp_batch = batch[active_indices]
    if y is None:
        tmp_batch[np.arange(len(active_indices)), :] -= eps * np.sign(r) * np.sign(
            tmp_batch[np.arange(len(active_indices)), :])
    else:
        tmp_batch[np.arange(len(active_indices)), :] += eps * np.sign(r) * np.sign(
            tmp_batch[np.arange(len(active_indices)), :])
    tmp_batch = np.clip(tmp_batch, clip_values[0], clip_values[1])
    batch[active_indices] = tmp_batch
    return batch
