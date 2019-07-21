import numpy as np
import innvestigate
import innvestigate.utils as iutils


def relevance(model, x, method, gamma, y=None, batch_size=100, norm=0, n=None, eps=None):
    """

    :param model:
    :param x:
    :param method:
    :param gamma:
    :param y:
    :param batch_size:
    :param norm:
    :param n:
    :param eps:
    :return:
    """
    dims = list(x.shape[1:])
    nb_features = np.product(dims)
    adv_x = np.reshape(x.astype(np.float32), (-1, nb_features))
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    preds = np.argmax(model.predict(x), axis=1)
    if y is None:
        analyzer = innvestigate.create_analyzer(method, model_wo_softmax)
    else:
        analyzer = innvestigate.create_analyzer(method, model_wo_softmax, neuron_selection_mode='index')

    for batch_id in range(int(np.ceil(adv_x.shape[0] / float(batch_size)))):

        batch_index_1, batch_index_2 = batch_id * batch_size, (batch_id + 1) * batch_size
        batch = adv_x[batch_index_1:batch_index_2]
        current_pred = preds[batch_index_1:batch_index_2]
        if y is None:
            active_indices = np.where(current_pred == preds[batch_index_1:batch_index_2])[0]
        else:
            target = np.zeros_like(current_pred) + y
            active_indices = np.where(current_pred != target)[0]
        it = 0
        used_features = np.zeros_like(batch)
        while len(active_indices) != 0 and it < np.floor(gamma * nb_features):
            r = analyzer.analyze(np.reshape(batch, [batch.shape[0]] + dims)[active_indices], neuron_selection=y)
            r = np.reshape(r, (-1, nb_features))
            if norm == 0:
                batch, used_features = _apply_l0_perturbation(batch, r, method, y, active_indices, used_features)
            elif norm == 2:
                batch = _apply_l2_perturbation(batch, r, y, active_indices, n, eps)
            current_pred = np.argmax(model.predict(np.reshape(batch, [batch.shape[0]] + dims)), axis=1)
            if y is None:
                active_indices = np.where(current_pred == preds[batch_index_1:batch_index_2])[0]
            else:
                active_indices = np.where(current_pred != target)[0]
            it += 1
        adv_x[batch_index_1:batch_index_2] = batch
    adv_x = np.reshape(adv_x, x.shape)
    return adv_x


def _apply_l0_perturbation(batch, r, method, y, active_indices, used_features):
    """
    Introduce perturbations to batch, and record the features that have been used.
    :param batch:
    :param r:
    :param method:
    :param y:
    :param active_indices:
    :param used_features:
    :return: perturbed batch, already used features
    """
    act_used_features = used_features[active_indices]
    if method == 'gradient':
        r[act_used_features == 1] = 0  # 将已用的特征的r值设为0
        ind = np.argpartition(np.abs(r), -1, axis=1)[:, -1:]  # 取r的绝对值最大的特征
        tmp_batch = batch[active_indices]
        if y is None:
            tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = -1 * np.sign(r[np.arange(len(active_indices)), ind[:, 0]])
            # tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = -0.5 * np.sign(r[np.arange(len(active_indices)), ind[:, 0]]) + 0.5
        else:
            tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = np.sign(r[np.arange(len(active_indices)), ind[:, 0]])
            # tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = 0.5 * np.sign(r[np.arange(len(active_indices)), ind[:, 0]]) + 0.5
        batch[active_indices] = tmp_batch

    elif method == 'lrp.z' or method == 'lrp.z_IB':
        if y is None:
            r[act_used_features == 1] = -np.inf  # 若为 untargeted，将已用的特征的r值设为 -inf
            ind = np.argpartition(r, -1, axis=1)[:, -1:]  # 取r值最大的特征
        else:
            r[act_used_features == 1] = np.inf  # 若为 targeted，将已用的特征的r值设为 inf
            ind = np.argpartition(r, 0, axis=1)[:, 0:]  # 取r值最小的特征
        tmp_batch = batch[active_indices]
        # tmp_batch[np.arange(len(active_indices)), ind[:, 0]] *= -1
        tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = -1 * np.sign(tmp_batch[np.arange(len(active_indices)), ind[:, 0]])
        # tmp_batch[np.arange(len(active_indices)), ind[:, 0]] = -0.5 * np.sign(tmp_batch[np.arange(len(active_indices)), ind[:, 0]] - 0.5) + 0.5
        batch[active_indices] = tmp_batch
    else:
        raise ValueError
    used_features[active_indices, ind[:, 0]] = 1
    return batch, used_features


def _apply_l2_perturbation(batch, r, y, active_indices, n, epsilon):
    if y is None:
        ind = np.argpartition(r, -n, axis=1)[:, (-n):]  # 取r值最大的特征
    else:
        ind = np.argpartition(r, n-1, axis=1)[:, (n-1):]  # 取r值最小的特征
    tmp_batch = batch[active_indices]
    for i in range(n):
        tmp_batch[np.arange(len(active_indices)), ind[:, i]] -= epsilon * np.sign(tmp_batch[np.arange(len(active_indices)), ind[:, i]])
    batch[active_indices] = tmp_batch
    return batch
