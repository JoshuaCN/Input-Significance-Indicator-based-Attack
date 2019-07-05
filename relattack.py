import numpy as np
import innvestigate
import keras.models
import innvestigate.utils as iutils
from utils_mnist import heatmap
from cleverhans.plot.pyplot_image import grid_visual


def relevance(model, X, method, max_iter, targeted=False):
    """
        The ordinary version of relattack, update relevance each iteration to generate an adversarial example.
        :param model: A trained keras model
        :param X: Benign samples [samples,rows,cols,channels]
        :param method: 'lrp.z' or 'gradient'
        :param max_iter: max_iterations allowed for one sample
        :param targeted:
        :return: if targeted, return adv_X, in shape of [samples,classes,rows,cols,channels],
                if not, return adv_X, in shape of [samples,rows,cols,channels]
        """

    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

    if not targeted:
        adv_X = np.zeros(X.shape)
        analyzer = innvestigate.create_analyzer(method, model_wo_softmax)
        for source in range(X.shape[0]):
            # print('--------------------------------------')
            # print('Attacking input %i/%i' % (source + 1, X.shape[0]))
            idx_start = 0; idx_end = 1; it = 0
            perturbed_ids = []
            adv_x = X[source:source + 1].copy()
            # prediction = model.predict_classes(adv_x)
            prediction = model.predict(adv_x).argmax()

            while prediction == model.predict(adv_x).argmax() and it < max_iter:

                r = analyzer.analyze(adv_x)

                # a = heatmap(r)
                # grid_visual(np.reshape(a,[1,1,28,28,3]))

                adv_x, idx = strategy(adv_x,r,method,targeted,idx_start,idx_end)
                it += 1

                if idx in perturbed_ids:
                    idx_start += 1
                    idx_end += 1
                    continue

                perturbed_ids.append(idx)

            adv_X[source, ...] = adv_x

    if targeted:

        adv_X = np.zeros([X.shape[0]*10,X.shape[1],X.shape[2],X.shape[3]])

        analyzer = innvestigate.create_analyzer(method, model_wo_softmax, neuron_selection_mode='index')
        for source in range(X.shape[0]):
            # print('--------------------------------------')
            # print('Attacking input %i/%i' % (source + 1, X.shape[0]))
            for target in range(10):
                # print('Generating adv. example for target class %i' % target)
                idx_start = 0;  idx_end = 1; it = 0
                adv_x = X[source:source + 1].copy()
                perturbed_ids = []
                # a = ivis.heatmap(analysis)
                # plt.imshow(a[0], cmap="seismic")
                # plt.show()
                while model.predict(adv_x).argmax() != target and it < max_iter:
                    r = analyzer.analyze(adv_x, neuron_selection=target)
                    adv_x, idx = strategy(adv_x, r, method, targeted, idx_start, idx_end)
                    it += 1

                    if idx in perturbed_ids:
                        idx_start += 1
                        idx_end += 1
                        continue

                    perturbed_ids.append(idx)

                adv_X[source*10 + target, ...] = adv_x
    return adv_X


def strategy(adv_x, r, method, targeted, idx1, idx2):
    if method == 'gradient':
        if not targeted:
            # Index search and Modification(SA)
            idx = (-(np.abs(r))).flatten().argsort()[idx1:idx2]
            adv_x.ravel()[idx] = -1 * np.sign(r.flatten()[idx])  # for x=[-1,1]
            # adv_x.ravel()[idx] = -0.5 * np.sign(r.flatten()[idx]) + 0.5  # for x=[0,1]
        elif targeted:
            idx = (-(np.abs(r))).flatten().argsort()[idx1:idx2]
            adv_x.ravel()[idx] = np.sign(r.flatten()[idx])  # for x=[-1,1]
        else:
            raise ValueError

    elif method == 'lrp.z' or method == 'lrp.z_IB':
        if not targeted:
            idx = (-r).flatten().argsort()[idx1:idx2]
            adv_x.ravel()[idx] *= -1  # pixel filpping x=[-1,1]
        elif targeted:
            idx = r.flatten().argsort()[idx1:idx2]
            adv_x.ravel()[idx] *= -1
        else:
            raise ValueError
    else:
        raise ValueError
    return adv_x, idx
