import numpy as np
import innvestigate
import innvestigate.utils as iutils


def FeatureEnhancementAttack(model, X, targeted=False, rule='iter', eps=0.1, n=10):
    print('Waiting...')
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    if not targeted:
        adv_X = np.zeros(X.shape)
        analyzer = innvestigate.create_analyzer("lrp.z", model_wo_softmax)
        for source in range(X.shape[0]):

            idx_start = 0
            if rule == 'flip':
                idx_end = 1
            else:
                idx_end = n
            adv_x = X[source:source + 1].copy()
            label = np.argmax(model.predict(adv_x))
            while label == np.argmax(model.predict(adv_x)):
                r = analyzer.analyze(adv_x)
                idx = (-r).flatten().argsort()[idx_start:idx_end]
                if rule == 'flip':
                    adv_x.ravel()[idx] *= -1
                    r = analyzer.analyze(adv_x)
                    idx1 = (-r).flatten().argsort()[0:idx_end]
                    if idx in idx1:
                        adv_x.ravel()[idx] *= -1
                        idx_start += 1
                        idx_end += 1
                else:
                    adv_x.ravel()[idx] -= eps * np.sign(adv_x.ravel())[idx]

            adv_X[source, ...] = adv_x

    if targeted:
        adv_X = np.zeros([X.shape[0],10,X.shape[1],X.shape[2],X.shape[3]])
        analyzer = innvestigate.create_analyzer("lrp.z", model_wo_softmax, neuron_selection_mode='index')
        for source in range(X.shape[0]):

            for target in range(10):

                idx_start = 0
                if rule == 'flip':
                    idx_end = 1
                else:
                    idx_end = n
                adv_x = X[source:source + 1].copy()

                while np.argmax(model.predict(adv_x)) != target:
                    r = analyzer.analyze(adv_x, neuron_selection=target)
                    idx = r.flatten().argsort()[idx_start:idx_end]
                    if rule == 'flip':
                        adv_x.ravel()[idx] *= -1
                        r = analyzer.analyze(adv_x, neuron_selection=target)
                        idx1 = r.flatten().argsort()[0:idx_end]
                        if idx in idx1:
                            adv_x.ravel()[idx] *= -1
                            idx_start += 1
                            idx_end += 1
                    else:
                        adv_x.ravel()[idx] -= eps * np.sign(adv_x.ravel())[idx]
                adv_X[source, target, ...] = adv_x

    return adv_X
