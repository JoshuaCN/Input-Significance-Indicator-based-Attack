# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack."""

from cleverhans.plot.pyplot_image import grid_visual
from cleverhans.attacks import CarliniWagnerL2 as CW_hans, SaliencyMapMethod as JSMA_hans
from cleverhans.utils_keras import KerasModelWrapper
from metrics import *
from train import create_lenet_model, create_cnn_model
from relattack import *
from art.attacks.carlini import CarliniL2Method as CW_2, CarliniLInfMethod as CW_inf
from art.attacks.projected_gradient_descent import ProjectedGradientDescent as PGD_art
from art.classifiers import KerasClassifier
from art.utils import load_dataset
import tensorflow as tf
import keras
import time
import numpy as np ; na = np.newaxis


dataset = [
        # 'mnist',
        'cifar10',
]
Targeted = True
Defense = False
Show = True
Transfer = False
NORM = 0
MAX_ITER = 20
EPS = 0.4
# m = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]
m = range(10)

attacks = [

            # 'SA',
            'FEA',
            # 'PGD_art',
            # 'JSMA_hans',
            # 'CW_art',
            # 'CW_hans',
]


# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

(_, _), (X, Y), _, _ = load_dataset(str(dataset[0]))
X = X[m, ...]
Y = Y[m, :]
if dataset == ['mnist']:
    model = create_lenet_model(X.shape[1:])
    model.load_weights('./models/mnist.h5')

else:
    model = create_cnn_model(X.shape[1:])
    model.load_weights('./models/cifar10.h5')

X = X * 2 - 1

art = KerasClassifier((-1., 1.), model=model)
clever = KerasModelWrapper(model)
(img_rows, img_cols, nchannels) = X.shape[1:4]


if 'SA' in attacks:
    start = time.time()
    SA_X = relevance(model, X, method='gradient', targeted=Targeted, max_iter=MAX_ITER)  # 25
    end = time.time()
    print('Time:%.1f' % (end - start))

    succ_rate, measure = evaluate(model, X, SA_X, targeted=Targeted)
    print('\nSA success rate: %.2f%%' % (succ_rate * 100))
    print('\nMetrics:%.2f, %.2f, %.3f, %.3f' % (measure[0:4]))
    if Defense:
        defenses(model, X, SA_X, FS=True, SS=True)
    if Transfer:
        transfer(X, SA_X)
    if Show:
        SA_X = (SA_X + 1)/2
        grid_visual(np.reshape(SA_X, (10, SA_X.shape[0] // 10, img_rows, img_cols, nchannels)))


if 'FEA' in attacks:
    start = time.time()
    if NORM == 0:
        FEA_X = relevance(model, X, 'lrp.z', MAX_ITER, Targeted)
        # FEA_X = FeatureEnhancementAttack(model, X, targeted=False, max_iter=MAX_ITER, norm=NORM, confidence=0)  # 25
    # elif NORM == 2:
    #     # FEA_X = FeatureEnhancementAttack(model, X, targeted=False, eps=0.1, n=20, max_iter=MAX_ITER, norm=NORM,
    #     #                                  confidence=0)  # 40
    #     FEA_X = fea_batch(max_iter=MAX_ITER,model=model,x=X,y=Y,eps=EPS,batch_size=128,norm=NORM)
    # elif NORM == np.inf:
    #     # FEA_X = FeatureEnhancementAttack(model, X, targeted=False, eps=EPS, max_iter=MAX_ITER, norm=NORM,norm2=np.inf)
    #     FEA_X = fea_batch(max_iter=MAX_ITER,model=model,x=X,y=Y,eps=EPS,batch_size=128,norm=NORM)
    end = time.time()
    print('Time:%.1f' % (end-start))

    # preds = np.argmax(art.predict(FEA_X), axis=1)
    # acc = np.sum(preds == np.argmax(Y, axis=1)) / Y.shape[0]
    # (_, acc) = model.evaluate(FEA_X, Y)
    succ_rate,measure = evaluate(model, X, FEA_X, targeted=Targeted)
    print("\nFEA-%s success rate: %.2f%%" % (NORM, succ_rate * 100))
    print('\nMetrics:%.2f, %.2f, %.3f, %.3f' % (measure[0:4]))
    if Defense:
        defenses(model,X,FEA_X,FS=True,SS=True)
    if Transfer:
        transfer(X, FEA_X)
    if Show:
        FEA_X = (FEA_X + 1) / 2
        grid_visual(np.swapaxes(np.reshape(FEA_X, (10, FEA_X.shape[0] // 10, img_rows, img_cols, nchannels)),0,1))

if 'PGD_art' in attacks:
    pgd = PGD_art(art)
    if NORM == 2:
        pgd_params = {
                    'eps': 10.,
                    'eps_step': EPS,
                    'max_iter': MAX_ITER,
                    'norm': 2,
                    # 'batch_size': 1,
                     }
    elif NORM == np.inf:
        pgd_params = {
                    'eps': 10.,
                    'eps_step': EPS,
                    'max_iter': MAX_ITER,
                    'norm': np.inf,
                    # 'batch_size': 1,
                     }
    start = time.time()
    PGD_X = pgd.generate(X, **pgd_params)
    end = time.time()
    print('Time:%.1f' % (end-start))
    succ_rate,measure = evaluate(model,X,PGD_X)
    print("\nPGD-%s success rate: %.2f%%" % (NORM, succ_rate * 100))
    print('\nMetrics:%.2f, %.2f, %.3f, %.3f' % (measure[0:4]))
    if Defense:
        defenses(model,X,PGD_X,FS=True,SS=True)
    if Transfer:
        transfer(X, PGD_X)
    if Show:
        PGD_X = (PGD_X + 1) / 2
        grid_visual(np.reshape(PGD_X, (10, PGD_X.shape[0] // 10, img_rows, img_cols, nchannels)))


if 'JSMA_hans' in attacks:
    jsma = JSMA_hans(clever, sess=sess)
    jsma_params = {'theta': 2., 'gamma': MAX_ITER / (img_cols*img_rows*nchannels),
                   'clip_min': -1., 'clip_max': 1., 'y_target': None
                   }
    if not Targeted:
        start = time.time()
        JSMA_X = jsma.generate_np(X, **jsma_params)
        end = time.time()
        print('Time:%.1f' % (end - start))
    elif Targeted:
        JSMA_X = np.zeros([10,np.size(m),img_rows,img_cols,nchannels])
        start = time.time()
        for target in range(10):
            one_hot_target = np.zeros((1, 10))
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            JSMA_X[target] = jsma.generate_np(X, **jsma_params)
        JSMA_X = JSMA_X.swapaxes(0, 1).reshape([10 * np.size(m), img_rows, img_cols, nchannels])
        end = time.time()
        print('Time:%.1f' % (end - start))
    succ_rate,measure = evaluate(model,X,JSMA_X,Targeted)
    print("\nJSMA-%s success rate: %.2f%%" % (NORM, succ_rate * 100))
    print('\nMetrics:%.2f, %.2f, %.3f, %.3f' % (measure[0:4]))
    if Defense:
        defenses(model,X,JSMA_X,FS=True,SS=True)
    if Transfer:
        transfer(X, JSMA_X)
    if Show:
        JSMA_X = (JSMA_X + 1) / 2
        grid_visual(np.reshape(JSMA_X, (10, JSMA_X.shape[0] // 10, img_rows, img_cols, nchannels)))


if 'CW_hans' in attacks:
    cw = CW_hans(clever, sess=sess)
    cw_params = {'binary_search_steps': 1,
                 'max_iterations': MAX_ITER,
                 'learning_rate': 0.1,
                 'initial_const': 10,
                 'clip_min': -1.,
                 'clip_max': 1.,
                 'confidence': 0,
                 }
    start = time.time()
    CW_X = cw.generate_np(X, **cw_params)
    end = time.time()
    print('Time:%.1f' % (end - start))
    succ_rate,measure = evaluate(model,X,CW_X)
    print("\nCW-%s success rate: %.2f%%" % (NORM, succ_rate * 100))
    print('\nMetrics:%.2f, %.2f, %.3f, %.3f' % (measure[0:4]))
    if Defense:
        defenses(model,X,CW_X,FS=True,SS=True)
    if Transfer:
        transfer(X, CW_X)
    if Show:
        CW_X = (CW_X + 1) / 2
        grid_visual(np.reshape(CW_X, (10, CW_X.shape[0] // 10, img_rows, img_cols, nchannels)))



