from skimage.measure import compare_ssim, compare_psnr, compare_mse
import numpy as np; na = np.newaxis
from keras.models import load_model
from art.defences.feature_squeezing import FeatureSqueezing
from art.defences.spatial_smoothing import SpatialSmoothing


def metrics(X, adv_X, avg=True):
    imgs = X.shape[0]
    mse = np.zeros(imgs); ps = np.zeros(imgs); ss = np.zeros(imgs)
    L_0 = np.zeros(imgs); L_2 = np.zeros(imgs); L_inf = np.zeros(imgs)
    for i in range(imgs):
        L_2[i] = np.linalg.norm((X[i,...]-adv_X[i,...]).flatten())
        L_0[i] = sum(np.abs(X[i,...]-adv_X[i,...]).flatten()>10e-8)
        L_inf[i] = np.linalg.norm((X[i,...]-adv_X[i,...]).flatten(),ord=np.inf)
        # mse[i] = compare_mse(X[i,...], adv_X[i,...])
        # ps[i] = compare_psnr(X[i,...],adv_X[i,...],data_range=2)
        ss[i] = compare_ssim(X[i, ...], adv_X[i, ...], data_range=2, multichannel=True, gaussian_weights=False)
    if avg:
        # return np.mean(L_0[L_0>0]), np.mean(L_2[L_2>0]), np.mean(L_inf[L_inf>0]), np.mean(ss[ss<1]), np.mean(mse[mse>0]), np.mean(ps[ps<100])
        return L_0.mean(), L_2.mean(), L_inf.mean(), ss.mean()
    else:
        return L_0, L_2, L_inf, ss


def evaluate(model,X,adv_X,targeted=False):
    m = X.shape[0]
    if targeted:
        target = np.tile(np.arange(10), m)
        X = np.repeat(X, 10, axis=0)
        truth = model.predict_classes(X)
        target[target == truth] = -1

        success = target == model.predict_classes(adv_X)
        succ_rate = np.sum(success) / (X.shape[0]-m)
        succ_idx = np.where(success)
        measure = metrics(X[succ_idx], adv_X[succ_idx])
        # target = np.zeros(X.shape[0])
        # for i in range(X.shape[0]):
        #     target[i] = np.argsort(model.predict(X[i:i+1]).flatten())[-2]
        # success = model.predict_classes(adv_X) == target
        # succ_rate = np.sum(success) / X.shape[0]
        # succ_idx = np.where(success)
        # measure = metrics(X[succ_idx], adv_X[succ_idx])
    else:
        success = np.argmax(model.predict(X), axis=1) != np.argmax(model.predict(adv_X), axis=1)
        succ_rate = np.sum(success) / X.shape[0]
        succ_idx = np.where(success)
        measure = metrics(X[succ_idx], adv_X[succ_idx])
    return succ_rate, measure


def transfer(X,adv_X,targeted=False):
    model_tran = load_model('./models/mnist_LeNet_02.h5')
    succ_rate, _ = evaluate(model_tran, X, adv_X,targeted=targeted)
    print("\nTransfer to 02 %s success rate: %.2f%%" % (targeted,succ_rate * 100))
    model_tran = load_model('./models/mnist_LeNet_03.h5')
    succ_rate, _ = evaluate(model_tran, X, adv_X,targeted=targeted)
    print("\nTransfer to 03 %s success rate: %.2f%%" % (targeted,succ_rate * 100))
    return 0


def defenses(model, X, adv_X, FS=False, SS=False, targeted=False):
    if FS:
        fs = FeatureSqueezing(bit_depth=1)(adv_X, clip_values=(-1, 1))
        succ_rate, _ = evaluate(model, X, fs,targeted=targeted)
        print("\nAfter FS %s success rate: %.2f%%" % (targeted,succ_rate * 100))
    if SS:
        ss = SpatialSmoothing(channel_index=1, window_size=20)(adv_X, clip_values=(-1, 1))
        succ_rate, _ = evaluate(model, X, ss,targeted=targeted)
        print("\nAfter SS %s success rate: %.2f%%" % (targeted,succ_rate * 100))
    return 0
