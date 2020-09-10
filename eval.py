import numpy as np


def lp_distance(x, adv_x, norm):
    """
    Compute average l0/l2/linf distance between clean and adversarial images.
    """
    num_imgs = x.shape[0]
    ss, lp = [], []
    if norm == 0:
        for i in range(num_imgs):
            lp.append(sum(np.abs(x[i:i + 1] - adv_x[i:i + 1]).flatten() > 1e-8))
    elif norm == 2:
        for i in range(num_imgs):
            lp.append(np.linalg.norm((x[i:i + 1] - adv_x[i:i + 1]).flatten()))
    elif norm == np.inf:
        for i in range(num_imgs):
            lp.append(np.linalg.norm((x[i:i + 1] - adv_x[i:i + 1]).flatten(), ord=np.inf))
    return np.mean(lp)


def metrics(model, x, adv_x, norm, targeted=False):
    m = x.shape[0]
    if targeted:
        target = np.tile(np.arange(10), m)
        x = np.repeat(x, 10, axis=0)

        # when target is identical to the original prediction, it's not count as a success.
        origin = model.predict_classes(x)
        target[target == origin] = -1
        success = target == model.predict_classes(adv_x)

        # when target is identical to the original prediction, the attack will do nothing, thus is excluded when
        # computing success rate
        succ_rate = np.sum(success) / (x.shape[0] - m)
        succ_idx = np.where(success)
    else:
        success = model.predict_classes(x) != model.predict_classes(adv_x)
        succ_rate = np.sum(success) / x.shape[0]
        succ_idx = np.where(success)
    return succ_rate, lp_distance(x[succ_idx], adv_x[succ_idx], norm)


def grid_visual(data):
    """
    This function displays a grid of images to show full misclassification
    :param data: grid data of the form;
      [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
    """
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [15, 15]

    # Ensure interactive mode is disabled and initialize our graph
    plt.ioff()
    figure = plt.figure()

    row = (lambda a: 1 if a < 10 else (10 if a > 100 else a // 10))(data.shape[0])
    column = np.amin([10, data.shape[0]])
    data = data[:int(row * column)].reshape((column, row) + data[0].shape)

    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    num_channels = data.shape[4]
    for y in range(num_rows):
        for x in range(num_cols):
            figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
            plt.axis('off')

            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])
    plt.show()
