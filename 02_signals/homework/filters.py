import numpy as np
from numpy.lib.stride_tricks import as_strided


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_h = Hk // 2
    pad_w = Wk // 2

    for i in range(Hi):
        for j in range(Wi):
            for m in range(-pad_h, pad_h + 1):
                for n in range(-pad_w, pad_w + 1):
                    x = i + m
                    y = j + n

                    if x < 0 or x >= Hi or y < 0 or y >= Wi:
                        continue

                    k_idx = pad_h - m
                    l_idx = pad_w - n

                    out[i, j] += image[x, y] * kernel[k_idx, l_idx]
    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width), dtype=image.dtype)
    out[pad_height: pad_height + H, pad_width: pad_width + W] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    kernel_flipped = np.flip(np.flip(kernel, axis=0), axis=1)

    pad_height = Hk // 2
    pad_width = Wk // 2

    padded_image = zero_pad(image, pad_height, pad_width)

    for i in range(Hi):
        for j in range(Wi):
            region = padded_image[i:i+Hk, j:j+Wk]
            out[i, j] = np.sum(region * kernel_flipped)

    return out


def conv_faster(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    pad_h = Hk // 2
    pad_w = Wk // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    shape = (Hi, Wi, Hk, Wk)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)
    
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    return np.einsum('ijkl,kl->ij', windows, kernel_flipped)


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    kernel = np.flip(g, axis=(0, 1))
    out = conv_faster(f, kernel)

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g_mean = np.mean(g)
    g_zero_mean = g - g_mean
    out = cross_correlation(f, g_zero_mean)

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    g_mean = np.mean(g)
    g_std = np.std(g)
    g_normalized = (g - g_mean) / (g_std + 1e-8)
    
    pad_h = Hg // 2
    pad_w = Wg // 2
    padded_f = zero_pad(f, pad_h, pad_w)
    
    out = np.zeros((Hf, Wf))

    for i in range(Hf):
        for j in range(Wf):
            patch = padded_f[i:i+Hg, j:j+Wg]
            
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            patch_normalized = (patch - patch_mean) / (patch_std + 1e-8)
            
            ncc_value = np.sum(patch_normalized * g_normalized)
            out[i, j] = ncc_value
            
    return out
