import matplotlib.pyplot as plt
import numpy as np
import pylab as pyl
from pylab import linalg

from nt_toolbox.general import *
from nt_toolbox.perform_wavelet_transf import *
from nt_toolbox.signal import *

n0 = 512
M = n0**2 // 64


def plot_image() -> np.ndarray:
    f = rescale(load_image("nt_toolbox/data/hibiscus.bmp", n0))
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image f")
    plt.figure(figsize=(5, 5))
    imageplot(f[n0 // 2 - 32 : n0 // 2 + 32, n0 // 2 - 32 : n0 // 2 + 32], "Zoom")
    plt.show()
    return f


def fourier_transform(f: np.ndarray):
    F = pyl.fft2(f) / n0  # forward transform
    print(f"Energy of Image: {linalg.norm(f):f}")
    print(f"Energy of Fourier: {linalg.norm(F):f}")
    ###
    # Q: Why is the signal energy preserved in the frequency domain?
    # A: The FT and IFT are lossless transforms of the input signals - the input
    #   can be reconstructed exactly 1:1 from the output. Hence, the signa
    #   energy must be exactly preserved, otherwise they would be lossy and
    #   reconstruction would be impossible.

    L = pyl.fftshift(np.log(abs(F) + 1e-1))
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image", [1, 2, 1])
    imageplot(L, "Log(Fourier transform)", [1, 2, 2])
    plt.show()


def filtering(f: np.ndarray) -> np.ndarray:
    q = int(np.sqrt(M))
    F = pyl.fftshift(pyl.fft2(f))
    Sel = np.zeros([n0, n0])
    Sel[n0 // 2 - q // 2 : n0 // 2 + q // 2, n0 // 2 - q // 2 : n0 // 2 + q // 2] = 1
    F_zeros = np.multiply(F, Sel)
    imageplot(
        np.log(abs(F_zeros) + 1e-1), "Log(Fourier transform)"
    )  # plot in frequency domain
    f_lpf = np.real(pyl.ifft2(pyl.fftshift(F_zeros)))  # inverse transform
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image", [1, 2, 1])
    imageplot(clamp(f_lpf), "Low-pass filtered image", [1, 2, 2])
    plt.show()

    ###
    # Q: Can you observe the difference between the original image and the filtered one?
    # A: the low pass filtered one is a blurred (smoothed) version of the original image
    #
    # Q: Now perform a high-pass filtering (how can you do this?). Can you explain the filtering outcome?
    Sel_hpf = np.ones([n0, n0])
    Sel_hpf[
        n0 // 2 - q // 2 : n0 // 2 + q // 2, n0 // 2 - q // 2 : n0 // 2 + q // 2
    ] = 0
    F_hpf = np.multiply(F, Sel_hpf)
    imageplot(
        np.log(abs(F_hpf) + 1e-1), "Log(Fourier transform)"
    )  # plot in frequency domain
    f_hpf = np.real(pyl.ifft2(pyl.fftshift(F_hpf)))
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image", [1, 2, 1])
    imageplot(clamp(f_hpf), "High-pass filtered image", [1, 2, 2])
    plt.show()
    # A: The filtering outcome is only the details of the image (ie the parts removed when we used the LPF).
    #   This acts like an edge detector
    return f_lpf


def fourier_vs_wavelet(f: np.ndarray, f_lpf: np.ndarray):
    Jmin = 0
    Fw = perform_wavelet_transf(f, Jmin, +1)  # forward transform
    plt.figure(figsize=(5, 5))
    plot_wavelet(Fw)
    plt.title("Wavelet coefficients")

    a = np.sort(np.ravel(abs(Fw)))[
        ::-1
    ]  # sort a 1D copy of magnitude of fw in descending order
    Fw_zeros = np.multiply(
        Fw, (abs(Fw) > a[M])
    )  # keeping the M largest wavelet coefficients
    plt.figure(figsize=(5, 5))
    plot_wavelet(Fw_zeros)  # comapre with the plot of coefficients above
    plt.title("Wavelet coefficients (thresholded)")
    plt.show()
    f_wav = perform_wavelet_transf(Fw_zeros, Jmin, -1)  # inverse transform
    plt.figure(figsize=(5, 5))
    imageplot(clamp(f_lpf), "Fourier approximation", [1, 2, 1])
    imageplot(clamp(f_wav), "Wavelet approximation", [1, 2, 2])
    plt.show()
    ###
    # Q: Can you comment on the comparison between the two approximations above. Why is one better than the other?
    # A: The wavelet approximation is overall much better, although the fourier approximation looks better in some
    #   small areas. The fourier approximation is uniformly blurred throughout the image, because it is a low-pass filter
    #   it cannot represent smaller details in the image. The wavelet approximation is better in this case for two
    #   reasons:
    #   - because the wavelet approximation is formed of the M largest wavelet coefficients, rather than the M lowest
    #     frequency fourier coefficients.
    #   - because the wavelets are localised in both position and frequency, so they can preserve "high frequency"
    #     spacial regions such as edges better than the fourier approximation.


def linear_denoising() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 256
    N = n**2
    name = "nt_toolbox/data/boat.bmp"
    f = load_image(name, n)
    plt.close("all")
    imageplot(f)
    sigma = 0.08
    y = f + sigma * np.random.standard_normal(f.shape)
    imageplot(clamp(y))

    cconv = lambda a, b: np.real(pyl.ifft2(pyl.fft2(a) * pyl.fft2(b)))
    normalize = lambda h: h / sum(h.flatten())
    t = np.transpose(np.concatenate((np.arange(0, n / 2), np.arange(-n / 2, 0))))
    [Y, X] = np.meshgrid(t, t)
    h = lambda mu: normalize(np.exp(-(X**2 + Y**2) / (2 * mu**2)))

    mu = 5
    plt.figure(figsize=(5, 5))
    imageplot(pyl.fftshift(h(mu)), "kernel", [1, 2, 1])
    imageplot(pyl.fftshift(np.real(pyl.fft2(h(mu)))), "FT of kernel", [1, 2, 2])
    denoise = lambda x, mu: cconv(h(mu), x)
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image", [1, 3, 1])
    imageplot(y, "Noisy image", [1, 3, 2])
    imageplot(denoise(y, mu), "Denoised image", [1, 3, 3])

    plt.show()
    ###
    # Q: Why is this denoising strategy linear?
    # A: This is linear because convolution with a fixed kernel is a linear operation with respect to the image
    #
    # Q: Can you find an optimal μ that minimises the l2-norm of the difference between original and denoised image?
    # A: solve via gradient descent:
    loss_fn = lambda mu: np.linalg.norm(denoise(y, mu) - f)
    epsilon = 1e-9
    gradient_fn = lambda mu: (
        (loss_fn(mu + epsilon) - loss_fn(mu - epsilon)) / (2 * epsilon)
    )  # approximate using central difference

    learning_rate = 0.5
    for step in range(10_000):
        gradient = gradient_fn(mu)
        if abs(gradient) < epsilon:
            print(f"Converged after {step} steps")
            break

        alpha = learning_rate / (1 + step)
        mu -= alpha * gradient

    print(f"Optimal mu: {mu}. Loss: {loss_fn(mu)}.")  # ~0.74
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image", [1, 3, 1])
    imageplot(y, "Noisy image", [1, 3, 2])
    imageplot(denoise(y, mu), "Optimally Denoised image", [1, 3, 3])

    plt.show()
    return f, y, denoise(y, mu)


def non_linear_denoising(f: np.ndarray, y: np.ndarray, f_denoised_linear: np.ndarray):
    def soft_threshold(T: float, alpha: np.ndarray):
        return (
            np.max(
                [
                    np.zeros_like(alpha),
                    np.ones_like(alpha) - T * np.ones_like(alpha) / np.abs(alpha),
                ]
            )
            * alpha
        )

    Jmin = 0
    Fw = perform_wavelet_transf(f, Jmin, +1)  # forward transform
    T = 0
    Fw_thresholded = soft_threshold(T, Fw)
    f_wav = perform_wavelet_transf(Fw_thresholded, Jmin, -1)  # inverse transform
    print(f"Loss of non-linear denoising: {np.linalg.norm(f_wav - f)}")  # ~2e-13
    print(f"SNR of linear denoising: {snr(f, f_denoised_linear)}")  # ~23
    print(f"SNR of non-linear denoising: {snr(f, f_wav)}")  # ~297
    plt.figure(figsize=(5, 5))
    imageplot(f, "Image", [2, 2, 1])
    imageplot(y, "Noisy image", [2, 2, 2])
    imageplot(f_denoised_linear, "Denoised image (Linear)", [2, 2, 3])
    imageplot(f_wav, "Denoised image (Non-linear)", [2, 2, 4])

    plt.show()


def main():
    f = plot_image()
    f_lpf = filtering(f)
    fourier_vs_wavelet(f, f_lpf)
    f, y, f_denoised_linear = linear_denoising()
    non_linear_denoising(f, y, f_denoised_linear)


if __name__ == "__main__":
    main()
