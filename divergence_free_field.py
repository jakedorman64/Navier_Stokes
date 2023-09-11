"""This code is provided by Niall Jeffrey to generate a 2D Divergence Free Field."""

import numpy as np

# This is the function that generates our divergence free vector field for the velocity.

def k2g_fft(kE, kB, dx, pad=True):
    """
    Convert kappa to gamma in Fourier space. If padding is
    set to True, include the same size of padding as the data
    on each side, the total grid is 9 times the original.
    """

    if pad:
        kE_temp = np.zeros((len(kE)*3, len(kE[0])*3))
        kB_temp = np.zeros((len(kB)*3, len(kE[0])*3))
        kE_temp[len(kE):len(kE)*2, len(kE[0]):len(kE[0])*2] = kE*1.0
        kB_temp[len(kB):len(kB)*2, len(kB[0]):len(kB[0])*2] = kB*1.0
        kE_3d_ft = np.fft.fft2(kE_temp)
        kB_3d_ft = np.fft.fft2(kB_temp)
    else:
        kE_3d_ft = np.fft.fft2(kE)
        kB_3d_ft = np.fft.fft2(kB)
   
    FF1 = np.fft.fftfreq(len(kE_3d_ft))
    FF2 = np.fft.fftfreq(len(kE_3d_ft[0]))

    dk = 1.0/dx*2*np.pi                     # max delta_k in 1/arcmin
    kx = np.dstack(np.meshgrid(FF2, FF1))[:,:,0]*dk
    ky = np.dstack(np.meshgrid(FF2, FF1))[:,:,1]*dk
    kx2 = kx**2
    ky2 = ky**2
    k2 = kx2 + ky2

    k2[k2==0] = 1e-15
    k2gamma1_ft = kE_3d_ft/k2*(kx2-ky2) - kB_3d_ft/k2*2*(kx*ky)
    k2gamma2_ft = kE_3d_ft/k2*2*(kx*ky) + kB_3d_ft/k2*(kx2-ky2)

    if pad:
        return np.fft.ifft2(k2gamma1_ft).real[len(kE):len(kE)*2, len(kE[0]):len(kE[0])*2], np.fft.ifft2(k2gamma2_ft).real[len(kE):len(kE)*2, len(kE[0]):len(kE[0])*2]
    else:
        return np.fft.ifft2(k2gamma1_ft).real, np.fft.ifft2(k2gamma2_ft).real
    
def compute_spectrum_map(power1d,size):
    """
    takes 1D power spectrum and makes it an isotropic 2D map
    :param power: 1d power spectrum
    :param size:
    :return:
    """

    power_map = np.zeros((size, size), dtype = float)
    k_map =  np.zeros((size, size), dtype = float)

    for (i,j), val in np.ndenumerate(power_map):

        k1 = i - size/2.0
        k2 = j - size/2.0
        k_map[i, j] = (np.sqrt(k1*k1 + k2*k2))

        if k_map[i,j] == 0:
            power_map[i, j] = 1e-15
        else:
            power_map[i, j] = power1d[int(k_map[i, j])]

    return power_map



def gaussian_mock(spectrum_map_flat_sqrt,image_size):
    gaussian_field = np.random.normal(0, spectrum_map_flat_sqrt) + 1j*np.random.normal(0, spectrum_map_flat_sqrt)
    gaussian_field = np.fft.ifft2(np.fft.fftshift(gaussian_field.reshape((image_size,image_size)))).imag
    return gaussian_field
     