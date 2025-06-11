import numpy as np

# probe
def init_config():
    cfg = {}
    cfg["GRID_SIZE_X"] = 512    # number of pixels in lateral direction
    cfg["GRID_SIZE_Z"] = 512    # number of pixels in axial (depth) direction

    # Acoustic & system parameters (from original MATLAB code)
    cfg["C"] = 1540            # speed of sound [m/s]
    cfg["Fc"] = 5.0e6          # transducer center frequency [Hz]
    cfg["Fs"] = 4 * cfg["Fc"]  # sampling frequency [Hz]
    cfg["Lambda"] = cfg["C"] / cfg["Fc"]  # wavelength [m]
    cfg["Elem_height"] = 4e-3  # element height [m]
    cfg["Elem_pitch"] = 298e-6 # element pitch (center-to-center spacing) [m]
    cfg["Elem_width"] = 0.95 * cfg["Elem_pitch"]  # element width [m]
    cfg["Kerf"] = 0.05 * cfg["Elem_pitch"]        # element spacing (kerf) [m]
    cfg["N_element"] = 128    # total number of transducer elements
    cfg["N_TX_element"] = 128 # number of transmit elements
    cfg["N_RX_element"] = 128 # number of receive elements
    cfg["N_scanline"] = 128   # number of scanlines for beamforming
    cfg["View_angle"] = 0     # steering angle (radian), 0 means no steering
    cfg["View_width"] = cfg["Elem_pitch"] * cfg["N_element"]  # imaging width [m]
    cfg["El_focus"] = 30e-3   # elevation focus [m]
    cfg["TX_focus"] = 30e-3   # transmit focus [m]
    cfg["RX_focus"] = 10e20   # receive focus (infinity means no focusing) [m]
    cfg["Impulse_Fc"] = 8e6   # center frequency of impulse response [Hz]
    cfg["Impulse_cycle"] = 3.2  # impulse response cycles
    cfg["TX_pulse_cycle"] = 1.0 # transmit pulse cycles
    cfg["depth"] = 70e-3      # maximum imaging depth [m]
    cfg["N_image_point"] = 1700  # number of image points for delay calculation
    cfg["Unit_Dis"] = cfg["C"] / cfg["Fs"]  # distance per sample [m]
    cfg["Dynamic_Range"] = 60  # dynamic range for log compression [dB]
    cfg["F_Number_limit_RX"] = 1.5  # maximum receive F-number
    cfg["eps"] = 1e-10         # epsilon for numerical stability
    print(f"Pixel Shape: {cfg['GRID_SIZE_X']}x{cfg['GRID_SIZE_Z']}")
    return cfg


# window function
def hamming(N):
    if N <= 1:
        return np.ones(1)
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def blackman(N):
    if N <= 1:
        return np.ones(1)
    n = np.arange(N)
    return 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

def bartlett(N):
    if N <= 1:
        return np.ones(1)
    n = np.arange(N)
    return 2 / (N - 1) * ((N - 1) / 2 - np.abs(n - (N - 1) / 2))

def kaiser(N, beta=14):
    if N <= 1:
        return np.ones(1)
    return np.kaiser(N, beta)

def hann(N):
    if N <= 1:
        return np.ones(1)
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))


# load rf data (linear focused)
def load_rf_data(scanline_idx, N_RX_element, N_image_point):
    file_name = f'C:/Users/user/workspace/deepcoherenceLearning_HSB/deepcoherence_code/datasets/seongbindclcode/RF_data/scanline_phantom{scanline_idx+1:03d}.bin'
    try:
        with open(file_name, 'rb') as fid:
            rf_data = np.fromfile(fid, dtype=np.float64)
        rf_data_2D = rf_data.reshape((len(rf_data) // N_RX_element, N_RX_element), order='F')
        if rf_data_2D.shape[0] < N_image_point:
            pad_width = ((0, N_image_point - rf_data_2D.shape[0]), (0, 0))
            RF_data = np.pad(rf_data_2D, pad_width, mode='constant')
        else:
            RF_data = rf_data_2D[:N_image_point, :]
        RF_data = RF_data.T
        return RF_data
    except FileNotFoundError:
        print(f'Cannot open file: {file_name}')
        return None



