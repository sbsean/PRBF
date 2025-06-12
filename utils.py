import numpy as np

# probe
def init_config():
    cfg = {}
    cfg["GRID_SIZE_X"] = 1024   # number of pixels in lateral direction
    cfg["GRID_SIZE_Z"] = 1024  # number of pixels in axial (depth) direction

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

    cfg["tx_offset"] = 13
    print(f"Pixel Shape: {cfg['GRID_SIZE_X']}x{cfg['GRID_SIZE_Z']}")
    return cfg


#50mm, RF_data_1
def init_config_2():
    cfg = {}
    cfg["GRID_SIZE_X"] = 1024   # number of pixels in lateral direction (기존 유지)
    cfg["GRID_SIZE_Z"] = 1024  # number of pixels in axial (depth) direction (기존 유지)

    # Acoustic & system parameters (MATLAB 기반으로 수정)
    cfg["C"] = 1450            # speed of sound [m/s] (수정)
    cfg["Fc"] = 5.2083e6       # transducer center frequency [Hz] (수정)
    cfg["Fs"] = 4 * cfg["Fc"]  # sampling frequency [Hz]
    cfg["Lambda"] = cfg["C"] / cfg["Fc"]  # wavelength [m]
    cfg["Elem_height"] = 4e-3  # element height [m] (기존 유지)
    cfg["Elem_pitch"] = 298e-6 # element pitch (center-to-center spacing) [m] (수정)
    cfg["Elem_width"] = 0.95 * cfg["Elem_pitch"]  # element width [m]
    cfg["Kerf"] = 0.05 * cfg["Elem_pitch"]        # element spacing (kerf) [m]
    cfg["N_element"] = 128    # total number of transducer elements
    cfg["N_TX_element"] = 128 # number of transmit elements
    cfg["N_RX_element"] = 128 # number of receive elements
    cfg["N_scanline"] = 128   # number of scanlines for beamforming
    cfg["View_angle"] = 0     # steering angle (radian)
    cfg["View_width"] = cfg["Elem_pitch"] * cfg["N_element"]  # imaging width [m]
    cfg["El_focus"] = 30e-3   # elevation focus [m] (기존 유지)
    cfg["TX_focus"] = 30e-3   # transmit focus [m] (기존 유지)
    cfg["RX_focus"] = 10e20   # receive focus (infinity means no focusing) [m] (기존 유지)
    cfg["Impulse_Fc"] = 8e6   # center frequency of impulse response [Hz] (기존 유지)
    cfg["Impulse_cycle"] = 3.2  # impulse response cycles (기존 유지)
    cfg["TX_pulse_cycle"] = 1.0 # transmit pulse cycles (기존 유지)
    cfg["depth"] = 70e-3      # maximum imaging depth [m] (수정)
    cfg["N_image_point"] = 1536  # number of image points for delay calculation (수정)
    cfg["Unit_Dis"] = cfg["C"] / cfg["Fs"]  # distance per sample [m]
    cfg["Dynamic_Range"] = 60  # dynamic range for log compression [dB]
    cfg["F_Number_limit_RX"] = 1.5  # maximum receive F-number (수정)
    cfg["eps"] = 1e-10         # epsilon for numerical stability

    cfg['tx_offset'] = 70  # transmit offset to be considered while loading RF data (수정)

    print(f"Pixel Shape: {cfg['GRID_SIZE_X']}x{cfg['GRID_SIZE_Z']}")
    return cfg

def init_config_3():
    cfg = {}
    cfg["GRID_SIZE_X"] = 512   # number of pixels in lateral direction
    cfg["GRID_SIZE_Z"] = 512   # number of pixels in axial (depth) direction

    # Acoustic & system parameters (최신 파라미터 반영)
    cfg["C"] = 1540            # speed of sound [m/s]
    cfg["Fc"] = 5e6            # transducer center frequency [Hz]
    cfg["Fs"] = 40000000       # sampling frequency [Hz]
    cfg["Lambda"] = 3.0800e-4  # wavelength [m] (지정된 값 직접 사용)
    
    cfg["Elem_height"] = 4e-3  # element height [m]
    cfg["Elem_pitch"] = 0.3e-3 # element pitch (center-to-center spacing) [m]
    cfg["Elem_width"] = 0.95 * cfg["Elem_pitch"]  # element width [m]
    cfg["Kerf"] = 0.05 * cfg["Elem_pitch"]        # element spacing (kerf) [m]

    cfg["N_element"] = 128    # total number of transducer elements
    cfg["N_TX_element"] = 128 # number of transmit elements
    cfg["N_RX_element"] = 128 # number of receive elements
    cfg["N_scanline"] = 128   # number of scanlines for beamforming

    cfg["View_angle"] = 0     # steering angle (radian)
    cfg["View_width"] = 0.0384  # imaging width [m] (지정된 값 사용)

    cfg["El_focus"] = 30e-3   # elevation focus [m]
    cfg["TX_focus"] = 30e-3   # transmit focus [m]
    cfg["RX_focus"] = 10e20   # receive focus (infinity means no focusing) [m]

    cfg["Impulse_Fc"] = 8e6   # center frequency of impulse response [Hz]
    cfg["Impulse_cycle"] = 3.2  # impulse response cycles
    cfg["TX_pulse_cycle"] = 1.0 # transmit pulse cycles

    cfg["depth"] = 0.0600     # maximum imaging depth [m]
    cfg["N_image_point"] = 3117  # number of image points for delay calculation

    cfg["Unit_Dis"] = 3.8500e-5  # distance per sample [m]
    cfg["Dynamic_Range"] = 60  # dynamic range for log compression [dB]
    cfg["F_Number_limit_RX"] = 1.5  # maximum receive F-number
    cfg["eps"] = 1e-10         # epsilon for numerical stability

    cfg['tx_offset'] = 65     # transmit offset to be considered while loading RF data

    print(f"Pixel Shape: {cfg['GRID_SIZE_X']}x{cfg['GRID_SIZE_Z']}")
    return cfg



# load rf data (linear focused)
def load_rf_data(scanline_idx, cfg):
    """
    Load RF data with tx_offset consideration.
    """
    file_name = f'C:/Users/user/workspace/deepcoherenceLearning_HSB/deepcoherence_code/datasets/seongbindclcode/RF_data/scanline_phantom{scanline_idx+1:03d}.bin'
    N_RX_element = cfg["N_RX_element"]
    N_image_point = cfg["N_image_point"]
    tx_offset = cfg["tx_offset"]

    try:
        with open(file_name, 'rb') as fid:
            rf_data = np.fromfile(fid, dtype=np.float64)
        rf_data_2D = rf_data.reshape((len(rf_data) // N_RX_element, N_RX_element), order='F')
        
        # Apply tx_offset: skip first tx_offset samples
        if rf_data_2D.shape[0] > tx_offset:
            rf_data_offset = rf_data_2D[tx_offset:, :]
        else:
            # If data is too short, return all zeros
            rf_data_offset = np.zeros((N_image_point, N_RX_element))

        # Pad or crop to match N_image_point
        if rf_data_offset.shape[0] < N_image_point:
            pad_width = ((0, N_image_point - rf_data_offset.shape[0]), (0, 0))
            RF_data = np.pad(rf_data_offset, pad_width, mode='constant')
        else:
            RF_data = rf_data_offset[:N_image_point, :]

        RF_data = RF_data.T
        return RF_data
    except FileNotFoundError:
        print(f'Cannot open file: {file_name}')
        return None
    

import scipy.io
# load RF_data_50mm
def load_mat_data(scanline_idx, cfg):
    """
    Load RF data from .mat file with tx_offset consideration.
    """
    file_name = f'C:/Users/user/workspace/IqPixelGrid_HSB/RF_data_50mm/RF_data_{scanline_idx+1:03d}.mat'
    N_RX_element = cfg["N_RX_element"]
    N_image_point = cfg["N_image_point"]
    tx_offset = cfg["tx_offset"]

    try:
        # .mat 파일 읽기
        mat_data = scipy.io.loadmat(file_name)

        # 실제 데이터는 'RF_data_align' 키에 있음
        if 'RF_data_align' in mat_data:
            rf_data_2D = mat_data['RF_data_align']  # shape: (N_samples, N_RX_element)
        else:
            print(f"'RF_data_align' key not found in {file_name}")
            return None

        # Apply tx_offset
        if rf_data_2D.shape[0] > tx_offset:
            rf_data_offset = rf_data_2D[tx_offset:, :]
        else:
            rf_data_offset = np.zeros((N_image_point, N_RX_element))

        # Pad or crop
        if rf_data_offset.shape[0] < N_image_point:
            pad_width = ((0, N_image_point - rf_data_offset.shape[0]), (0, 0))
            RF_data = np.pad(rf_data_offset, pad_width, mode='constant')
        else:
            RF_data = rf_data_offset[:N_image_point, :]

        # Transpose to shape: (N_RX_element, N_image_point)
        RF_data = RF_data.T
        return RF_data

    except FileNotFoundError:
        print(f'Cannot open file: {file_name}')
        return None
    
# load RF_data_1
def load_mat_data_2(scanline_idx, cfg):
    """
    Load RF data from .mat file with tx_offset consideration.
    """
    file_name = f'C:/Users/user/workspace/IqPixelGrid_HSB/RF_data_1/Tx_{scanline_idx+1:03d}.mat'
    N_RX_element = cfg["N_RX_element"]
    N_image_point = cfg["N_image_point"]
    tx_offset = cfg["tx_offset"]

    try:
        # .mat 파일 읽기
        mat_data = scipy.io.loadmat(file_name)

        # 실제 데이터는 'RF_data_align' 키에 있음
        if 'RF_data_align' in mat_data:
            rf_data_2D = mat_data['RF_data_align']  # shape: (N_samples, N_RX_element)
        else:
            print(f"'RF_data_align' key not found in {file_name}")
            return None

        # Apply tx_offset
        if rf_data_2D.shape[0] > tx_offset:
            rf_data_offset = rf_data_2D[tx_offset:, :]
        else:
            rf_data_offset = np.zeros((N_image_point, N_RX_element))

        # Pad or crop
        if rf_data_offset.shape[0] < N_image_point:
            pad_width = ((0, N_image_point - rf_data_offset.shape[0]), (0, 0))
            RF_data = np.pad(rf_data_offset, pad_width, mode='constant')
        else:
            RF_data = rf_data_offset[:N_image_point, :]

        # Transpose to shape: (N_RX_element, N_image_point)
        RF_data = RF_data.T
        return RF_data

    except FileNotFoundError:
        print(f'Cannot open file: {file_name}')
        return None
    

# load RcvData
def load_mat_data_3(scanline_idx, cfg):
    """
    Load RF data from .mat file with tx_offset consideration and normalization.
    Used for beamforming stage (visualization oriented).
    """
    file_name = f'C:/Users/user/workspace/IqPixelGrid_HSB/RcvData/rfData_{scanline_idx+1:03d}.mat'
    N_RX_element = cfg["N_RX_element"]
    N_image_point = cfg["N_image_point"]
    tx_offset = cfg["tx_offset"]

    try:
        mat_data = scipy.io.loadmat(file_name)

        if 'rfTemp' in mat_data:
            rf_data_2D = mat_data['rfTemp']  # shape: (N_samples, N_RX_element)
            # ✅ Normalization only for beamforming visualization
            rf_data_2D /= np.max(np.abs(rf_data_2D) + 1e-20)
        else:
            print(f"'rfTemp' key not found in {file_name}")
            return None

        # Apply tx_offset
        if rf_data_2D.shape[0] > tx_offset:
            rf_data_offset = rf_data_2D[tx_offset:, :]
        else:
            rf_data_offset = np.zeros((N_image_point, N_RX_element))

        # Pad or crop
        if rf_data_offset.shape[0] < N_image_point:
            pad_width = ((0, N_image_point - rf_data_offset.shape[0]), (0, 0))
            RF_data = np.pad(rf_data_offset, pad_width, mode='constant')
        else:
            RF_data = rf_data_offset[:N_image_point, :]

        RF_data = RF_data.T  # (N_RX_element, N_image_point)
        return RF_data

    except FileNotFoundError:
        print(f'Cannot open file: {file_name}')
        return None


# window function
def hamming(N):
    if N <= 1:
        return np.ones(1)
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def rectwin(N):
    if N <= 1:
        return np.ones(1)
    return np.ones(N)


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

