#드롭박스 매트랩 코드 활용
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from tqdm import tqdm 
from utils import init_config_2, load_mat_data_2, hamming
from PixelGrid import make_pixel_grid

cfg = init_config_2()

# 트랜스듀서 엘리먼트 위치
Elem_x_p = np.arange(0, cfg["N_RX_element"] // 2) * cfg["Elem_pitch"] + cfg["Elem_pitch"] / 2
Elem_x_n = np.arange(-1 * (cfg["N_RX_element"] // 2 - 1), 1) * cfg["Elem_pitch"] - cfg["Elem_pitch"] / 2
Elem_x = np.concatenate((Elem_x_n, Elem_x_p))
Elem_z = np.zeros(cfg["N_RX_element"])
Elem_y = np.zeros(cfg["N_RX_element"])
elem_pos = np.stack((Elem_x, Elem_y, Elem_z), axis=1)

# 이미지 포인트 좌표 
k = np.arange(0, cfg["N_image_point"])
Incr_z = cfg["Unit_Dis"]
Image_x = 0 + (0 * k / 2)
Image_z = 0 + (Incr_z * k / 2)

# make pixel grid
xlims = [-cfg["View_width"] / 2, cfg["View_width"] / 2]
zlims = [5e-3, cfg["depth"]]
pixel_grid, dx, dz = make_pixel_grid(xlims, zlims, cfg["GRID_SIZE_X"], cfg["GRID_SIZE_Z"])


# Delay 계산
Addr_4f0 = np.zeros((cfg["N_RX_element"], cfg["N_image_point"]))
Addr_16f0 = np.zeros((cfg["N_RX_element"], cfg["N_image_point"]))
fine_delay = np.zeros((cfg["N_RX_element"], cfg["N_image_point"]))

for j in range(cfg["N_image_point"]):
    for i in range(cfg["N_RX_element"]):
        Addr_16f0[i, j] = np.floor(
            (np.sqrt((Image_x[j] - Elem_x[i]) ** 2 + (Image_z[j] - Elem_z[i]) ** 2) +
             np.sqrt(Image_x[j] ** 2 + Image_z[j] ** 2)) * cfg["Fs"] / cfg["C"] * 4)

Addr_16f0 = np.minimum(Addr_16f0, cfg["N_image_point"] * 4)
Addr_4f0 = np.floor(Addr_16f0 / 4)
fine_delay = np.floor(Addr_16f0) / 4 - np.floor(Addr_16f0 / 4)
Addr_4f0_phase = np.mod(Addr_4f0, 4)
Addr_total_4f0 = Addr_4f0_phase + fine_delay
print('Delay Calculation Complete\n')

# Apodization 맵 생성 
Ch_Num_active = np.zeros(cfg["N_image_point"])
F_number = np.zeros(cfg["N_image_point"])

DOF = np.zeros(cfg["N_image_point"])

for depth in range(cfg["N_image_point"]):
    Ch_Num_active[depth] = np.floor(((depth + 1) * (cfg["Unit_Dis"] / 2)) / (cfg["Elem_pitch"]) * cfg["F_Number_limit_RX"])
    Ch_Num_active[depth] = min(Ch_Num_active[depth], cfg["N_RX_element"])
    F_number[depth] = ((depth + 1) * cfg["Unit_Dis"]) / (cfg["Elem_pitch"] * Ch_Num_active[depth]) if Ch_Num_active[depth] > 0 else 0
    DOF[depth] = 7.2 * cfg["Lambda"] * F_number[depth] ** 2

aperture_mark_half = np.ones((cfg["N_RX_element"] // 2, cfg["N_image_point"]))
for depth in range(cfg["N_image_point"]):
    for ii in range(cfg["N_RX_element"] // 2):
        if Ch_Num_active[depth] < ii + 1:
            aperture_mark_half[ii, depth] = 0

aperture_mark = np.vstack((np.flipud(aperture_mark_half), aperture_mark_half))
apodization_coef = np.zeros((cfg["N_RX_element"], cfg["N_image_point"]))
for depth in range(cfg["N_image_point"]):
    active_elements = int(np.sum(aperture_mark[:, depth]))
    if active_elements > 0:
        apo = hamming(active_elements)
        blank = (cfg["N_RX_element"] - active_elements) // 2
        apodization_coef[:, depth] = np.concatenate([np.zeros(blank), apo, np.zeros(blank)])



beamformed_I = np.zeros((cfg["GRID_SIZE_X"], cfg["GRID_SIZE_Z"]))
beamformed_Q = np.zeros((cfg["GRID_SIZE_X"], cfg["GRID_SIZE_Z"]))
output_dir = f'IQBF_RF_data_1_{cfg["GRID_SIZE_X"]}x{cfg["GRID_SIZE_Z"]}'
os.makedirs(output_dir, exist_ok=True)
scanline_x = np.linspace(xlims[0], xlims[1], cfg["N_scanline"])


for scanline_idx in tqdm(range(cfg["N_scanline"]), desc="Beamforming scanlines"):
    # RF_data = load_rf_data(scanline_idx, cfg["N_RX_element"], cfg["N_image_point"])
    RF_data = load_mat_data_2(scanline_idx, cfg)
    if RF_data is None:
        continue

    DC_coef = signal.firwin(17, 0.1, pass_zero='highpass')# matlab 16 --> 17
    DC_Cancel_out = np.array([signal.convolve(RF_data[i], DC_coef, mode='same') for i in range(cfg["N_RX_element"])])

    t = np.arange(1, cfg["N_image_point"] + 1)
    Cos_t = np.cos(t * 2 * np.pi * cfg["Fc"] / cfg["Fs"])
    Sin_t = np.sin(t * 2 * np.pi * cfg["Fc"] / cfg["Fs"])
    Data_I = DC_Cancel_out * Cos_t
    Data_Q = DC_Cancel_out * Sin_t
    #FIR 필터에서 일반적으로 사용하는 중간 품질 수준의 길이 : 53
    LPF_coef = signal.firwin(53, cfg["Fc"] / cfg["Fs"])
    QDM_Inph = np.array([signal.convolve(Data_I[i], LPF_coef, mode='same') * 4.5 for i in range(cfg["N_RX_element"])])
    QDM_Quad = np.array([signal.convolve(Data_Q[i], LPF_coef, mode='same') * 4.5 for i in range(cfg["N_RX_element"])])

    Data_buffer_Inph = np.zeros_like(QDM_Inph)
    Data_buffer_Quad = np.zeros_like(QDM_Quad)
    for j in range(cfg["N_image_point"]):
        for i in range(cfg["N_RX_element"]):
            addr = int(Addr_4f0[i, j])
            if 0 <= addr < cfg["N_image_point"]:
                Data_buffer_Inph[i, j] = QDM_Inph[i, addr] * apodization_coef[i, j]
                Data_buffer_Quad[i, j] = QDM_Quad[i, addr] * apodization_coef[i, j]

    Cos_pr = np.cos(2 * np.pi * cfg["Fc"] / cfg["Fs"] * Addr_total_4f0)
    Sin_pr = np.sin(2 * np.pi * cfg["Fc"] / cfg["Fs"] * Addr_total_4f0)
    PR_Inph = Data_buffer_Inph * Cos_pr - Data_buffer_Quad * (-Sin_pr)
    PR_Quad = Data_buffer_Inph * (-Sin_pr) + Data_buffer_Quad * Cos_pr

    Sum_out_I = np.sum(PR_Inph, axis=0)
    Sum_out_Q = np.sum(PR_Quad, axis=0)

    x_mapped = int((scanline_x[scanline_idx] - xlims[0]) / (xlims[1] - xlims[0]) * (cfg["GRID_SIZE_X"] - 1))
    x_mapped = max(0, min(cfg["GRID_SIZE_X"] - 1, x_mapped))
    z_scale = cfg["N_image_point"] / cfg["GRID_SIZE_Z"]
    for z_idx in range(cfg["GRID_SIZE_Z"]):
        orig_z_idx = z_idx * z_scale
        z1, z2 = int(np.floor(orig_z_idx)), int(np.ceil(orig_z_idx))
        z1 = min(z1, cfg["N_image_point"] - 1)
        z2 = min(z2, cfg["N_image_point"] - 1)
        weight = orig_z_idx - z1 if z1 != z2 else 0
        interp_I = Sum_out_I[z1] * (1 - weight) + Sum_out_I[z2] * weight
        interp_Q = Sum_out_Q[z1] * (1 - weight) + Sum_out_Q[z2] * weight
        beamformed_I[x_mapped, z_idx] = interp_I
        beamformed_Q[x_mapped, z_idx] = interp_Q

# x 방향 보간
for z_idx in range(cfg["GRID_SIZE_Z"]):
    valid_x = np.nonzero(beamformed_I[:, z_idx])[0]
    if len(valid_x) > 1:
        beamformed_I[:, z_idx] = np.interp(np.arange(cfg["GRID_SIZE_X"]), valid_x, beamformed_I[valid_x, z_idx])
        beamformed_Q[:, z_idx] = np.interp(np.arange(cfg["GRID_SIZE_X"]), valid_x, beamformed_Q[valid_x, z_idx])

# 저장
np.save(f'{output_dir}/bf_I_real.npy', beamformed_I)
np.save(f'{output_dir}/bf_Q_image.npy', beamformed_Q)
print("Beamforming Complete\n")

# Envelope 및 시각화
envelope = np.sqrt(beamformed_I ** 2 + beamformed_Q ** 2)
envelope_safe = np.maximum(envelope, cfg["eps"])
log_env = 20 * np.log10(envelope_safe / np.max(envelope_safe))
log_env_normalized = np.clip(log_env, -cfg["Dynamic_Range"], 0)
log_env_normalized = (log_env_normalized + cfg["Dynamic_Range"]) / cfg["Dynamic_Range"]
log_env_normalized_255 = log_env_normalized * 255

extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
plt.figure(figsize=(5, 5))
plt.imshow(log_env_normalized_255.T, aspect='auto', cmap='gray', extent=extent, interpolation='none')
plt.title(f'Output Image')
plt.xlabel(f'Lateral position')
plt.ylabel(f'Depth')
plt.colorbar()
plt.clim(0, 255)
plt.tight_layout()
plt.show()