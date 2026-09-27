import numpy as np
import matplotlib.pyplot as plt

fc = 3.5e9
c = 3e8
lambda_c = c / fc
N = 20
d_2d = 15.0
h_tx = 3.0
h_rx = 1.0
v_rx_kmh = 3.0
v_rx_ms = v_rx_kmh / 3.6
angulo_movimento = np.radians(45)
vetor_v_rx = np.array([np.cos(angulo_movimento), np.sin(angulo_movimento), 0]) * v_rx_ms

if d_2d <= 1.2:
    prob_los = 1.0
elif d_2d < 6.5:
    prob_los = np.exp(-(d_2d - 1.2) / 4.7)
else:
    prob_los = np.exp(-(d_2d - 6.5) / 32.6) * 0.32

is_los = np.random.rand() < prob_los

if is_los:
    mu_log_DS = -7.69; sigma_log_DS = 0.18
    mu_log_AoA = 1.62; sigma_log_AoA = 0.25
    mu_log_EoA = 1.08; sigma_log_EoA = 0.36
    K_R_dB_mean = 7.0; K_R_dB_std = 4.0
    r_tau = 3.6
    sigma_xi = 3.0
else:
    mu_log_DS = -7.53; sigma_log_DS = 0.24
    mu_log_AoA = 1.77; sigma_log_AoA = 0.16
    mu_log_EoA = 1.15; sigma_log_EoA = 0.43
    K_R_dB_mean = -np.inf; K_R_dB_std = 0.0
    r_tau = 3.0
    sigma_xi = 6.0

log_sigma_tau = np.random.normal(mu_log_DS, sigma_log_DS)
log_sigma_phi_AoA = np.random.normal(mu_log_AoA, sigma_log_AoA)
log_sigma_theta_AoA = np.random.normal(mu_log_EoA, sigma_log_EoA)

sigma_tau = 10**(log_sigma_tau)
sigma_phi_AoA = np.minimum(10**(log_sigma_phi_AoA), 104.0)
sigma_theta_AoA = np.minimum(10**(log_sigma_theta_AoA), 52.0)

mu_tau = r_tau * sigma_tau
tau_n_puro = np.random.exponential(scale=mu_tau, size=N)
tau_n = np.sort(tau_n_puro - np.min(tau_n_puro))

xi_n = np.random.normal(0, sigma_xi, N)
alpha_n_hat_sq = np.exp(-tau_n * (r_tau - 1) / (r_tau * sigma_tau)) * (10 ** (-xi_n / 10))
alpha_n_sq = np.zeros(N)

if is_los:
    K_R_linear = 10 ** (np.random.normal(K_R_dB_mean, K_R_dB_std) / 10)
    omega_c = np.sum(alpha_n_hat_sq[1:])
    alpha_n_sq[0] = K_R_linear / (K_R_linear + 1)
    alpha_n_sq[1:] = (1 / (K_R_linear + 1)) * (alpha_n_hat_sq[1:] / omega_c)
else:
    alpha_n_sq = alpha_n_hat_sq / np.sum(alpha_n_hat_sq)

phi_LoS_linha = 0.0
theta_LoS_linha = np.degrees(np.arctan((h_tx - h_rx) / d_2d))
max_alpha_sq = np.max(alpha_n_sq)

def gerar_angulos_azimute(alpha, max_a, sigma, phi_los):
    phi_n_2linhas = 1.42 * sigma * np.sqrt(-np.log(alpha / max_a))
    U_n = np.random.choice([-1, 1], size=N)
    Y_n = np.random.normal(0, sigma / 7.0, size=N)
    return U_n * phi_n_2linhas + Y_n + phi_los

def gerar_angulos_elevacao(alpha, max_a, sigma, theta_los):
    theta_n_2linhas = -sigma * np.log(alpha / max_a)
    U_n = np.random.choice([-1, 1], size=N)
    Y_n = np.random.normal(0, sigma / 7.0, size=N)
    return U_n * theta_n_2linhas + Y_n + theta_los

ang_azimute = gerar_angulos_azimute(alpha_n_sq, max_alpha_sq, sigma_phi_AoA, phi_LoS_linha)
ang_elevacao = gerar_angulos_elevacao(alpha_n_sq, max_alpha_sq, sigma_theta_AoA, theta_LoS_linha)

if is_los:
    ang_azimute[0] = phi_LoS_linha
    ang_elevacao[0] = theta_LoS_linha

azim_rad = np.radians(ang_azimute)
elev_rad = np.radians(ang_elevacao)

r_n_x = np.cos(elev_rad) * np.cos(azim_rad)
r_n_y = np.cos(elev_rad) * np.sin(azim_rad)
r_n_z = np.sin(elev_rad)
r_n_matriz = np.column_stack((r_n_x, r_n_y, r_n_z))


doppler_n = np.dot(r_n_matriz, vetor_v_rx) / lambda_c

t_macro = np.linspace(0, 0.01, 1000)
fase_constante = -2 * np.pi * (fc + doppler_n) * tau_n
fase_n_t = 2 * np.pi * np.outer(doppler_n, t_macro) + fase_constante[:, np.newaxis]

alpha_n = np.sqrt(alpha_n_sq)
h_n_t = alpha_n[:, np.newaxis] * np.exp(1j * fase_n_t)

fs = 1e9
t_micro = np.arange(0, 150e-9, 1/fs)
pulso_tx = np.zeros_like(t_micro)
pulso_tx[10:20] = 1.0

sinal_rx = np.zeros_like(t_micro, dtype=complex)
h_estatico = h_n_t[:, 0]

for i in range(N):
    atraso_amostras = int(tau_n[i] * fs)
    pulso_atrasado = np.zeros_like(t_micro)
    if atraso_amostras < len(t_micro):
        pulso_atrasado[atraso_amostras:] = pulso_tx[:len(t_micro)-atraso_amostras]
    sinal_rx += h_estatico[i] * pulso_atrasado

Omega_c = np.sum(alpha_n_sq)

def rho_TT(kappa, sigma_t):
    soma = 0
    for n in range(N):
        termo_freq = np.exp(-1j * 2 * np.pi * tau_n[n] * kappa)
        termo_tempo = np.exp(1j * 2 * np.pi * doppler_n[n] * sigma_t)
        soma += alpha_n_sq[n] * termo_freq * termo_tempo
    return soma / Omega_c

rho_B = 0.5
kappas = np.linspace(0, 50e6, 5000)
rho_freq = np.abs(rho_TT(kappas, 0))
idx_bc = np.where(rho_freq < rho_B)[0]
banda_coerencia = kappas[idx_bc[0]] if len(idx_bc) > 0 else float('inf')

rho_T = 0.5
sigmas = np.linspace(0, 0.5, 5000)
rho_tempo = np.abs(rho_TT(0, sigmas))
idx_tc = np.where(rho_tempo < rho_T)[0]
tempo_coerencia = sigmas[idx_tc[0]] if len(idx_tc) > 0 else float('inf')

tau_medio = np.sum(alpha_n_sq * tau_n) / Omega_c
tau_rms = np.sqrt(np.sum(alpha_n_sq * (tau_n - tau_medio)**2) / Omega_c)
doppler_maximo = np.max(np.abs(doppler_n))


# EXIBIÇÃO DE RESULTADOS E GRÁFICOS

print("="*75)
print(f"CENÁRIO INH-OFFICE | Distância: {d_2d}m")
print(f"Probabilidade base calculada -> LoS: {prob_los*100:.1f}% | NLoS: {(1-prob_los)*100:.1f}%")
print(f"Condição SORTEADA para esta simulação: {'LoS (Com Visada)' if is_los else 'NLoS (Sem Visada)'}")
print("="*75)
print(f"{'Raio':<5} | {'Atraso(ns)':<10} | {'Potência':<10} | {'Azimute(º)':<10} | {'Elevação(º)':<12} | {'Doppler(Hz)':<10}")
print("-" * 75)
for i in range(5):
    print(f"{i+1:<5} | {tau_n[i]*1e9:<10.2f} | {alpha_n_sq[i]:<10.4f} | {np.degrees(azim_rad[i]):<10.2f} | {np.degrees(elev_rad[i]):<12.2f} | {doppler_n[i]:<10.2f}")

print("\n" + "="*75)
print("MÉTRICAS DE COERÊNCIA (Eq. 13 - Slide 37)")
print("="*75)
print(f"Espalhamento de Atraso RMS: {tau_rms*1e9:.2f} ns")
print(f"Desvio Doppler Máximo: {doppler_maximo:.2f} Hz")
print(f"Banda de Coerência (Limiar 0.5): {banda_coerencia / 1e6:.2f} MHz")
print(f"Tempo de Coerência (Limiar 0.5): {tempo_coerencia * 1000:.2f} ms")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(tau_n * 1e9, alpha_n_sq, basefmt=" ")
plt.title("Perfil de Atraso de Potência (PDP)")
plt.xlabel("Atraso (ns)")
plt.ylabel("Potência Linear Normalizada")
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(t_micro * 1e9, pulso_tx, label="Sinal Transmitido", color="blue", linewidth=2)
plt.plot(t_micro * 1e9, np.abs(sinal_rx), label="Sinal Recebido", color="red", linewidth=2)
plt.title("Espalhamento Temporal do Sinal")
plt.xlabel("Tempo (ns)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

fig_3d = plt.figure(figsize=(8, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

for i in range(1, N):
    ax_3d.quiver(0, 0, 0, r_n_x[i], r_n_y[i], r_n_z[i], color='r', arrow_length_ratio=0.1)
ax_3d.quiver(0, 0, 0, r_n_x[0], r_n_y[0], r_n_z[0], color='b', arrow_length_ratio=0.1, linewidth=2)

ax_3d.set_xlim([-1, 1])
ax_3d.set_ylim([-1, 1])
ax_3d.set_zlim([-1, 1])
ax_3d.set_xlabel('Eixo X')
ax_3d.set_ylabel('Eixo Y')
ax_3d.set_zlabel('Eixo Z')
ax_3d.set_title("Direções de Chegada")

plt.show()