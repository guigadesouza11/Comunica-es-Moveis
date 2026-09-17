import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, hyp1f1

def gerar_e_plotar_momentos():
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    N = 1000000  # Número de amostras
    n_values = np.arange(1, 11)
    colors = ['#1f77b4', '#ef9f22', '#65a525']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sigmas_rayleigh = [1, 3, 5]
    markers_added = False

    for i, sigma in enumerate(sigmas_rayleigh):
        x = np.random.normal(0, sigma, N)
        y = np.random.normal(0, sigma, N)
        R_sim = np.sqrt(x**2 + y**2)
        
        
        mom_sim = [np.mean(R_sim**n) for n in n_values]
        mom_teo = [(sigma**n) * (2**(n/2)) * gamma(1 + n/2) for n in n_values]
        
        ax1.plot(n_values, mom_teo, '-', color=colors[i], lw=2, label=rf'$\sigma = {sigma}$')
        if not markers_added:
            ax1.plot(n_values, mom_sim, 'kP', markersize=6, label='Simulação')
            markers_added = True
        else:
            ax1.plot(n_values, mom_sim, 'kP', markersize=6)

    ax1.set_yscale('log')
    ax1.set_title('Momentos de $n$-ésima Ordem')
    ax1.set_xlabel('Ordem ($n$)')
    ax1.set_ylabel(r'$\mathbf{E}[\beta^n]$')
    ax1.set_xlim(1, 10)
    ax1.set_ylim(1, 5*10**10)
    ax1.set_xticks(n_values)
    ax1.grid(True, which='major', color='#cccccc', linestyle='-')
    ax1.grid(True, which='minor', color='#eeeeee', linestyle=':')
    leg1 = ax1.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)
    leg1.get_frame().set_alpha(1)

    K_dB_values = [0, 5, 10]
    sigma_rice = 1.0
    markers_added = False

    for i, K_dB in enumerate(K_dB_values):
        K = 10**(K_dB / 10)
        A = np.sqrt(2 * K * sigma_rice**2)
        x = np.random.normal(A, sigma_rice, N)
        y = np.random.normal(0, sigma_rice, N)
        R_sim = np.sqrt(x**2 + y**2)
        
      
        mom_sim = [np.mean(R_sim**n) for n in n_values]
        mom_teo = [((2 * sigma_rice**2)**(n/2)) * gamma(1 + n/2) * np.exp(-K) * hyp1f1(1 + n/2, 1, K) for n in n_values]
        
 
        ax2.plot(n_values, mom_teo, '-', color=colors[i], lw=2, label=f'K = {K_dB} dB')
        if not markers_added:
            ax2.plot(n_values, mom_sim, 'kP', markersize=6, label='Simulação')
            markers_added = True
        else:
            ax2.plot(n_values, mom_sim, 'kP', markersize=6)

    ax2.set_yscale('log')
    ax2.set_title('Momentos de $n$-ésima Ordem (Rice, $\sigma=1.0$)')
    ax2.set_xlabel('Ordem ($n$)')
    ax2.set_ylabel(r'$\mathbf{E}[\beta^n]$')
    ax2.set_xlim(1, 10)
    ax2.set_xticks(n_values)
    ax2.grid(True, which='major', color='#cccccc', linestyle='-')
    ax2.grid(True, which='minor', color='#eeeeee', linestyle=':')
    leg2 = ax2.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)
    leg2.get_frame().set_alpha(1)

    plt.tight_layout()
    plt.show()
gerar_e_plotar_momentos()