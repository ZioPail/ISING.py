import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from numba import njit, prange
from dask import delayed as dask_delayed, compute
from dask.distributed import Client, LocalCluster
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import logging
import psutil
#import matplotlib

# Usa il backend non interattivo per via di problemi della GUI con calcolo su più thread
#matplotlib.use('Agg')  

# Configurazione del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Funzione per calcolare il limite di memoria per ciascun worker
def calculate_memory_limit(total_ram, n_workers, usage_percentage=1):
    total_memory_to_use = total_ram * usage_percentage
    memory_per_worker = total_memory_to_use / n_workers
    memory_per_worker_mb = memory_per_worker / (1024**2)
    return f"{int(memory_per_worker_mb)}MB"

# Ottimizzazione della gestione dei worker Dask
#def configure_dask_cluster(memory_limit, n_workers, threads_per_worker):
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        scheduler_port=0,
        dashboard_address=':0'
    )
    client = Client(cluster)
    return client, cluster

# Funzione per configurare i cluster in modo automatico
def configure_dask_cluster():
    """
    Configura il cluster Dask in modo ottimale in base alle risorse della macchina.
    
    Returns:
        client (Client): Client Dask configurato.
        cluster (LocalCluster): Cluster Dask configurato.
    """
    # Ottieni le informazioni sulle risorse di sistema
    total_ram = psutil.virtual_memory().total
    total_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    # Configura il numero di worker e thread per worker
    n_workers = total_cores  # Un worker per ogni core fisico
    threads_per_worker = logical_cores // total_cores  # Distribuisci i thread sui core logici

    # Calcola il limite di memoria per ogni worker
    memory_limit = calculate_memory_limit(total_ram, n_workers)
    
    # Configura il cluster con i parametri ottimizzati
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,  # Usa processi separati per ogni worker
        scheduler_port=0,  # Porta casuale per evitare conflitti
        dashboard_address=None,  # Disabilita la dashboard
    )
    
    client = Client(cluster)

    # Log delle informazioni del cluster
    logger.info(f"Dask cluster configurato: {n_workers} worker, {threads_per_worker} thread per worker, memoria per worker: {memory_limit}")

    return client, cluster

# Funzione per inizializzare il reticolo con Numba e parallelizzazione
@njit
def initialize_lattice(N, T, Tc):
    lattice = np.empty((N, N), dtype=np.int8)
    if T < Tc - 1.0:
        lattice.fill(1)
    else:
        for i in range(N):
            for j in prange(N):
                lattice[i, j] = 1 if np.random.random() < 0.5 else -1
    return lattice

# Funzione dell'algoritmo a cluster per il MonteCarlo
@njit
def swendsen_wang_initialize(lattice, J, T, N):
    P_add = 1 - np.exp(-2 * J / T)
    labels = np.zeros_like(lattice, dtype=np.int32)
    current_label = 1

    for i in range(N):
        for j in prange(N):
            if labels[i, j] == 0:
                spin_initial = lattice[i, j]
                stack = [(i, j)]
                labels[i, j] = current_label

                while stack:
                    x, y = stack.pop()

                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        x_new = int((x + dx) % N)
                        y_new = int((y + dy) % N)

                        if lattice[x_new, y_new] == spin_initial and labels[x_new, y_new] == 0:
                            if np.random.random() < P_add:
                                stack.append((x_new, y_new))
                                labels[x_new, y_new] = current_label

                current_label += 1

    clusters = []
    for label in range(1, current_label):
        cluster = [(i, j) for i in range(N) for j in prange(N) if labels[i, j] == label]
        clusters.append(cluster)

    return clusters

# Funzione per aggiornare con cluster update il reticolo con l'algoritmo Swendsen-Wang
def swendsen_wang_update(lattice, clusters):
    for cluster in clusters:
        spin = np.random.randint(0, 2) * 2 - 1
        for x, y in cluster:
            lattice[x, y] = spin
    return lattice

# Funzione per calcolare l'energia del reticolo
@njit
def calculate_energy(lattice, J):
    N = lattice.shape[0]
    energy = 0.0
    for i in prange(N):
        for j in range(N):
            S = lattice[i, j]
            neighbors_sum = (
                lattice[(i + 1) % N, j] +
                lattice[(i - 1) % N, j] +
                lattice[i, (j + 1) % N] +
                lattice[i, (j - 1) % N]
            )
            energy += -J * S * neighbors_sum
    return energy / 2

# Funzione per una singola evoluzione dell'Ising per il calcolo distribuito
@dask_delayed
def single_simulation(t_idx, N, J, temperatures, Tc, equilibration_steps, sampling_steps, num_resamples):
    T = temperatures[t_idx]
    logger.info(f"Inizio simulazione per T={T:.4f}")

    lattice = initialize_lattice(N, T, Tc)

    # Equilibrazione
    for _ in range(equilibration_steps):
        clusters = swendsen_wang_initialize(lattice, J, T, N)
        lattice = swendsen_wang_update(lattice, clusters)

    magnetizations = np.zeros(sampling_steps)
    energies = np.zeros(sampling_steps)

    # Campionamento
    for s in range(sampling_steps):
        clusters = swendsen_wang_initialize(lattice, J, T, N)
        lattice = swendsen_wang_update(lattice, clusters)

        energy = calculate_energy(lattice, J)
        energies[s] = energy
        magnetizations[s] = np.abs(np.sum(lattice)) / (N * N)

    # Bootstrap
    resampled_magnetizations = np.zeros((num_resamples, sampling_steps))
    resampled_energies = np.zeros((num_resamples, sampling_steps))

    for i in range(num_resamples):
        resampled_indices = np.random.randint(0, sampling_steps, size=sampling_steps)
        resampled_magnetizations[i] = magnetizations[resampled_indices]
        resampled_energies[i] = energies[resampled_indices]

    mean_magnetizations = np.mean(resampled_magnetizations, axis=1)
    mean_energies = np.mean(resampled_energies, axis=1)

    # Calcolo della suscettività usando il bootstrap
    susceptibility = (np.mean(resampled_magnetizations ** 2, axis=1) - np.mean(mean_magnetizations) ** 2) / T

    # Calcolo del calore specifico usando il bootstrap
    specific_heat = (np.mean(resampled_energies ** 2, axis=1) - np.mean(mean_energies) ** 2) / (T ** 2)

    logger.info(f"Simulazione completata per T={T:.4f}")

    return mean_magnetizations, mean_energies, susceptibility, specific_heat

# Funzione per simulare il modello di Ising usando l'algoritmo a Cluster Swendsen-Wang
def simulate_ising_swendsen_wang(N, J, Tc, temperatures, equilibration_steps, sampling_steps, num_resamples):
    tasks = [
        dask_delayed(single_simulation)(t_idx, N, J, temperatures, Tc, equilibration_steps, sampling_steps, num_resamples)
        for t_idx in range(len(temperatures))
    ]

    results = compute(*tasks)

    magnetizations_list = []
    energies_list = []
    susceptibilities_list = []
    specific_heats_list = []

    for result in results:
        mean_magnetizations, mean_energies, susceptibility, specific_heat = result
        magnetizations_list.append(mean_magnetizations)
        energies_list.append(mean_energies)
        susceptibilities_list.append(susceptibility)
        specific_heats_list.append(specific_heat)

    magnetizations_array = np.array(magnetizations_list)
    energies_array = np.array(energies_list)
    susceptibilities_array = np.array(susceptibilities_list)
    specific_heats_array = np.array(specific_heats_list)

    return magnetizations_array, energies_array, susceptibilities_array, specific_heats_array

# Funzione per calcolare i cumulanti di Binder della magnetizzazione 
@njit
def calculate_binder_cumulant(magnetizations):
    m2 = np.mean(magnetizations ** 2)
    m4 = np.mean(magnetizations ** 4)
    return 1 - (m4 / (3 * m2 ** 2))

# Funzione per stimare la temperatura critica e l'incertezza statistica
def bootstrap_critical_temperature(temperatures, observables, num_resamples):
    Tc_estimates = []

    num_temperatures = len(temperatures)
    num_observables = len(observables)  

    # Itera attraverso i campioni bootstrap
    for _ in range(num_resamples):
        # Creiamo un array per memorizzare la media bootstrap per ciascuna osservabile a ciascuna temperatura
        resampled_means = np.zeros((num_temperatures, num_observables))

        # Itera su ciascuna osservabile
        for i, observable in enumerate(observables):
            # Resampling con replacement delle osservabili
            resampled_indices = np.random.choice(num_temperatures, size=num_temperatures, replace=True)
            resampled_values = observable[resampled_indices]

            # Calcola la media dei valori resampled per ciascuna temperatura
            resampled_means[:, i] = resampled_values

        # Selezioniamo l'osservabile da usare per stimare Tc
        chosen_observable = resampled_means[:, 0]  # Ad esempio, l'osservabile 0 è il cumulante di Binder

        # Interpoliamo la funzione
        spline = UnivariateSpline(temperatures, chosen_observable, s=0)

        # Troviamo la temperatura associata al massimo della spline
        Tc_estimate = temperatures[np.argmax(spline(temperatures))]
        Tc_estimates.append(Tc_estimate)

    # Calcoliamo media e deviazione standard delle stime di Tc
    Tc_mean = np.mean(Tc_estimates)
    Tc_std = np.std(Tc_estimates)

    return Tc_mean, Tc_std

# Funzione per stimare i parametri critici alpha e beta e le rispettive incertezze statistiche
def estimate_critical_exponents(temperatures, magnetizations, energies, Tc, num_resamples):
    def magnetization_func(T, beta, Tc):
        C = 1e-5
        M0 = 1
        return C + M0 * (np.abs(Tc - T)) ** beta

    def energy_func(T, alpha, Tc):
        C = 1e-5
        E0 = 1
        return C + E0 * (np.abs(Tc - T)) ** alpha

    beta_estimates = []
    alpha_estimates = []

    for _ in range(num_resamples):
        # Resampling dei dati (bootstrap)
        sample_indices = np.random.choice(len(temperatures), size=len(temperatures), replace=True)
        T_sampled = temperatures[sample_indices]
        M_sampled = np.mean(np.abs(magnetizations[:, sample_indices]), axis=0)
        E_sampled = np.mean(energies[:, sample_indices], axis=0)

        # Fitting di beta usando curve_fit
        p0_beta = [0.125, Tc]  # Valori iniziali per beta e Tc
        bounds_beta = ([0.05, 1.5], [0.3, 2.5])  # Ampliamento dei bounds per beta e Tc

        popt_beta, _ = curve_fit(magnetization_func, T_sampled, M_sampled, p0=p0_beta, bounds=bounds_beta)
        beta_estimates.append(popt_beta[0])

        # Fitting di alpha usando curve_fit
        p0_alpha = [0, Tc]  # Valori iniziali per alpha e Tc
        bounds_alpha = ([-0.2, 1.5], [0.2, 2.5])  # Ampliamento dei bounds per alpha e Tc

        popt_alpha, _ = curve_fit(energy_func, T_sampled, E_sampled, p0=p0_alpha, bounds=bounds_alpha)
        alpha_estimates.append(popt_alpha[0])

    # Calcola la media e la deviazione standard dalle distribuzioni bootstrap
    beta_mean = np.mean(beta_estimates)
    beta_std = np.std(beta_estimates)
    alpha_mean = np.mean(alpha_estimates)
    alpha_std = np.std(alpha_estimates)

    return beta_mean, beta_std, alpha_mean, alpha_std

# Funzione per calcolare l'autocorrelazione delle osservabili termodinamiche
@njit
def calculate_autocorrelation(data):
    n = len(data)
    data_mean = np.mean(data)
    data = data - data_mean
    autocorr = np.correlate(data, data, mode='full')[-n:]
    autocorr /= autocorr[0]
    return autocorr

# Funzione per generare i grafici dell'autocorrelazioni 
def plot_autocorrelation(temperatures, autocorr_data, label):
    plt.figure(figsize=(10, 6))
    for idx, autocorr in enumerate(autocorr_data):
        plt.plot(autocorr, label=f'T = {temperatures[idx]:.2f}')
    
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelazione')
    plt.title(f'Autocorrelazione {label}')
    plt.grid(True)
    plt.show()

# Funzione per generare i grafici delle quattro osservabili termodinamiche
def plot_ising_results(temperatures, magnetizations, energies, susceptibilities, specific_heats, Tc):
    sns.set_style('whitegrid')

    mean_magnetizations = np.mean(np.abs(magnetizations), axis=1)
    mean_energies = np.mean(energies, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, mean_magnetizations, marker='o')
    plt.title('Andamento della Magnetizzazione Media')
    plt.xlabel('Temperatura (T)')
    plt.ylabel('Magnetizzazione Media |M|')
    plt.axvline(x=Tc, color='blue', linestyle='--', label=f'Tc teorica: {Tc:.3f}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, mean_energies, marker='o')
    plt.title('Andamento dell\'Energia Media')
    plt.xlabel('Temperatura (T)')
    plt.ylabel('Energia Media')
    plt.axvline(x=Tc, color='blue', linestyle='--', label=f'Tc teorica: {Tc:.3f}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, np.mean(susceptibilities, axis=1), marker='o')
    plt.title('Suscettibilità Magnetica')
    plt.xlabel('Temperatura (T)')
    plt.ylabel('Suscettibilità')
    plt.axvline(x=Tc, color='blue', linestyle='--', label=f'Tc teorica: {Tc:.3f}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, np.mean(specific_heats, axis=1), marker='o')
    plt.title('Calore Specifico')
    plt.xlabel('Temperatura (T)')
    plt.ylabel('Calore Specifico')
    plt.axvline(x=Tc, color='blue', linestyle='--', label=f'Tc teorica: {Tc:.3f}')
    plt.legend()
    plt.show()

# Funzione per generare il grafico del cumulante di Binder per la magnetizzazione
def plot_binder_cumulant(temperatures, binder_cumulants, Tc):
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, binder_cumulants, marker='o', linestyle='-', color='blue', label='Cumulante di Binder')
    plt.xlabel('Temperatura (T)')
    plt.ylabel('Cumulante di Binder')
    plt.title('Analisi del Cumulante di Binder')
    plt.axvline(x=Tc, color='green', linestyle='--', label=f'Tc teorica: {Tc:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Funzione principale 
def main():
   
    client, cluster = configure_dask_cluster()
    
    try:
        
        # Definisce i parametri della simulazione
        N = 32
        J = 1.0
        T_min, T_max, T_steps = 1.5, 2.5, 10
        equilibration_steps = 4000
        sampling_steps = 2000
        num_resamples = 4000
        temperatures = np.linspace(T_min, T_max, T_steps)

        # Parte il cronometro per valutare i tempi di esecuzione del codice
        logger.info("Inizio simulazione")
        start_time = time.time()
        
        # Valori teorici dei parametri critici
        alpha_theoretical = 0.0
        beta_theoretical = 0.125
        Tc = 2 / np.log(1 + np.sqrt(2))
        
        # Esegue la simulazione e calcola le osservabili termodinamiche
        magnetizations, energies, susceptibilities, specific_heats = simulate_ising_swendsen_wang(
            N, J, Tc, temperatures, equilibration_steps, sampling_steps, num_resamples)  
        
        # Calcolo cumulante di Binder
        binder_cumulants = np.zeros_like(temperatures)
        for t_idx in range(len(temperatures)):
            binder_cumulants[t_idx] = calculate_binder_cumulant(np.abs(magnetizations[t_idx]))

        plot_binder_cumulant(temperatures, binder_cumulants, Tc)
        
        plot_ising_results(temperatures, magnetizations, energies, susceptibilities, specific_heats, Tc)
        
        # Autocorrelazioni
        autocorr_magnetizations = [calculate_autocorrelation(np.abs(magnetizations[i])) for i in range(len(temperatures))]
        autocorr_energies = [calculate_autocorrelation(energies[i]) for i in range(len(temperatures))]
        autocorr_susceptibilities = [calculate_autocorrelation(susceptibilities[i]) for i in range(len(temperatures))]
        autocorr_specific_heats = [calculate_autocorrelation(specific_heats[i]) for i in range(len(temperatures))]

        plot_autocorrelation(temperatures, autocorr_magnetizations, 'Magnetizzazione Media')
        plot_autocorrelation(temperatures, autocorr_energies, 'Energia Media')
        plot_autocorrelation(temperatures, autocorr_susceptibilities, 'Suscettibilità')
        plot_autocorrelation(temperatures, autocorr_specific_heats, 'Calore Specifico')

        # Stima delle temperature critiche con bootstrap
        Tc_magnetization, Tc_magnetization_err = bootstrap_critical_temperature(temperatures, magnetizations, num_resamples)
        Tc_energy, Tc_energy_err = bootstrap_critical_temperature(temperatures, energies, num_resamples)
        Tc_susceptibility, Tc_susceptibility_err = bootstrap_critical_temperature(temperatures, susceptibilities, num_resamples)
        Tc_specific_heat, Tc_specific_heat_err = bootstrap_critical_temperature(temperatures, specific_heats, num_resamples)
        Tc_binder, Tc_binder_err = bootstrap_critical_temperature(temperatures, binder_cumulants.reshape(1, -1), num_resamples)

        print(f"Stima della temperatura critica dalla magnetizzazione: {Tc_magnetization:.4f} ± {Tc_magnetization_err:.4f}")
        print(f"Stima della temperatura critica dall'energia: {Tc_energy:.4f} ± {Tc_energy_err:.4f}")
        print(f"Stima della temperatura critica dalla suscettibilità: {Tc_susceptibility:.4f} ± {Tc_susceptibility_err:.4f}")
        print(f"Stima della temperatura critica dal calore specifico: {Tc_specific_heat:.4f} ± {Tc_specific_heat_err:.4f}")
        print(f"Stima della temperatura critica dal cumulante di Binder: {Tc_binder:.4f} ± {Tc_binder_err:.4f}")

        # Stima degli esponenti critici
        beta_mean, beta_std, alpha_mean, alpha_std = estimate_critical_exponents(
            temperatures, magnetizations, energies, Tc_magnetization, num_resamples
        )

        print(f"Esponente critico beta: {beta_mean:.4f} ± {beta_std:.4f} (Teorico: {beta_theoretical:.4f})")
        print(f"Esponente critico alpha: {alpha_mean:.4f} ± {alpha_std:.4f} (Teorico: {alpha_theoretical:.4f})")
        
        end_time = time.time()
        print(f"Tempo impiegato per la simulazione: {end_time - start_time:.2f} secondi")
    
    finally:
        client.close()
        cluster.close()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
