import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. Model Parameters
T = 20000          # Simulation periods
gamma = 0.05       # Learning gain parameter
sigma_d = 0.021    # Dividend/Return volatility
sigma_x = 1.0      # Predictor variance
kappa = 0.65       # Feedback parameter
lmbda = 0.0025     # LASSO penalty parameter
w = 5.0            # Bounding parameter for tanh

# 2. Simulation Initialization
beta_ols = np.zeros(T)
M = sigma_x**2     # Initialize second moment

np.random.seed(42) # For reproducibility
x = np.random.normal(0, sigma_x, T)
eta = np.random.normal(0, sigma_d, T)

# 3. Simulation Loop
for t in range(2, T):
    # Actual Law of Motion (ALM)
    r_prev = (eta[t-1] + 
              np.log(1 - kappa * np.exp(w * np.tanh(x[t-2] * beta_ols[t-2] / w))) - 
              np.log(1 - kappa * np.exp(w * np.tanh(x[t-1] * beta_ols[t-1] / w))))
    
    # Belief Updating: Constant Gain Least Squares
    M = M + gamma * (x[t-1]**2 - M)
    beta_ols[t] = beta_ols[t-1] + gamma * (1/M) * x[t-2] * (r_prev - x[t-2] * beta_ols[t-1])

# 4. Visualization (Standardized Font & Style)
plt.rcParams.update({
    'font.family': 'serif',            # Use a serif font for everything
    'mathtext.fontset': 'dejavuserif', # Ensure math text matches the serif font
    'axes.titlesize': 14,              # Consistent title size
    'axes.labelsize': 12,              # Consistent axis label size
    'xtick.labelsize': 10,             # Consistent tick size
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

fig, ax = plt.subplots(figsize=(10, 6))

# Generate Histogram
n_bins = 100
n, bins, patches = ax.hist(beta_ols, bins=n_bins, density=True, edgecolor='none')

# Color the bars based on the lambda threshold
for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
    bin_center = (left_edge + right_edge) / 2
    if abs(bin_center) <= lmbda:
        patch.set_facecolor('#d3d3d3')  # Light gray (Sparsity Zone)
    else:
        patch.set_facecolor('#2f2f2f')  # Dark gray/Black (Active Beliefs)

# Create custom legend elements
legend_elements = [
    Patch(facecolor='#2f2f2f', label=r'Active Beliefs ($|\beta_{OLS}| > \lambda$)'),
    Patch(facecolor='#d3d3d3', label=r'Penalized Beliefs ($|\beta_{OLS}| \leq \lambda$)'),
    plt.Line2D([0], [0], color='black', lw=1.5, label=r'$\pm \lambda$ Threshold')
]

# Vertical lines for +/- lambda
ax.axvline(lmbda, color='black', linestyle='-', lw=1.5)
ax.axvline(-lmbda, color='black', linestyle='-', lw=1.5)

# Style formatting (JoF Style)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel(r'$\beta_{OLS}$')
ax.set_ylabel('Density')
ax.set_title('Distribution of OLS Estimates and the Soft-Thresholding Rule', pad=20)
ax.legend(handles=legend_elements, frameon=False)

plt.tight_layout()
plt.savefig(r'C:\Users\jonat\Lasso_paper\Results\Figures\lasso_threshold_jf_style.png', dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def run_lasso_cmce_simulation():
    np.random.seed(2)

    # ---------------------------------------------------------
    # 1. Model Parameters
    # ---------------------------------------------------------
    T = 1000       # Increased horizon to show the long-term flatline in full memory
    kappa = 0.92   # Slightly higher feedback to amplify early fluctuations
    epsilon = 0.05 # Bounding parameter
    sigma_x = 1.0  # Volatility of predictor
    sigma_d = 0.4  # Increased volatility of dividends to boost early forecast errors
    
    # Learning parameters
    gamma_cg = 0.01  # Slightly higher constant gain for more persistent pockets
    lam = 0.02    # LASSO penalty threshold (lambda)
    
    w = -np.log(kappa) - epsilon

    # ---------------------------------------------------------
    # 2. ALM Nonlinear Function
    # ---------------------------------------------------------
    def g(beta, x):
        return np.log(1.0 - kappa * np.exp(w * np.tanh(x * beta / w)))

    def soft_threshold(beta_ols, threshold):
        return np.sign(beta_ols) * np.maximum(0, np.abs(beta_ols) - threshold)

    # ---------------------------------------------------------
    # 3. Initialization
    # ---------------------------------------------------------
    x = np.random.normal(0, sigma_x, T)
    eta = np.random.normal(0, sigma_d, T)
    
    # Arrays for Constant Gain (CG)
    beta_ols_cg = np.zeros(T)
    beta_lasso_cg = np.zeros(T)
    M_cg = np.zeros(T)
    
    # Arrays for Full Memory (FM)
    beta_ols_fm = np.zeros(T)
    beta_lasso_fm = np.zeros(T)
    M_fm = np.zeros(T)
    
    # Start at the CMCE 
    beta_ols_cg[0:3] = 0.0 
    beta_lasso_cg[0:3] = 0.0
    M_cg[0:3] = sigma_x**2  

    beta_ols_fm[0:3] = 0.0 
    beta_lasso_fm[0:3] = 0.0
    M_fm[0:3] = sigma_x**2  

    # ---------------------------------------------------------
    # 4. Learning Loop (Both CG and FM)
    # ---------------------------------------------------------
    for t in range(3, T):
        # --- Constant Gain Update ---
        r_prev_cg = eta[t-1] + g(beta_lasso_cg[t-2], x[t-2]) - g(beta_lasso_cg[t-1], x[t-1])
        M_cg[t] = M_cg[t-1] + gamma_cg * (x[t-1]**2 - M_cg[t-1])
        forecast_error_cg = x[t-2] * r_prev_cg - (x[t-2]**2) * beta_ols_cg[t-1]
        beta_ols_cg[t] = beta_ols_cg[t-1] + (gamma_cg / M_cg[t-1]) * forecast_error_cg
        beta_lasso_cg[t] = soft_threshold(beta_ols_cg[t], lam)

        # --- Full Memory Update (Decreasing Gain) ---
        gamma_fm = 1.0 / t # Gain shrinks over time
        r_prev_fm = eta[t-1] + g(beta_lasso_fm[t-2], x[t-2]) - g(beta_lasso_fm[t-1], x[t-1])
        M_fm[t] = M_fm[t-1] + gamma_fm * (x[t-1]**2 - M_fm[t-1])
        forecast_error_fm = x[t-2] * r_prev_fm - (x[t-2]**2) * beta_ols_fm[t-1]
        beta_ols_fm[t] = beta_ols_fm[t-1] + (gamma_fm / M_fm[t-1]) * forecast_error_fm
        beta_lasso_fm[t] = soft_threshold(beta_ols_fm[t], lam)

    # ---------------------------------------------------------
    # 5. Compute Pocket Durations
    # ---------------------------------------------------------
    def calculate_pockets(beta_lasso_array):
        is_active = np.abs(beta_lasso_array) > 0
        pocket_lengths = []
        current_length = 0
        for active in is_active:
            if active:
                current_length += 1
            else:
                if current_length > 0:
                    pocket_lengths.append(current_length)
                    current_length = 0
        if current_length > 0:
            pocket_lengths.append(current_length)
        return is_active, pocket_lengths

    is_active_cg, pockets_cg = calculate_pockets(beta_lasso_cg)
    is_active_fm, pockets_fm = calculate_pockets(beta_lasso_fm)

    print("--- CONSTANT GAIN SIMULATION ---")
    if len(pockets_cg) > 0:
        print(f"Total pockets: {len(pockets_cg)}")
        print(f"Average duration: {np.mean(pockets_cg):.1f} periods")
    else:
        print("No pockets emerged.")

    print("\n--- FULL MEMORY SIMULATION ---")
    if len(pockets_fm) > 0:
        print(f"Total pockets: {len(pockets_fm)}")
        print(f"Average duration: {np.mean(pockets_fm):.1f} periods")
    else:
        print("No pockets emerged.")

    # ---------------------------------------------------------
    # 6. Plotting the Comparison
    # ---------------------------------------------------------
    
    # NEW: Global font settings to ensure standard text and math text match perfectly
    plt.rcParams.update({
        'font.family': 'serif',            # Use a serif font for everything
        'mathtext.fontset': 'dejavuserif', # Ensure math text matches the serif font
        'axes.titlesize': 14,              # Consistent title size
        'axes.labelsize': 12,              # Consistent axis label size
        'xtick.labelsize': 10,             # Consistent tick size
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    y_max = max(np.max(np.abs(beta_ols_cg)) * 1.1, lam * 2)

    # --- Plot Full Memory (Now on Top: ax1) ---
    ax1.plot(range(T), beta_ols_fm, label='$\\beta^{OLS}_t$', color='gray', linewidth=1.5, alpha=0.4)
    ax1.plot(range(T), beta_lasso_fm, label='$\\beta^{LASSO}_t$', color='#2ca02c', linewidth=2.5)
    ax1.fill_between(range(T), -y_max, y_max, where=is_active_fm, color='#ff7f0e', alpha=0.15, label='Pocket of Predictability')
    ax1.axhline(lam, color='#d62728', linestyle='--', alpha=0.8, label='LASSO Threshold ($\\pm \\lambda$)')
    ax1.axhline(-lam, color='#d62728', linestyle='--', alpha=0.8)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_title('Simulated Beliefs under Full Memory: Convergence to CMCE')
    ax1.set_ylabel('Belief Parameter')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_ylim(-y_max, y_max)

    # --- Plot Constant Gain (Now on Bottom: ax2) ---
    ax2.plot(range(T), beta_ols_cg, label='$\\beta^{OLS}_t$', color='gray', linewidth=1.5, alpha=0.4)
    ax2.plot(range(T), beta_lasso_cg, label='$\\beta^{LASSO}_t$', color='#1f77b4', linewidth=2.5)
    ax2.fill_between(range(T), -y_max, y_max, where=is_active_cg, color='#ff7f0e', alpha=0.15)
    ax2.axhline(lam, color='#d62728', linestyle='--', alpha=0.8)
    ax2.axhline(-lam, color='#d62728', linestyle='--', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax2.set_title('Simulated Beliefs under Finite Memory: Persistent Fluctuations')
    ax2.set_xlabel('Time ($t$)')
    ax2.set_ylabel('Belief Parameter')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(r'C:\Users\jonat\Lasso_paper\Results\Figures\lasso_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_lasso_cmce_simulation()