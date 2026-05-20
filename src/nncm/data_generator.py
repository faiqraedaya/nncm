import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------
# User settings
# -----------------------------------------

N_PRESSURE_VESSELS = 1000
N_LEAKS_PER_VESSEL = 10

# Bounds (Phase 1)
T_MIN, T_MAX = -40, 200          # degC
P_MIN, P_MAX = 1, 100            # barg
D_MIN, D_MAX = 1, 1000           # mm

# -----------------------------------------
# Latin Hypercube Sampling
# -----------------------------------------

def lhs_sample(n_samples, bounds):
    """
    Latin Hypercube sample generator.
    bounds: list of (min, max) pairs for each dimension.
    """
    dim = len(bounds)
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n_samples)
    scaled = qmc.scale(sample,
                       [b[0] for b in bounds],
                       [b[1] for b in bounds])
    return scaled


def plot_lhs_distributions(temperatures, pressures, orifice_diams,
                           t_bounds=(T_MIN, T_MAX),
                           p_bounds=(P_MIN, P_MAX),
                           d_bounds=(D_MIN, D_MAX)):
    """
    Generate plots showing the distribution of LHS sampled data points.

    Parameters:
    -----------
    temperatures : array-like
        Temperature values from LHS sampling
    pressures : array-like
        Pressure values from LHS sampling
    orifice_diams : array-like
        Orifice diameter values from LHS sampling
    t_bounds : tuple
        (min, max) bounds for temperature
    p_bounds : tuple
        (min, max) bounds for pressure
    d_bounds : tuple
        (min, max) bounds for orifice diameter
    """
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    ax1.scatter(temperatures, pressures, alpha=0.6, s=20, c='steelblue', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure (barg)', fontsize=12, fontweight='bold')
    ax1.set_title('LHS Distribution: Temperature vs Pressure\n(Pressure Vessels)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t_bounds[0] - 5, t_bounds[1] + 5)
    ax1.set_ylim(p_bounds[0] - 2, p_bounds[1] + 2)

    from matplotlib.patches import Rectangle
    rect = Rectangle((t_bounds[0], p_bounds[0]),
                     t_bounds[1] - t_bounds[0],
                     p_bounds[1] - p_bounds[0],
                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.7)
    ax1.add_patch(rect)
    ax1.text(0.02, 0.98, f'n = {len(temperatures)}',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]
    ax2.hist(temperatures, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(t_bounds[0], color='red', linestyle='--', linewidth=2, label='Bounds')
    ax2.axvline(t_bounds[1], color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('LHS Distribution: Temperature\n(Marginal Distribution)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    ax3 = axes[2]
    ax3.hist(orifice_diams, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(d_bounds[0], color='red', linestyle='--', linewidth=2, label='Bounds')
    ax3.axvline(d_bounds[1], color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Orifice Diameter (mm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('LHS Distribution: Orifice Diameter\n(Leaks)',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    ax3.text(0.02, 0.98, f'n = {len(orifice_diams)}',
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out1 = PROJECT_ROOT / "data" / "plots" / "lhs_distribution_plots.png"
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    print(f"\nLHS distribution plots saved as '{out1}'")
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax4 = axes2[0]
    ax4.hist(pressures, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(p_bounds[0], color='red', linestyle='--', linewidth=2, label='Bounds')
    ax4.axvline(p_bounds[1], color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Pressure (barg)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('LHS Distribution: Pressure\n(Marginal Distribution)',
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()

    ax5 = axes2[1]
    hb = ax5.hexbin(temperatures, pressures, gridsize=30, cmap='Blues', mincnt=1)
    ax5.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Pressure (barg)', fontsize=12, fontweight='bold')
    ax5.set_title('LHS Distribution: Temperature vs Pressure\n(Hexbin Density)',
                  fontsize=13, fontweight='bold')
    cb = plt.colorbar(hb, ax=ax5)
    cb.set_label('Count', fontsize=10)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = PROJECT_ROOT / "data" / "plots" / "lhs_distribution_detailed.png"
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    print(f"Detailed LHS distribution plots saved as '{out2}'")
    plt.show()


def main():
    # -----------------------------------------
    # Pressure Vessel generation
    # -----------------------------------------
    pv_samples = lhs_sample(
        N_PRESSURE_VESSELS,
        [(T_MIN, T_MAX), (P_MIN, P_MAX)]
    )

    temperatures = pv_samples[:, 0]
    pressures = pv_samples[:, 1]

    df_pressure_vessels = pd.DataFrame({
        "Use": ["Yes"] * N_PRESSURE_VESSELS,
        "Study": ["Study"] * N_PRESSURE_VESSELS,
        "Folder": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Name": [f"PV_{i+1}" for i in range(N_PRESSURE_VESSELS)],
        "Material": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Specify volume inventory?": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Mass inventory": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Volume inventory": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Material to track": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Specified condition": ["" for _ in range(N_PRESSURE_VESSELS)],
        "Temperature": temperatures,
        "Pressure (gauge)": pressures,
        "Liquid mole fraction": ["" for _ in range(N_PRESSURE_VESSELS)],
    })

    # -----------------------------------------
    # Leak generation (linked to Pressure Vessels)
    # -----------------------------------------
    total_leaks = N_PRESSURE_VESSELS * N_LEAKS_PER_VESSEL
    leak_samples = lhs_sample(total_leaks, [(D_MIN, D_MAX)])
    orifice_diams = leak_samples[:, 0]

    pv_index_for_leak = np.repeat(
        np.arange(N_PRESSURE_VESSELS),
        N_LEAKS_PER_VESSEL
    )

    df_leaks = pd.DataFrame({
        "Use": ["Yes"] * total_leaks,
        "Study": ["Study"] * total_leaks,
        "Folder": ["" for _ in range(total_leaks)],
        "Pressure vessel": [f"PV_{i+1}" for i in pv_index_for_leak],
        "Folder.1": ["" for _ in range(total_leaks)],
        "Atmospheric storage tank": ["" for _ in range(total_leaks)],
        "Folder.2": ["" for _ in range(total_leaks)],
        "Name": [f"LEAK_{i+1}" for i in range(total_leaks)],
        "Orifice diameter": orifice_diams,
        "Use specified discharge coefficient?": ["" for _ in range(total_leaks)],
        "Discharge coefficient": ["" for _ in range(total_leaks)],
        "Elevation": ["" for _ in range(total_leaks)],
        "Tank head": ["" for _ in range(total_leaks)],
        "Outdoor release direction": ["" for _ in range(total_leaks)],
        "Outdoor release angle": ["" for _ in range(total_leaks)],
    })

    print("Pressure Vessel Data (head):")
    print(df_pressure_vessels.head())
    print("\nLeak Data (head):")
    print(df_leaks.head())

    plot_lhs_distributions(temperatures, pressures, orifice_diams)

    out_pv = PROJECT_ROOT / "data" / "generated" / "pressure_vessel_input.xlsx"
    out_lk = PROJECT_ROOT / "data" / "generated" / "leak_input.xlsx"
    df_pressure_vessels.to_excel(out_pv, index=False)
    df_leaks.to_excel(out_lk, index=False)
    print(f"\nExported: {out_pv}")
    print(f"Exported: {out_lk}")


if __name__ == "__main__":
    main()
