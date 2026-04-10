import numpy as np
from scipy.interpolate import interp1d

def build_inverse_interpolator(
    energies: np.ndarray,
    csda_ranges: np.ndarray,
) -> interp1d:
    """
    Build interpolation E(R). Assumes CSDA range is monotonic in energy.
    """
    sort_idx = np.argsort(csda_ranges)
    r_sorted = csda_ranges[sort_idx]
    e_sorted = energies[sort_idx]

    return interp1d(
        r_sorted,
        e_sorted,
        kind="cubic",
        bounds_error=True,
    )


def mc_energy_from_range(
    r_mean: float,
    r_sigma: float,
    e_of_r: interp1d,
    n_samples: int = 20000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Monte Carlo propagation from remaining range -> remaining energy.

    Parameters
    ----------
    r_mean : float
        Mean remaining CSDA range in g/cm^2.
    r_sigma : float
        Standard uncertainty of remaining range in g/cm^2.
    e_of_r : interp1d
        Inverse interpolation E(R).
    n_samples : int
        Number of MC samples.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    dict with mean, std, median, q16, q84, samples
    """
    if rng is None:
        rng = np.random.default_rng()

    r_samples = rng.normal(loc=r_mean, scale=r_sigma, size=n_samples)

    # Keep only samples inside interpolation domain
    r_min = float(e_of_r.x.min())
    r_max = float(e_of_r.x.max())
    mask = (r_samples >= r_min) & (r_samples <= r_max)
    r_samples = r_samples[mask]

    if len(r_samples) == 0:
        raise ValueError(
            "All MC samples lie outside ASTAR interpolation range.")

    e_samples = e_of_r(r_samples)

    return {
        "mean": float(np.mean(e_samples)),
        "std": float(np.std(e_samples, ddof=1)),
        "median": float(np.median(e_samples)),
        "q16": float(np.quantile(e_samples, 0.16)),
        "q84": float(np.quantile(e_samples, 0.84)),
        "n_used": int(len(e_samples)),
        "samples": e_samples,
    }


def main() -> None:
    # -------------------------------------------------
    # PART A: run once to create ASTAR input energies
    # -------------------------------------------------
    # Uncomment once to generate the ASTAR input file:
    
    write_energy_grid_for_astar(
         filename="astar_energy_input_po214.txt",
         e_min=2,
         e_max=8.5,
         step=0.01,
     )
    
    # Then feed this file into ASTAR and save the output,
     #e.g. as "astar_air_output.txt".

    # -------------------------------------------------
    # PART B: after ASTAR output exists
    # -------------------------------------------------
    astar_output_file = "astar_air_output.txt"

    energies, csda_ranges = load_astar_output(astar_output_file)
    e_of_r = build_inverse_interpolator(energies, csda_ranges)

    # Example remaining ranges in g/cm^2
    # Replace with your actual values
    remaining_ranges = np.array([
        1.977e-3,
        3.826e-3,
        2.500e-3,
        4.100e-3,
    ])

    remaining_range_unc = np.array([
        0.040e-3,
        0.050e-3,
        0.040e-3,
        0.060e-3,
    ])

    rng = np.random.default_rng(12345)

    for i, (r_mean, r_sigma) in enumerate(zip(remaining_ranges, remaining_range_unc), start=1):
        result = mc_energy_from_range(
            r_mean=r_mean,
            r_sigma=r_sigma,
            e_of_r=e_of_r,
            n_samples=50000,
            rng=rng,
        )

        print(f"Point {i}")
        print(f"  R_rem = {r_mean:.6e} ± {r_sigma:.6e} g/cm^2")
        print(f"  E_rem = {result['mean']:.4f} ± {result['std']:.4f} MeV")
        print(
            f"  16-50-84 % = {result['q16']:.4f}, {result['median']:.4f}, {result['q84']:.4f} MeV")
        print(f"  used samples = {result['n_used']}")
        print()


if __name__ == "__main__":
    main()

