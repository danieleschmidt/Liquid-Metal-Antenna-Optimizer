"""Demo: optimise a liquid metal patch antenna for 2.4 GHz WiFi."""

from liquid_metal_antenna import AntennaGeometry, EMSolver, DifferentiableOptimizer

TARGET_FREQ = 2.4e9  # Hz — 2.4 GHz WiFi


def main():
    print("=" * 60)
    print("Liquid Metal Antenna Optimizer — 2.4 GHz WiFi Demo")
    print("=" * 60)

    solver = EMSolver(substrate_height=1.6e-3)

    # Theoretical starting point: λ/2 ≈ c/(2f) ≈ 62.5 mm
    start = AntennaGeometry(length=0.0312, width=0.040)
    f_start = solver.resonant_frequency(start)
    print(f"\nInitial geometry : {start}")
    print(f"Initial resonant frequency : {f_start / 1e9:.4f} GHz")

    print("\nRunning CMA-ES optimisation (50 iterations)…")
    optimizer = DifferentiableOptimizer(em_solver=solver, pop_size=20, random_state=7)
    best = optimizer.optimize(
        target_freq=TARGET_FREQ,
        n_iterations=50,
        initial_length=start.length,
        initial_width=start.width,
    )

    f_best = solver.resonant_frequency(best)
    error_mhz = abs(f_best - TARGET_FREQ) / 1e6

    print(f"\nBest geometry found : {best}")
    print(f"Resonant frequency  : {f_best / 1e9:.4f} GHz")
    print(f"Error from target   : {error_mhz:.2f} MHz")
    print(f"Patch area          : {best.area() * 1e4:.2f} cm²")
    print("\nDone.")


if __name__ == "__main__":
    main()
