from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    structure_id = 1
    for cloud_number in range(1, 7):
        cloud_id = f"MC{cloud_number:02d}"
        count = 120
        radii = 10 ** rng.uniform(-1.0, 1.1, count)
        hierarchy = rng.integers(0, 5, count)
        log_column = rng.normal(21.5 + 0.08 * hierarchy, 0.35, count)
        slope = np.where(log_column >= 21.7, 0.62, 0.43)
        velocity = 0.75 * radii**slope * 10 ** rng.normal(0, 0.12, count)
        mass = 180 * radii**2.15 * 10 ** rng.normal(0, 0.25, count)
        virial = 5.0 * velocity**2 * radii / np.maximum(mass / 100, 1e-3)

        first_id = structure_id
        for i in range(count):
            level = int(hierarchy[i])
            if level == 0 or i < 5:
                parent_id = None
            else:
                candidates = [row["structure_id"] for row in rows[-i:] if row["hierarchy_level"] < level]
                parent_id = int(rng.choice(candidates)) if candidates else first_id
            rows.append(
                {
                    "structure_id": structure_id,
                    "parent_id": parent_id,
                    "cloud_id": cloud_id,
                    "radius_pc": float(radii[i]),
                    "velocity_dispersion_kms": float(velocity[i]),
                    "mass_msun": float(mass[i]),
                    "column_density_cm2": float(10 ** log_column[i]),
                    "virial_parameter": float(virial[i]),
                    "hierarchy_level": level,
                }
            )
            structure_id += 1

    output = Path("data/sample_catalog.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

