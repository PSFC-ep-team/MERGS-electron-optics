import re
from concurrent.futures import Executor
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, Optional

import matplotlib.pyplot as plt
from numpy import geomspace
from scipy import optimize

from hyperparameters import calculate_resolution, optimize_foil_thickness


def plot_pareto_front(*filenames: str):
	plt.figure(figsize=(4.5, 4.0))
	plt.grid()
	for filename in filenames:
		resolutions, efficiencies = find_pareto_front(filename, None)
		plt.plot(resolutions, efficiencies)
	plt.xlim(0, 500)
	plt.yscale("log")
	plt.ylim(0.1, 10)
	plt.xlabel("Resolution")
	plt.ylabel("Efficiency (detections/MJ)")
	plt.tight_layout()
	plt.savefig("pareto.pdf")
	plt.show()


def find_pareto_front(filename: str, executor: Optional[Executor]) -> tuple[Sequence[float], Sequence[float]]:
	with open(f"{filename}.fox") as file:
		script_content = file.read()
	max_foil_diameter = float(re.search(r"foil_width := ([0-9.]+)", script_content).group(1))
	max_aperture_diameter = float(re.search(r"aperture_width := ([0-9.]+)", script_content).group(1))
	min_aperture_distance = float(re.search(r"drift_pre_aperture := ([0-9.]+)", script_content).group(1))

	# n.b. 0.1 counts/MJ means that we can make a ±10% measurement every 10 seconds at 10 MW operation,
	# and 10 counts/MJ means that we can make ten ±10% measurements every second at 100 MW operation
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ
	resolutions = []

	for efficiency in efficiencies:
		print(efficiency)

		def objective(hyperparameters: float) -> float:
			foil_diameter, aperture_distance, aperture_diameter = hyperparameters
			foil_thickness = optimize_foil_thickness(
				foil_diameter, aperture_distance, aperture_diameter, efficiency, executor)
			resolution = calculate_resolution(
				foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
				parameters=None, executor=executor)
			print(f"{hyperparameters} -> {resolution:.2f}")
			return resolution

		solution = optimize.minimize(
			objective,
			[max_foil_diameter, min_aperture_distance, max_aperture_diameter],
			method="Nelder-Mead",
			bounds=[
				(.001, max_foil_diameter),
				(min_aperture_distance, 3.00),
				(.001, max_aperture_diameter),
			],
			options=dict(
				xatol=0.001,  # it doesn't need to be more precise than the nearest millimeter
				disp=True,
			)
		)
		print(solution)

		resolutions.append(objective(solution.x))

	return resolutions, efficiencies


if __name__ == "__main__":
	plot_pareto_front("mergs_electron_optics")
