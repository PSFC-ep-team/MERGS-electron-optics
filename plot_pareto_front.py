import os.path
import re
from concurrent.futures import Executor, Future
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import geomspace, stack, zeros_like
from scipy import optimize

from hyperparameters import calculate_resolution, optimize_foil_thickness


def plot_pareto_front(*filenames: str):
	fronts = []

	for filename in filenames:
		if os.path.isfile(f"generated/{filename}_pareto_front.txt"):
			resolutions, efficiencies = np.loadtxt(f"{filename}_pareto_front.txt", unpack=True)
		else:
			resolutions, efficiencies = find_pareto_front(filename)
			np.savetxt(
				f"generated/{filename}_pareto_front.txt",
				stack([resolutions, efficiencies], axis=1))
		fronts.append((resolutions, efficiencies))

	plt.rcParams["font.size"] = 12
	plt.rcParams['xtick.labelsize'] = 12
	plt.rcParams['ytick.labelsize'] = 12

	plt.figure(figsize=(4.5, 4.0))
	plt.grid()
	for resolutions, efficiencies in fronts:
		plt.plot(resolutions, efficiencies)
	plt.xlim(0, 500)
	plt.yscale("log")
	plt.ylim(0.1, 10)
	plt.xlabel("Resolution")
	plt.ylabel("Efficiency (detections/MJ)")
	plt.tight_layout()
	plt.savefig("pareto.pdf")
	plt.show()


def find_pareto_front(filename: str) -> tuple[Sequence[float], Sequence[float]]:
	with open(f"{filename}.fox") as file:
		script_content = file.read()
	max_foil_diameter = float(re.search(r"foil_width := ([0-9.]+)", script_content).group(1))
	max_aperture_diameter = float(re.search(r"aperture_width := ([0-9.]+)", script_content).group(1))
	min_aperture_distance = float(re.search(r"drift_pre_aperture := ([0-9.]+)", script_content).group(1))

	# n.b. 0.1 counts/MJ means that we can make a ±10% measurement every 10 seconds at 10 MW operation,
	# and 10 counts/MJ means that we can make ten ±10% measurements every second at 100 MW operation
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ
	results: list[Future] = []

	with ProcessPoolExecutor(max_workers=8) as executor:
		for i, efficiency in enumerate(efficiencies):
			results.append(executor.submit(
				find_suitable_hyperparameters,
				efficiency, max_foil_diameter, min_aperture_distance, max_aperture_diameter))

	resolutions = zeros_like(efficiencies)
	for i, result in enumerate(results):
		resolutions[i] = result.result()

	return resolutions, efficiencies


def find_suitable_hyperparameters(
		efficiency: float, max_foil_diameter: float,
		min_aperture_distance: float, max_aperture_diameter: float):

	def objective(hyperparameters: float) -> float:
		foil_diameter, aperture_distance, aperture_diameter = hyperparameters
		foil_thickness = optimize_foil_thickness(
			foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
		resolution = calculate_resolution(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			parameters=None, executor=None)
		print(f"{efficiency:.3g}: {hyperparameters} -> {resolution:.2f}")
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
			initial_simplex=[
				[max_foil_diameter, min_aperture_distance, max_aperture_diameter],
				[max_foil_diameter/2, min_aperture_distance, max_aperture_diameter],
				[max_foil_diameter/2, min_aperture_distance, max_aperture_diameter/2],
				[max_foil_diameter/2, min_aperture_distance*2, max_aperture_diameter/2],
			],
			xatol=0.001,  # it doesn't need to be more precise than the nearest millimeter
			disp=True,
		)
	)
	print(solution)
	return solution.fun


if __name__ == "__main__":
	plot_pareto_front("mergs_electron_optics")
