"""
code for finding the full Pareto curve of a design with fixed magnet geometry
"""
import os.path
import re
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy import geomspace, stack, zeros_like
from numpy.ma.core import empty_like
from scipy import optimize

from hyperparameters import calculate_resolution, optimize_foil_thickness, calculate_foil_resolution


def plot_pareto_fronts(*designs: str | tuple[float, float, float]):
	fronts = []

	for design in designs:
		if type(design) is str:
			name = str(design)
			if os.path.isfile(f"generated/{name}_pareto_front.txt"):
				resolutions, efficiencies = np.loadtxt(f"generated/{name}_pareto_front.txt", unpack=True)
			else:
				resolutions, efficiencies = find_pareto_front_of_magnet_design(name)
				np.savetxt(
					f"generated/{name}_pareto_front.txt",
					stack([resolutions, efficiencies], axis=1))
		else:
			foil_diameter, aperture_distance, aperture_diameter = design
			resolutions, efficiencies = find_pareto_front_of_aperture_design(foil_diameter, aperture_distance, aperture_diameter)
			name = f"{foil_diameter*100}-{aperture_distance*100}-{aperture_diameter*100}"
		fronts.append((resolutions, efficiencies, name))

	plt.rcParams["font.size"] = 12
	plt.rcParams['xtick.labelsize'] = 12
	plt.rcParams['ytick.labelsize'] = 12

	plt.figure(figsize=(4.5, 4.0))
	plt.grid()
	for resolutions, efficiencies, name in fronts:
		plt.plot(resolutions, efficiencies, label=name)
	if len(fronts) > 1:
		plt.legend()
	plt.xlim(0, 500)
	plt.yscale("log")
	plt.ylim(0.1, 10)
	plt.xlabel("Resolution")
	plt.ylabel("Efficiency (counts/MJ)")
	plt.tight_layout()
	plt.savefig("pareto.pdf")
	plt.show()


def find_pareto_front_of_aperture_design(foil_diameter: float, aperture_distance: float, aperture_diameter: float) -> tuple[Sequence[float], Sequence[float]]:
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ
	resolutions = empty_like(efficiencies)
	with ProcessPoolExecutor(max_workers=8) as executor:
		for i, efficiency in enumerate(efficiencies):
			resolutions[i] = calculate_foil_resolution(
				optimize_foil_thickness(foil_diameter, aperture_distance, aperture_diameter, efficiency, executor))
	return resolutions, efficiencies


def find_pareto_front_of_magnet_design(filename: str) -> tuple[Sequence[float], Sequence[float]]:
	with open(f"{filename}.fox") as file:
		script_content = file.read()
	max_foil_diameter = float(re.search(r"foil_width := ([0-9.]+)", script_content).group(1))
	max_aperture_diameter = float(re.search(r"aperture_width := ([0-9.]+)", script_content).group(1))
	aperture_distance = float(re.search(r"drift_pre_aperture := ([0-9.]+)", script_content).group(1))

	# n.b. 0.1 counts/MJ means that we can make a ±10% measurement every 10 seconds at 100 MW operation,
	# and 10 counts/MJ means that we can make a ±10% measurement every second at 10 MW operation
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ
	results: list[Future] = []

	with ProcessPoolExecutor(max_workers=8) as executor:
		for i, efficiency in enumerate(efficiencies):
			results.append(executor.submit(
				find_suitable_hyperparameters,
				efficiency, max_foil_diameter, aperture_distance, max_aperture_diameter))

	resolutions = zeros_like(efficiencies)
	for i, result in enumerate(results):
		resolutions[i] = result.result()

	return resolutions, efficiencies


def find_suitable_hyperparameters(
		efficiency: float, max_foil_diameter: float,
		aperture_distance: float, max_aperture_diameter: float):

	def objective(hyperparameters: float) -> float:
		foil_diameter, aperture_diameter = hyperparameters
		foil_thickness = optimize_foil_thickness(
			foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
		resolution = calculate_resolution(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			parameters=None, executor=None)
		print(f"{efficiency:.3g}: {hyperparameters} -> {resolution:.2f}")
		return resolution

	solution = optimize.minimize(
		objective,
		[max_foil_diameter, max_aperture_diameter],
		method="Nelder-Mead",
		bounds=[
			(.001, max_foil_diameter),
			(.001, max_aperture_diameter),
		],
		options=dict(
			initial_simplex=[
				[max_foil_diameter, max_aperture_diameter],
				[max_foil_diameter*0.6, max_aperture_diameter*0.8],
				[max_foil_diameter*0.8, max_aperture_diameter*0.6],
			],
			xatol=0.001,  # it doesn't need to be more precise than the nearest millimeter
			disp=True,
		)
	)
	print(solution)
	return solution.fun


if __name__ == "__main__":
	plot_pareto_fronts((.03, .50, .03), (.03, .25, .04))
