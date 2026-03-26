"""
code for finding the full Pareto curve of a design with fixed magnet geometry
"""
import os.path
import re
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from typing import Sequence, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy import geomspace, stack, array
from numpy.ma.core import empty_like
from scipy import optimize

from hyperparameters import calculate_resolution, optimize_foil_thickness


def plot_pareto_fronts(*designs: str | tuple[float] | tuple[float, float, float]):
	fronts = []

	for design in designs:
		if type(design) is str:
			name = str(design)
			label = name
		elif len(design) == 3:
			foil_diameter, aperture_distance, aperture_diameter = design
			name = f"{foil_diameter*100}-{aperture_distance*100}-{aperture_diameter*100}"
			label = name
		elif len(design) == 1:
			foil_diameter, = design
			name = str(foil_diameter*100)
			label = "Ideal"
		else:
			raise ValueError(f"wth does {design} mean?")

		if os.path.isfile(f"generated/{name}_pareto_front.txt"):
			resolutions, efficiencies = np.loadtxt(f"generated/{name}_pareto_front.txt", unpack=True)
			print(f"re-loaded a previously calculated pareto front for {name}")

		else:
			if type(design) is str:
				resolutions, efficiencies = find_pareto_front_of_magnet_design(name)
			elif len(design) == 3:
				foil_diameter, aperture_distance, aperture_diameter = design
				resolutions, efficiencies = find_pareto_front_of_aperture_design(foil_diameter, aperture_distance, aperture_diameter)
			elif len(design) == 1:
				foil_diameter, = design
				resolutions, efficiencies = find_pareto_front_of_collimator(foil_diameter)
			else:
				raise ValueError(f"wth does {design} mean?")

			np.savetxt(
				f"generated/{name}_pareto_front.txt",
				stack([resolutions, efficiencies], axis=1))

		fronts.append((resolutions, efficiencies, label))

	plt.rcParams["font.size"] = 12
	plt.rcParams['xtick.labelsize'] = 12
	plt.rcParams['ytick.labelsize'] = 12

	plt.figure(figsize=(4.5, 4.0))
	plt.grid()
	for resolutions, efficiencies, label in fronts:
		linestyle = "dashed" if "Ideal" in label else "solid"
		plt.plot(resolutions, efficiencies, label=label, linestyle=linestyle)
	if len(fronts) > 1:
		plt.legend()
	plt.xlim(0, 1000)
	plt.gca().xaxis.set_major_locator(MultipleLocator(250))
	plt.yscale("log")
	plt.ylim(0.1, 10)
	plt.xlabel("Resolution (keV)")
	plt.ylabel("Efficiency (counts/MJ)")
	plt.title("Performance for 16.7 MeV photons")
	plt.tight_layout()
	plt.savefig("pareto.pdf")
	plt.show()


def find_pareto_front_of_aperture_design(foil_diameter: float, aperture_distance: float, aperture_diameter: float) -> tuple[Sequence[float], Sequence[float]]:
	"""
	find the range of achievable performances for a given set of aperture parameters,
	ignoring ion-optics and only varying foil thickness
	"""
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ
	resolutions = empty_like(efficiencies)
	with ProcessPoolExecutor(max_workers=8) as executor:
		for i, efficiency in enumerate(efficiencies):
			foil_thickness = optimize_foil_thickness(foil_diameter, aperture_distance, aperture_diameter, efficiency, executor)
			resolutions[i] = calculate_resolution(
				foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
				magnet_system_filename=None, parameters=None, executor=executor)
	return resolutions, efficiencies


def find_pareto_front_of_collimator(foil_diameter: float) -> tuple[Sequence[float], Sequence[float]]:
	"""
	find the range of achievable performances for a given foil diameter,
	ignoring ion-optics and varying foil thickness, aperture distance, and aperture diameter
	"""
	# n.b. 0.1 counts/MJ means that we can make a ±10% measurement every 10 seconds at 100 MW operation,
	# and 10 counts/MJ means that we can make a ±10% measurement every second at 10 MW operation
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ

	resolutions = array(run_concurrently(
		find_suitable_hyperparameters,
		efficiencies, foil_diameter,
	))

	return resolutions, efficiencies


def find_suitable_hyperparameters(
		efficiency: float, foil_diameter: float):

	def objective(hyperparameters: Sequence[float]) -> float:
		aperture_distance, aperture_diameter = hyperparameters
		foil_thickness = optimize_foil_thickness(
			foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
		resolution = calculate_resolution(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			magnet_system_filename=None, parameters=None, executor=None)
		print(f"{efficiency:.3g}: {hyperparameters} -> {resolution:.2f}")
		return resolution

	solution = optimize.minimize(
		objective,
		[.40, .04],
		method="Nelder-Mead",
		bounds=[
			(.03, 10.00),
			(.01, .10),
		],
		options=dict(
			initial_simplex=[
				[.30, .04],
				[.40, .05],
				[.50, .03],
			],
			xatol=0.001,  # it doesn't need to be more precise than the nearest millimeter
			disp=True,
		)
	)
	print(solution)
	return solution.fun


def find_pareto_front_of_magnet_design(filename: str) -> tuple[Sequence[float], Sequence[float]]:
	"""
	find the range of achievable performances for a given magnet system,
	accounting for all sources of degradation and only varying foil thickness, foil diameter, and aperture diameter
	"""
	with open(f"{filename}.fox") as file:
		script_content = file.read()
	max_foil_diameter = float(re.search(r"foil_width := ([0-9.]+)", script_content).group(1))
	max_aperture_diameter = float(re.search(r"aperture_width := ([0-9.]+)", script_content).group(1))
	aperture_distance = float(re.search(r"drift_pre_aperture := ([0-9.]+)", script_content).group(1))

	# n.b. 0.1 counts/MJ means that we can make a ±10% measurement every 10 seconds at 100 MW operation,
	# and 10 counts/MJ means that we can make a ±10% measurement every second at 10 MW operation
	efficiencies = geomspace(0.1, 10, 11)  # counts/MJ

	resolutions = array(run_concurrently(
		find_suitable_configuration,
		efficiencies, max_foil_diameter, aperture_distance, max_aperture_diameter, filename,
	))

	return resolutions, efficiencies


def find_suitable_configuration(
		efficiency: float, max_foil_diameter: float,
		aperture_distance: float, max_aperture_diameter: float,
		magnet_system_filename: str):

	def objective(hyperparameters: Sequence[float]) -> float:
		foil_diameter, aperture_diameter = hyperparameters
		foil_thickness = optimize_foil_thickness(
			foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
		resolution = calculate_resolution(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			magnet_system_filename, parameters=None, executor=None)
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


def run_concurrently(function: Callable, parameter_sweep: Sequence, *args, **kwargs):
	results: list[Future] = []

	with ProcessPoolExecutor(max_workers=8) as executor:
		for i, parameter in enumerate(parameter_sweep):
			results.append(executor.submit(
				function, parameter, *args, **kwargs))

	resolutions = []
	for result in results:
		resolutions.append(result.result())
	return resolutions


if __name__ == "__main__":
	plot_pareto_fronts("mergs_electron_optics", (.03,))
