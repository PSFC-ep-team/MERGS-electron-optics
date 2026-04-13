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
from numpy import geomspace, stack, concatenate, array
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
			front = np.loadtxt(f"generated/{name}_pareto_front.txt")
			resolutions = front[:, 0]
			efficiencies = front[:, 1]
			hyperparameters = front[:, 2:]
			print(f"re-loaded a previously calculated pareto front for {name}")

		else:
			if type(design) is str:
				resolutions, efficiencies, hyperparameters = find_pareto_front_of_magnet_design(name)
			elif len(design) == 3:
				foil_diameter, aperture_distance, aperture_diameter = design
				resolutions, efficiencies, hyperparameters = find_pareto_front_of_aperture_design(foil_diameter, aperture_distance, aperture_diameter)
			elif len(design) == 1:
				foil_diameter, = design
				resolutions, efficiencies, hyperparameters = find_pareto_front_of_collimator(foil_diameter)
			else:
				raise ValueError(f"wth does {design} mean?")

			np.savetxt(
				f"generated/{name}_pareto_front.txt",
				concatenate([
					stack([resolutions, efficiencies], axis=1),
					hyperparameters,
				], axis=1))

		fronts.append((resolutions, efficiencies, hyperparameters, label))

	performance_fig = plt.figure(figsize=(4.5, 4.0))
	performance_ax = performance_fig.add_subplot()
	parameter_fig = plt.figure(figsize=(6.0, 3.0))
	parameter_ax = parameter_fig.add_subplot()

	performance_ax.grid()
	parameter_ax.grid()

	for resolutions, efficiencies, hyperparameters, label in fronts:
		linestyle = "dashed" if "Ideal" in label else "solid"
		performance_ax.plot(resolutions, efficiencies, label=label, linestyle=linestyle)
		parameter_ax.plot(array(hyperparameters)[:, 2]*100, array(hyperparameters)[:, 3]*50, label=label)

	if len(fronts) > 1:
		performance_ax.legend()
		parameter_ax.legend()

	performance_ax.set_xlim(0, 800)
	performance_ax.xaxis.set_major_locator(MultipleLocator(200))
	performance_ax.set_yscale("log")
	performance_ax.set_ylim(0.1, 10)
	performance_ax.set_xlabel("Resolution (keV)")
	performance_ax.set_ylabel("Efficiency (counts/MJ)")
	performance_ax.set_title("Performance for 16.7 MeV photons")
	performance_fig.tight_layout()
	performance_fig.savefig("pareto.pdf")

	parameter_ax.set_xlim(0, 150)
	parameter_ax.set_ylim(0, 10)
	parameter_ax.set_xlabel("Aperture distance (cm)")
	parameter_ax.set_ylabel("Aperture radius (cm)")
	parameter_ax.set_title("Optimal aperture locations")
	parameter_fig.tight_layout()

	plt.show()


def find_pareto_front_of_aperture_design(foil_diameter: float, aperture_distance: float, aperture_diameter: float) -> tuple[Sequence[float], Sequence[float], Sequence[tuple[float, float, float, float]]]:
	"""
	find the range of achievable performances for a given set of aperture parameters,
	ignoring ion-optics and only varying foil thickness
	"""
	efficiencies = geomspace(0.1, 10, 9)  # counts/MJ
	resolutions = empty_like(efficiencies)
	hyperparameters = []
	with ProcessPoolExecutor(max_workers=9) as executor:
		for i, efficiency in enumerate(efficiencies):
			foil_thickness = optimize_foil_thickness(foil_diameter, aperture_distance, aperture_diameter, efficiency, executor)
			resolutions[i] = calculate_resolution(
				foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
				magnet_system_filename=None, parameters=None, executor=executor)
			hyperparameters.append((foil_diameter, foil_thickness, aperture_distance, aperture_diameter))
	return resolutions, efficiencies, hyperparameters


def find_pareto_front_of_collimator(foil_diameter: float) -> tuple[Sequence[float], Sequence[float], Sequence[tuple[float, float, float, float]]]:
	"""
	find the range of achievable performances for a given foil diameter,
	ignoring ion-optics and varying foil thickness, aperture distance, and aperture diameter
	"""
	# n.b. 0.1 counts/MJ means that we can make a ±10% measurement every 10 seconds at 100 MW operation,
	# and 10 counts/MJ means that we can make a ±10% measurement every second at 10 MW operation
	efficiencies = geomspace(0.1, 10, 9)  # counts/MJ

	resolutions, hyperparameters = zip(*run_concurrently(
		find_suitable_hyperparameters,
		efficiencies, foil_diameter,
	))

	return resolutions, efficiencies, hyperparameters


def find_suitable_hyperparameters(
		efficiency: float, foil_diameter: float) -> tuple[float, tuple[float, float, float, float]]:

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

	aperture_distance, aperture_diameter = solution.x
	foil_thickness = optimize_foil_thickness(
		foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
	return solution.fun, (foil_diameter, foil_thickness, aperture_distance, aperture_diameter)


def find_pareto_front_of_magnet_design(filename: str) -> tuple[Sequence[float], Sequence[float], Sequence[tuple[float, float, float, float]]]:
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
	efficiencies = geomspace(0.1, 10, 9)  # counts/MJ

	resolutions, hyperparameters = zip(*run_concurrently(
		find_suitable_configuration,
		efficiencies, max_foil_diameter, aperture_distance, max_aperture_diameter, filename,
	))

	return resolutions, efficiencies, hyperparameters


def find_suitable_configuration(
		efficiency: float, max_foil_diameter: float,
		aperture_distance: float, max_aperture_diameter: float,
		magnet_system_filename: str) -> tuple[float, tuple[float, float, float, float]]:

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

	foil_diameter, aperture_diameter = solution.x
	foil_thickness = optimize_foil_thickness(
		foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
	return solution.fun, (foil_diameter, foil_thickness, aperture_distance, aperture_diameter)


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
