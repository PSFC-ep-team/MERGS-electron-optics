"""
code for finding the full Pareto curve of a design with fixed magnet geometry
"""
import logging
import os.path
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import current_process
from typing import Sequence, Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from numpy import inf, geomspace, stack, concatenate, array, loadtxt, savetxt, degrees
from numpy.ma.core import empty_like
from scipy import optimize

from electron_optics import run_cosy, load_script
from hyperparameters import optimize_foil_thickness, calculate_resolution_of_map

ORDER = 9

# generate and save the ideal map, in case we need it
IDEAL_MAP_FILENAME = f"generated/ideal_map.txt"
IDEAL_MAP = (
	"0.0  0.0  0.0  0.0  0.0  100000\n"
	"0.0  0.0  0.0  0.0  0.0  010000\n"
	"0.0  0.0  1.0  0.0  0.0  001000\n"
	"0.0  0.0  0.0  1.0  0.0  000100\n"
	"0.0  0.0  0.0  0.0  1.0  000010\n"
	"1.0  0.0  0.0  0.0  0.0  000001\n"
)
with open(IDEAL_MAP_FILENAME, "w") as file:
	file.write(IDEAL_MAP)


def plot_pareto_fronts(*designs: str | tuple[float] | tuple[float, float, float]):
	fronts = []

	for design in designs:
		if type(design) is str:
			name = os.path.basename(str(design))
			label = str(design)
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
			front = loadtxt(f"generated/{name}_pareto_front.txt")
			resolutions = front[:, 0]
			efficiencies = front[:, 1]
			hyperparameters = front[:, 2:]
			logging.info(f"re-loaded a previously calculated pareto front for {name}")

		else:
			if type(design) is str:
				resolutions, efficiencies, hyperparameters = find_pareto_front_of_magnet_design(design)
			elif len(design) == 3:
				foil_diameter, aperture_distance, aperture_diameter = design
				resolutions, efficiencies, hyperparameters = find_pareto_front_of_aperture_design(foil_diameter, aperture_distance, aperture_diameter)
			elif len(design) == 1:
				foil_diameter, = design
				resolutions, efficiencies, hyperparameters = find_pareto_front_of_collimator(foil_diameter)
			else:
				raise ValueError(f"wth does {design} mean?")

			savetxt(
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

	colors = ["#e6b648", "#2abd41", "#04d6e7", "r", "b", "m"]
	for (resolutions, efficiencies, hyperparameters, label), color in zip(fronts, colors):
		linestyle = "dashed" if "Ideal" in label else "solid"
		performance_ax.plot(resolutions, efficiencies, label=label, linestyle=linestyle, linewidth=2, color=color)
		parameter_ax.plot(array(hyperparameters)[:, 2]*100, array(hyperparameters)[:, 3]*50, label=label, color=color)

	if len(fronts) > 1:
		parameter_ax.legend()

	performance_ax.set_xlim(0, 800)
	performance_ax.xaxis.set_major_locator(MultipleLocator(200))
	performance_ax.set_yscale("log")
	performance_ax.set_ylim(1e-14, 1e-12)
	performance_ax.set_xlabel("Resolution (keV)")
	performance_ax.set_ylabel("Efficiency")
	performance_ax.set_title("Performance at 16.75 MeV")
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
	efficiencies = geomspace(1e-14, 1e-12, 9)  # Compton counts per photon born in the plasma
	resolutions = empty_like(efficiencies)
	hyperparameters = []
	with ProcessPoolExecutor(max_workers=9) as executor:
		for i, efficiency in enumerate(efficiencies):
			foil_thickness = optimize_foil_thickness(foil_diameter, aperture_distance, aperture_diameter, efficiency, executor)
			resolutions[i] = calculate_resolution_of_map(
				foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
				map_filename=IDEAL_MAP_FILENAME, central_energy=10, order=1,
				tilt_angle=0, arc_radius=inf, executor=executor,
				num_recoil_particles=10_000)
			hyperparameters.append((foil_diameter, foil_thickness, aperture_distance, aperture_diameter))
	return resolutions, efficiencies, hyperparameters


def find_pareto_front_of_collimator(foil_diameter: float) -> tuple[Sequence[float], Sequence[float], Sequence[tuple[float, float, float, float]]]:
	"""
	find the range of achievable performances for a given foil diameter,
	ignoring ion-optics and varying foil thickness, aperture distance, and aperture diameter
	"""
	# n.b. 1e-13 means you get about 1 Compton count per MJ of fusion
	efficiencies = geomspace(1e-14, 1e-12, 9)  # Compton counts per photon born in the plasma

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
		resolution = calculate_resolution_of_map(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			map_filename=IDEAL_MAP_FILENAME, central_energy=10, order=1,
			tilt_angle=0, arc_radius=inf, executor=None,
			num_recoil_particles=10_000)
		logging.info(f"{efficiency:.3g}: [{foil_diameter:.4g}, {foil_thickness:.4g}, {aperture_distance:.4g}, {aperture_diameter:.4g}] -> {resolution:.2f}")
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
			fatol=inf,
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
	# n.b. 1e-13 means you get about 1 Compton count per MJ of fusion
	efficiencies = geomspace(1e-14, 1e-12, 9)  # Compton counts per photon born in the plasma

	resolutions, hyperparameters = zip(*run_concurrently(
		find_suitable_configuration,
		efficiencies, filename,
	))

	return resolutions, efficiencies, hyperparameters


def find_suitable_configuration(
		efficiency: float, magnet_system_filename: str) -> tuple[float, tuple[float, float, float, float]]:

	magnet_system_info = run_cosy(
		load_script(magnet_system_filename),
		parameter_vector=None, output_mode="none")

	map_filename = f"generated/proc{current_process().pid}_map.txt"
	with open(map_filename, "w") as file:
		file.write(magnet_system_info["map"])
	central_energy = magnet_system_info["central_energy"]
	max_foil_diameter = magnet_system_info["foil_width"]
	aperture_distance = magnet_system_info["drift_pre_aperture"]
	max_aperture_diameter = magnet_system_info["aperture_width"]
	detector_tilt_angle = degrees(magnet_system_info["p_detector_tilt"])
	detector_arc_radius = -100/magnet_system_info["p_detector_curvature"]

	def objective(hyperparameters: Sequence[float]) -> float:
		foil_diameter, aperture_diameter = hyperparameters
		foil_thickness = optimize_foil_thickness(
			foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
		resolution = calculate_resolution_of_map(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			map_filename,
			central_energy=central_energy,
			order=ORDER,
			tilt_angle=detector_tilt_angle,
			arc_radius=detector_arc_radius,
			executor=None,
			num_recoil_particles=10_000)
		logging.info(f"{efficiency:.3g}: [{foil_diameter:.6g}, {foil_thickness:.1f}, {aperture_distance:.6g}, {aperture_diameter:.6g}] -> {resolution:.2f}")
		return resolution

	solution = optimize.minimize(
		objective,
		[max_foil_diameter, max_aperture_diameter],
		method="Nelder-Mead",
		bounds=[
			(.01, max_foil_diameter),
			(.01, max_aperture_diameter),
		],
		options=dict(
			initial_simplex=[
				[max_foil_diameter, max_aperture_diameter],
				[max_foil_diameter*0.6, max_aperture_diameter*0.8],
				[max_foil_diameter*0.8, max_aperture_diameter*0.6],
			],
			xatol=0.001,  # it doesn't need to be more precise than the nearest millimeter
			fatol=inf,
			disp=True,
		)
	)
	print(solution)

	foil_diameter, aperture_diameter = solution.x
	foil_thickness = optimize_foil_thickness(
		foil_diameter, aperture_distance, aperture_diameter, efficiency, executor=None)
	exact_resolution = calculate_resolution_of_map(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
		map_filename,
		central_energy=central_energy,
		order=ORDER,
		tilt_angle=detector_tilt_angle,
		arc_radius=detector_arc_radius,
		executor=None,
		num_recoil_particles=100_000)
	return exact_resolution, (foil_diameter, foil_thickness, aperture_distance, aperture_diameter)


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
	plot_pareto_fronts(
		"generated/MERGS500_electron_optics",
		"generated/MERGS350_electron_optics",
		"generated/MERGS250_electron_optics")
