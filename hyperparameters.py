"""
code for scanning hyperparameters to find the set of all good designs
"""
import multiprocessing
from concurrent.futures import Executor
from concurrent.futures.process import ProcessPoolExecutor
from typing import Optional

from MPR_Tools import MPRSpectrometer, ConversionFoil, Hodoscope, PerformanceAnalyzer
from MPR_Tools.config.constants import FOIL_MATERIALS
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from numpy import log1p, inf, degrees, zeros, isfinite, array
from scipy import optimize

from electron_optics import optimize_electron_optics, load_script, run_cosy


# turn off pair production
for material in FOIL_MATERIALS.values():
	for interaction in material["interactions"][:]:
		if interaction["type"] == "pair_production":
			material["interactions"].remove(interaction)

plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.2


def optimize_hyperparameters(name: str, target_resolution: float, target_efficiency: float):
	"""
	come up with a spectrometer design that meets the given resolution and efficiency
	for the lowest cost possible, and save it to disk at the given name
	:param name: the final filename at which to save the COSY file
	:param target_resolution: the desired resolution at 16.7 MeV, in keV
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:return: the optimal foil diameter, foil thickness, aperture distance, and aperture diameter
	"""
	# in general we want as large a foil as possible, as long as we can put the aperture far enuff away.  but it doesn't make sense for it to be bigger than the collimator.  so fix it at 3 cm.
	foil_diameter = 0.03

	all_aperture_distances = []
	all_aperture_diameters = []
	all_resolutions = []
	all_costs = []

	with ProcessPoolExecutor(max_workers=8) as executor:

		def objective(aperture_dimensions):
			# run the inner optimization scan
			_, _, resolution, cost = optimize_parameters_and_frugality(
				foil_diameter, aperture_dimensions[0], aperture_dimensions[1],
				target_resolution, target_efficiency, executor)
			# save the results
			all_aperture_distances.append(aperture_dimensions[0]/1e-2)
			all_aperture_diameters.append(aperture_dimensions[1]/1e-2)
			all_resolutions.append(resolution)
			all_costs.append(cost)
			# make a plot so the user can see our progress
			fig = plt.figure(figsize=(5.5, 3), facecolor="none")
			ax = fig.add_subplot()
			ax.grid()
			ax.scatter(
				array(all_aperture_distances)[~isfinite(all_costs)],
				array(all_aperture_diameters)[~isfinite(all_costs)],
				s=20, zorder=2, c="k", marker="x")
			ax.scatter(
				array(all_aperture_distances)[isfinite(all_costs)],
				array(all_aperture_diameters)[isfinite(all_costs)],
				s=10, zorder=2, c=array(all_costs)[isfinite(all_costs)], vmax=max(array(all_costs)[isfinite(all_costs)][-20:]))
			for i in range(max(0, len(all_costs) - 10), len(all_costs)):
				ax.text(all_aperture_distances[i], all_aperture_diameters[i], f"{all_resolutions[i]:.5g} keV, {all_costs[i]:.5g} $")
			ax.set_xlabel("Aperture distance (cm)")
			ax.set_ylabel("Aperture diameter (cm)")
			fig.tight_layout()
			fig.savefig("generated/hyperparameter_optimization.pdf")
			plt.close(fig)
			return cost

		# calculate the optimal hyperparameters
		solution = optimize.minimize(
			objective,
			[.50, .03],
			method="Nelder-Mead",
			bounds=[(.20, 3.00), (.001, .10)],
			options=dict(
				disp=True,
				initial_simplex=[
					[.40, .02],
					[.50, .03],
					[.30, .04],
				],
				xatol=0.001,
			),
		)
		print(solution)
		aperture_distance, aperture_diameter = solution.x

		# calculate and save the optimal magnet parameters
		foil_thickness, magnet_parameters, _, _ = optimize_parameters_and_frugality(
			foil_diameter, aperture_distance, aperture_diameter,
			target_resolution, target_efficiency, executor, save_name=f"{name}_electron_optics")

	return foil_diameter, foil_thickness, aperture_distance, aperture_diameter, magnet_parameters


def optimize_parameters_and_frugality(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		target_resolution: float, target_efficiency: float,
		executor: Optional[Executor], save_name: str = None) -> tuple[float, Optional[list[float]], float, float]:
	"""
	for a given foil and aperture dimensions, find the optimal magnet system and foil thickness that achieves the given resolution and efficiency for the lowest cost
	:param foil_diameter: the foil diameter in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param target_resolution: the desired resolution at 16.7 MeV, in keV
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:param executor: the process pool to use for the multiprocessed bits
	:param save_name: a filename at which to save the optimal magnet parameters
	:return: the optimal foil thickness (μm), optimal magnet parameters, resolution at 16.7 MeV (keV), and cost (emerald broams)
	"""
	cache: dict[float, tuple[list[float], float, float]] = {}

	foil_thickness = optimize_foil_thickness(
		foil_diameter, aperture_distance, aperture_diameter, target_efficiency, executor)
	best_possible_resolution = calculate_resolution(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
		magnet_system_filename=None, parameters=None, executor=executor)

	if best_possible_resolution > target_resolution:
		print(f"It is not possible to make a system with the hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m] that has efficiency {target_efficiency:.2g} and resolution {target_resolution:.0f} keV.")
		return foil_thickness, None, best_possible_resolution, inf
	best_practical_resolution = optimize_parameters(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter, 0.001,
		cache, executor)[1]
	if best_practical_resolution > target_resolution:
		print(f"It is infeasible to make a system with the hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m] that has efficiency {target_efficiency:.2g} and resolution {target_resolution:.0f} keV.")
		return foil_thickness, None, best_practical_resolution, inf
	cheapest_resolution = optimize_parameters(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter, 100.,
		cache, executor)[1]
	if cheapest_resolution <= target_resolution:
		print(f"The hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m] automatically achieve efficiency {target_efficiency:.2g} and resolution {cheapest_resolution:.0f} keV.")
		parameters, resolution, cost = cache[100.]
		return foil_thickness, parameters, resolution, cost

	optimum = optimize.root_scalar(
		lambda frugality: optimize_parameters(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality,
			cache, executor, save_name)[1] - target_resolution,
		bracket=(0.001, 100),
		rtol=0.02,  # note the large tolerance; we don't need to get the frugality _that_ precisely
	)
	magnet_parameters, resolution, cost = cache[optimum.root]

	print(f"For the hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m], a cost of {cost:.3f} $ achieves efficiency {target_efficiency:.2g} and resolution {resolution:.0f} keV")

	return foil_thickness, magnet_parameters, resolution, cost


def optimize_parameters(
		foil_diameter: float, foil_thickness: float, aperture_distance: float, aperture_diameter: float,
		frugality: float, cache: dict[float, tuple[list[float], float, float]], executor: Optional[Executor],
		save_name: str = None) -> tuple[list[float], float, float]:
	"""
	for a given foil/aperture dimensions and frugality, find the optimal magnet system that achieves the given efficiency with the best resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param frugality: how much to wey cost when evaluating performance
	:param cache: the cache of previus optimizations with these hyperparameters and target efficiency
	:param executor: the process pool to use for the multiprocessed bits
	:param save_name: a filename at which to save the optimal magnet parameters
	:return: the optimal magnet parameters, resolution at 16.7 MeV (keV), and cost (emerald broams)
	"""
	# check the local cache
	if frugality in cache:
		return cache[frugality]

	# check the permanent cache
	try:
		parameters, cost = find_in_permanent_cache(foil_diameter, aperture_distance, aperture_diameter, frugality)
		print("loading a previus optimized magnet system from the cache...")

	# optimize the magnet parameters
	except ValueError:
		print(f"optimizing the magnet system for [{foil_diameter}, {aperture_distance}, {aperture_diameter}]...")
		parameters, optical_resolution, cost = optimize_electron_optics(
			foil_diameter, aperture_distance, aperture_diameter, frugality,
			save_name=save_name)
		append_to_permanent_cache(foil_diameter, aperture_distance, aperture_diameter, frugality, parameters, cost)

	# calculate the resolution
	total_resolution = calculate_resolution(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter, "mergs_electron_optics", parameters,
		executor)

	# print, save, and return
	print(f"\t{total_resolution:.0f} keV, {cost:.2f} $")
	cache[frugality] = parameters, total_resolution, cost
	return cache[frugality]


def optimize_foil_thickness(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		target_efficiency: float,
		executor: Optional[Executor]) -> float:
	"""
	for a given foil radius and material, calculate the thickness that achieves the given efficiency
	:param foil_diameter: the foil diameter in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:param executor: the process pool to use for the multiprocessed bits
	:return: the optimal foil thickness in μm
	"""
	# first use a quick MC to calculate the geometric efficiency
	foil = ConversionFoil(foil_diameter/2, 1, aperture_distance, aperture_diameter/2, foil_material="B")
	_, geometric_efficiency, _ = foil.calculate_efficiency(
		16.7, num_samples=100_000, executor=executor, max_workers=8 if executor else 1)
	nuclear_efficiency = 2.4e-5*.89/(17.6*1.6e-19)  # photons/MJ (only counting the 89% that fall above 11 MeV)
	collimator_efficiency = 1.5*7e-10*(foil_diameter/.03)**2
	target_foil_efficiency = target_efficiency/nuclear_efficiency/collimator_efficiency
	target_scattering_efficiency = target_foil_efficiency/geometric_efficiency
	total_cross_section = 0
	scattering_cross_section = 0
	for interaction in foil.interactions:
		total_cross_section += interaction.get_cross_section(16.7)
		if interaction.generates_recoil_particles:
			scattering_cross_section += interaction.get_cross_section(16.7)
	return -log1p(-target_scattering_efficiency/scattering_cross_section*total_cross_section)/total_cross_section/1e-6


def calculate_resolution(
		foil_diameter: float, foil_thickness: float,
		aperture_distance: float, aperture_diameter: float,
		magnet_system_filename: Optional[str], parameters: Optional[list[float]],
		executor: Optional[Executor]) -> float:
	"""
	evaluate a complete design to determine its total energy resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param magnet_system_filename: name of a file containing the electron optics configuration and default parameters,
	                               or None to neglect the electron optics and just worry about the foil and aperture
	:param parameters: the electron optics parameters, if different from what's currently in the file
	:param executor: the process pool to use for the multiprocessed bits
	:return: resolution at 16.7 MeV (keV)
	"""
	# first make sure the foil is a reasonable thickness
	foil_broadening = calculate_foil_broadening(foil_thickness)
	if foil_broadening > 5000:  # if it's really really thick, skip this calculation as it might not work properly
		return 5000

	if magnet_system_filename is not None:
		# use COSY to get the transfer map matrix and optimal detector shape
		cosy_script = load_script(magnet_system_filename, foil_diameter, aperture_distance, aperture_diameter, order=3)
		cosy_outputs = run_cosy(cosy_script, parameters, output_mode="none")
		map_filename = f"generated/proc{multiprocessing.current_process().pid}_map.txt"
		with open(map_filename, "w") as file:
			file.write(cosy_outputs["map"])
		tilt_angle = degrees(cosy_outputs["p_detector_tilt"])
		arc_radius = 100/cosy_outputs["p_detector_curvature"]

	else:
		# or make the map ideal so that we don't have to worry about the magnets
		map_filename = f"generated/ideal_map.txt"
		ideal_map = (
			"0.0  0.0  0.0  0.0  0.0  100000\n"
			"0.0  0.0  0.0  0.0  0.0  010000\n"
			"0.0  0.0  1.0  0.0  0.0  001000\n"
			"0.0  0.0  0.0  1.0  0.0  000100\n"
			"0.0  0.0  0.0  0.0  1.0  000010\n"
			"1.0  0.0  0.0  0.0  0.0  000001\n"
		)
		with open(map_filename, "w") as file:
			file.write(ideal_map)
		tilt_angle = 0
		arc_radius = inf

	# use MPR_Tools to calculate the resolution
	monte_carlo = PerformanceAnalyzer(
		MPRSpectrometer(
			conversion_foil=ConversionFoil(
				foil_radius=foil_diameter/2,
				thickness=foil_thickness,
				aperture_distance=aperture_distance,
				aperture_radius=aperture_diameter/2,
				foil_material="B",
			),
			transfer_map_path=map_filename,
			reference_energy=13.5, min_energy=9.45, max_energy=17.55,
			hodoscope=Hodoscope(
				tilt_angle=tilt_angle,
				arc_radius=arc_radius,
				channels=zeros((2, 2))
			),
			run_directory="generated/monte-carlo-dump/",
		),
	)

	try:
		_, _, resolution, _ = monte_carlo.analyze_monoenergetic_performance(
			incident_energy=16.7, num_recoil_particles=10_000, executor=executor, max_workers=8 if executor else 1)
	except ValueError:
		return inf  # an aperture that's much smaller than the foil can make this calculation arbitrarily slow.  return inf to discourage that.

	return min(5000, resolution)  # don't report resolutions above 5 MeV because it gets hard to define then


def calculate_foil_broadening(foil_thickness: float) -> float:
	""" calculate the resolution for a perfect ion-optic system, just accounting for broadening and kinematics """
	foil = ConversionFoil(0, foil_thickness, 0, 0, foil_material="B")
	initial_energy = foil.interactions[0].get_recoil_energy(16.7, 0., None)
	min_exit_energy = foil.calculate_stopping_power_loss(initial_energy, foil_thickness*1e-6)
	return (initial_energy - min_exit_energy)*1000


def find_in_permanent_cache(foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float) -> tuple[list[float], float]:
	try:
		key_string = f"{foil_diameter}, {aperture_distance}, {aperture_diameter}, {frugality}"
		with open("generated/magnet_optimization_cache.txt", mode="r") as file:
			for line in file.readlines():
				input_string, output_string = line.split(": ")
				if key_string == input_string:
					outputs = [float(x) for x in output_string.split(", ")]
					parameters = outputs[:-1]
					cost = outputs[-1]
					return parameters, cost
		raise ValueError("not in cache")
	except FileNotFoundError:
		raise ValueError("cache is empty")


def append_to_permanent_cache(foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float, parameters: list[float], cost: float):
	with open("generated/magnet_optimization_cache.txt", mode="a") as file:
		file.write(f"{foil_diameter}, {aperture_distance}, {aperture_diameter}, {frugality}: {', '.join(str(x) for x in parameters)}, {cost}\n")


if __name__ == "__main__":
	optimize_hyperparameters("MERGS", 500, 1)
