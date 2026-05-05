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
from numpy import any, log1p, inf, degrees, zeros, isfinite, array, full, nan, meshgrid, seterr, log, sqrt

from electron_optics import optimize_electron_optics, load_script, run_cosy

# silence this error
seterr(divide="ignore")

# turn off pair production
for material in FOIL_MATERIALS.values():
	for interaction in material["interactions"][:]:
		if interaction["type"] == "pair_production":
			material["interactions"].remove(interaction)

# fix the annoying plot style changes that Nick makes in MPR_Tools
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.5


# avoid using super high orders when you're just trying to work out the aperture geometry
SCAN_ORDER = 5
# go up to 9th order in the final scan since it's the highest order supported by MPR_Tools
FINAL_ORDER = 9


def optimize_hyperparameters(name: str, target_resolution: float, target_efficiency: float):
	"""
	come up with a spectrometer design that meets the given resolution and efficiency
	for the lowest cost possible, and save it to disk at the given name
	:param name: the final filename at which to save the COSY file
	:param target_resolution: the desired resolution at 16.75 MeV, in keV
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:return: the optimal foil diameter, foil thickness, aperture distance, and aperture diameter
	"""
	foil_diameters = array([.03])
	aperture_distances = array([.25, .30, .40, .50, .60, .80, 1.00])
	aperture_diameters = array([.015, .02, .025, .03, .035, .04, .05])
	frugalities = array([0.0001, 0.001, 0.01, 0.1, 1.0])
	aperture_distance_grid, aperture_diameter_grid = meshgrid(aperture_distances, aperture_diameters, indexing="ij")
	resolution_grid = full((aperture_distances.size, aperture_diameters.size), 5000)
	cost_grid = full((aperture_distances.size, aperture_diameters.size), nan)
	best_cost = inf
	best: Optional[tuple[float, float, float, float, float]] = None

	with ProcessPoolExecutor(max_workers=8) as executor:
		for foil_diameter in foil_diameters:
			for j, aperture_distance in enumerate(aperture_distances):
				for k, aperture_diameter in enumerate(aperture_diameters):
					for frugality in frugalities:

						# calculate the foil thickness
						foil_thickness = optimize_foil_thickness(
							foil_diameter, aperture_distance, aperture_diameter, target_efficiency, executor=executor)
						foil_resolution = calculate_foil_broadening(foil_thickness)
						if foil_resolution > target_resolution:
							break

						# run the inner optimization scan
						try:
							_, resolution, cost = optimize_parameters(
								foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality,
								executor=executor, final=False)
						except RuntimeError as e:
							print(e)
							break  # skip the higher frugalities since it _probably_ won't work there if it didn't work here

						# save the results
						if resolution_grid[j, k] <= target_resolution:
							this_is_better_than_whats_in_the_grid = resolution <= target_resolution and cost < cost_grid[j, k]
						else:
							this_is_better_than_whats_in_the_grid = resolution < resolution_grid[j, k]
						if this_is_better_than_whats_in_the_grid:
							resolution_grid[j, k] = resolution
							cost_grid[j, k] = cost
						if resolution <= target_resolution and cost < best_cost:
							best = (foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality)
							best_cost = cost

						# make a plot so the user can see our progress
						if any(isfinite(cost_grid)):
							if any(resolution_grid <= target_resolution):
								vmax = cost_grid.max(where=resolution_grid <= target_resolution, initial=-inf)
							else:
								vmax = cost_grid.max()
							fig = plt.figure(figsize=(5.5, 3), facecolor="none")
							ax = fig.add_subplot()
							mesh = ax.pcolormesh(
								aperture_distances, aperture_diameters, cost_grid.T, vmax=vmax, shading="gouraud")
							ax.scatter(
								aperture_distance_grid, aperture_diameter_grid, c=cost_grid, vmax=vmax)
							ax.contour(
								aperture_distances, aperture_diameters, resolution_grid.T,
								levels=[target_resolution], colors=["k"])
							ax.set_xlabel("Aperture distance (cm)")
							ax.set_ylabel("Aperture diameter (cm)")
							plt.colorbar(mesh).set_label("Cost")
							fig.tight_layout()
							fig.savefig("generated/hyperparameter_optimization.pdf")
							plt.close(fig)

						# if the resolution requirement was not met here, you can skip the higher frugalities
						if resolution > target_resolution:
							break

	if best is None:
		print("none of these met the resolution requirement.  it's probably not possible.")
		raise RuntimeError("impossible requirements")

	else:
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality = best
		print(f"the best one was [{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality}], "
		      f"which had a thickness of {foil_thickness:.1f} μm and cost {best_cost} $")

		# calculate and save the optimal magnet parameters
		magnet_parameters, _, _ = optimize_parameters(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality,
			executor=None, final=True, save_name=f"{name}_electron_optics")
		print(f"has been saved to {name}_electron_optics!")
		return foil_diameter, foil_thickness, aperture_distance, aperture_diameter, magnet_parameters


def optimize_parameters(
		foil_diameter: float, foil_thickness: float, aperture_distance: float, aperture_diameter: float,
		frugality: float, executor: Optional[Executor], final=True, save_name: str = None) -> tuple[list[float], float, float]:
	"""
	for a given foil/aperture dimensions and frugality, find the optimal magnet system that achieves the given efficiency with the best resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param frugality: how much to wey cost when evaluating performance
	:param executor: the process pool to use for the multiprocessed bits
	:param final: whether to make this calculation accurate (otherwise we'll just do something quick and easy)
	:param save_name: a filename at which to save the optimal magnet parameters
	:return: the optimal magnet parameters, resolution at 16.75 MeV (keV), and cost (emerald broams)
	"""
	order = FINAL_ORDER if final else SCAN_ORDER

	# check the permanent cache
	if save_name is None:
		try:
			parameters, cost, perfect_match = find_nearest_in_permanent_cache(
				foil_diameter, aperture_distance, aperture_diameter, frugality, order)
		except ValueError:
			parameters, cost = None, None
			perfect_match = False
	else:
		parameters, cost = None, None
		perfect_match = False

	# optimize the magnet parameters
	if perfect_match:
		print(f"loading an optimized magnet system for [{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality}, {order}]...")
	else:
		print(f"optimizing the magnet system {'from scratch' if parameters is None else 'based on a prior one'} for [{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality}, {order}]...")
		parameters, optical_resolution, cost = optimize_electron_optics(
			foil_diameter, aperture_distance, aperture_diameter, frugality,
			initial_guess=parameters, order=order, save_name=save_name)
		append_to_permanent_cache(foil_diameter, aperture_distance, aperture_diameter, frugality, order, parameters, cost)

	# calculate the resolution
	total_resolution = calculate_resolution(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter, "mergs_electron_optics", parameters,
		order=order, executor=executor)

	# print, save, and return
	print(f"\t{total_resolution:.0f} keV, {cost:.2f} $")
	return parameters, total_resolution, cost


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
		16.75, num_samples=500_000, executor=executor, max_workers=8 if executor else 1)
	nuclear_efficiency = 2.4e-5*.934/(17.6*1.6e-19)  # photons/MJ (only counting the 93% that fall above 10 MeV)
	collimator_efficiency = 1e-9*(foil_diameter/.03)**2
	target_foil_efficiency = target_efficiency/nuclear_efficiency/collimator_efficiency
	target_scattering_efficiency = target_foil_efficiency/geometric_efficiency
	total_cross_section = 0
	scattering_cross_section = 0
	for interaction in foil.interactions:
		total_cross_section += interaction.get_cross_section(16.75)
		if interaction.generates_recoil_particles:
			scattering_cross_section += interaction.get_cross_section(16.75)
	return -log1p(-target_scattering_efficiency/scattering_cross_section*total_cross_section)/total_cross_section/1e-6


def calculate_resolution(
		foil_diameter: float, foil_thickness: float,
		aperture_distance: float, aperture_diameter: float,
		magnet_system_filename: Optional[str], parameters: Optional[list[float]],
		executor: Optional[Executor], order: Optional[int] = None) -> float:
	"""
	evaluate a complete design to determine its total energy resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param magnet_system_filename: name of a file containing the electron optics configuration and default parameters,
	                               or None to neglect the electron optics and just worry about the foil and aperture
	:param parameters: the electron optics parameters, if different from what's currently in the file
	:param order: the number of COSY orders to use in the calculation
	:param executor: the process pool to use for the multiprocessed bits
	:return: resolution at 16.75 MeV (keV)
	"""
	# first make sure the foil is a reasonable thickness
	foil_broadening = calculate_foil_broadening(foil_thickness)
	if foil_broadening > 5000:  # if it's really really thick, skip this calculation as it might not work properly
		return 5000

	if magnet_system_filename is not None:
		# use COSY to get the transfer map matrix and optimal detector shape
		if order is None:
			raise TypeError("You have to pass an order if we're using COSY")
		cosy_script = load_script(magnet_system_filename, foil_diameter, aperture_distance, aperture_diameter, order)
		cosy_outputs = run_cosy(cosy_script, parameters, output_mode="none")
		map_filename = f"generated/proc{multiprocessing.current_process().pid}_map.txt"
		with open(map_filename, "w") as file:
			file.write(cosy_outputs["map"])
		central_energy = cosy_outputs["central_energy"]
		tilt_angle = degrees(cosy_outputs["p_detector_tilt"])
		if cosy_outputs["p_detector_curvature"] != 0:
			arc_radius = -100/cosy_outputs["p_detector_curvature"]
		else:
			arc_radius = -inf

	else:
		# or make the map ideal so that we don't have to worry about the magnets
		order = 1
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
		central_energy = 10
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
			reference_energy=central_energy, min_energy=6, max_energy=18,
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
			incident_energy=16.75, num_recoil_particles=100_000, map_order=order,
			executor=executor, max_workers=8 if executor else 1)
	except ValueError as e:
		# inconsistencies in how we do the transfer map might cause this to fail (TODO: if I make MPR_Tools use multiple in series then we can probably remove this)
		if str(e) == "Some of these rays don't hit the curved detector.":
			print("MPR_Tools had an invalid ray geometry with the detector, even though COSY thought it was fine.")
			return inf  # shikatanai.  just avoid that geometry, I gess, since the map is probably not even converged.
		# an aperture that's much smaller than the foil can make this calculation arbitrarily slow.
		elif str(e) == "Failed to generate electron":
			print("The aperture geometry is failing.  Consider increasing the allowed number of attempts.")
			return inf  # return inf to discourage that, but also print in case it's happening a lot
		# if it's something else, then I'm scared and confused and we should probably stop.
		else:
			raise e

	return min(5000, abs(resolution))  # don't report resolutions above 5 MeV because it gets hard to define then


def calculate_foil_broadening(foil_thickness: float) -> float:
	""" calculate the resolution for a perfect ion-optic system, just accounting for broadening """
	foil = ConversionFoil(0, foil_thickness, 0, 0, foil_material="B")
	initial_energy = foil.interactions[0].get_recoil_energy(16.75, 0., None)
	min_exit_energy = foil.calculate_stopping_power_loss(initial_energy, foil_thickness*1e-6)
	return (initial_energy - min_exit_energy)*1000


def find_nearest_in_permanent_cache(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int,
) -> tuple[list[float], float, bool]:
	best_distance = inf
	best_outputs = None
	try:
		with open("generated/magnet_optimization_cache.txt", mode="r") as file:
			for line in file.readlines():
				input_string, output_string = line.split(": ")
				cached_foil_diameter, cached_aperture_distance, cached_aperture_diameter, cached_frugality, cached_order = (
					float(x) for x in input_string.split(", "))
				distance = sqrt(
					((foil_diameter - cached_foil_diameter)/0.15)**2 +
					((aperture_distance - cached_aperture_distance)/0.008)**2 +
					((aperture_diameter - cached_aperture_diameter)/0.005)**2 +
					((log(frugality) - log(cached_frugality))/5)**2 +
					((order - cached_order)/5)**2
				)
				if distance < best_distance:
					best_distance = distance
					best_outputs = [float(x) for x in output_string.split(", ")]
	except FileNotFoundError:
		raise ValueError("cache is absent")
	if best_outputs is None:
		raise ValueError("cache is empty")
	else:
		parameters = best_outputs[:-1]
		cost = best_outputs[-1]
		return parameters, cost, best_distance == 0


def append_to_permanent_cache(foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int, parameters: list[float], cost: float):
	with open("generated/magnet_optimization_cache.txt", mode="a") as file:
		file.write(f"{foil_diameter}, {aperture_distance}, {aperture_diameter}, {frugality}, {order}: "
		           f"{', '.join(str(x) for x in parameters)}, {cost}\n")


if __name__ == "__main__":
	optimize_hyperparameters("MERGS cheap", 500, 1)
