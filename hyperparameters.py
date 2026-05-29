"""
code for scanning hyperparameters to find the set of all good designs
"""
import logging
import multiprocessing
import traceback
from concurrent.futures import Executor
from concurrent.futures.process import ProcessPoolExecutor
from typing import Optional, Sequence

import multiprocessing_logging
from MPR_Tools import MPRSpectrometer, ConversionFoil, Hodoscope, PerformanceAnalyzer
from MPR_Tools.config.constants import FOIL_MATERIALS
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import any, log1p, inf, degrees, zeros, isfinite, array, full, nan, seterr, log, sqrt, nanmin, nanmedian, \
	empty, shape

from draw_magnets import draw_magnets
from electron_optics import optimize_electron_optics, load_script, run_cosy

# try to silence this error
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

# configure the logger
logging.basicConfig(
	level=logging.INFO, filename="out.log", encoding="utf-8",
	datefmt="%m-%d %H:%M:%S", format="%(asctime)s %(process)05d %(levelname)-5.5s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
multiprocessing_logging.install_mp_handler()
plt.set_loglevel("warning")


# avoid using super high orders when you're just trying to work out the aperture geometry
SCAN_ORDER = 9
# go up to 9th order in the final scan since it's the highest order supported by MPR_Tools
FINAL_ORDER = 9


file_lock = multiprocessing.Lock()

def initialize_process(new_lock):
	""" this janky function is how you do file locks in multiprocessing """
	global file_lock
	file_lock = new_lock


def optimize_hyperparameters(
		name: str, target_resolution: float, target_efficiency: float,
		min_foil_diameter=0., max_foil_diameter=inf,
		min_aperture_distance=0., max_aperture_distance=inf,
		min_aperture_diameter=0., max_aperture_diameter=inf,
):
	"""
	come up with a spectrometer design that meets the given resolution and efficiency
	for the lowest cost possible, and save it to disk at the given name
	:param name: the final filename at which to save the COSY file
	:param target_resolution: the desired resolution at 16.75 MeV, in keV
	:param target_efficiency: the desired number of Compton counts per photons born in the plasma
	:param min_foil_diameter: the smallest foil diameter to bother checking (m)
	:param max_foil_diameter: the largest foil diameter to bother checking (m)
	:param min_aperture_distance: the smallest aperture distance to bother checking (m)
	:param max_aperture_distance: the largest aperture distance to bother checking (m)
	:param min_aperture_diameter: the smallest aperture diameter to bother checking (m)
	:param max_aperture_diameter: the largest aperture diameter to bother checking (m)
	"""
	logging.info("---")
	logging.info(f"Starting optimization of '{name}' to achieve {target_resolution} keV and {target_efficiency}.")
	foil_diameters = array([.03, .02, .01])
	foil_diameters = foil_diameters[(foil_diameters >= min_foil_diameter) & (foil_diameters <= max_foil_diameter)]
	aperture_distances = array([.25, .30, .40, .50, .60, .70, .80, .90, 1.00, 1.10, 1.20, 1.30])
	aperture_distances = aperture_distances[(aperture_distances >= min_aperture_distance) & (aperture_distances <= max_aperture_distance)]
	aperture_diameters = array([.06, 0.055, .05, .045, .04, .035, .03, .025, .02, .015])
	aperture_diameters = aperture_diameters[(aperture_diameters >= min_aperture_diameter) & (aperture_diameters <= max_aperture_diameter)]
	frugalities = array([30, 50, 100, 150, 200, 300, 400, 550, 700, 850, 1000, 1200, 1400, 1600, 2000])**2
	task_grid = empty((foil_diameters.size, aperture_distances.size, aperture_diameters.size), dtype=object)
	resolution_grid = full((foil_diameters.size, aperture_distances.size, aperture_diameters.size), nan)
	cost_grid = full((foil_diameters.size, aperture_distances.size, aperture_diameters.size), nan)
	best_cost = inf
	best: Optional[tuple[float, float, float, float, float]] = None

	with ProcessPoolExecutor(max_workers=8, initializer=initialize_process, initargs=(file_lock,)) as executor:
		for i, foil_diameter in enumerate(foil_diameters):
			for j, aperture_distance in enumerate(aperture_distances):
				for k, aperture_diameter in enumerate(aperture_diameters):
					# scan frugality to calculate the other parameters
					task_grid[i, j, k] = executor.submit(
						optimize_mesoparameters,
						foil_diameter, aperture_distance, aperture_diameter,
						frugalities, target_resolution, target_efficiency,
					)

		# as the optimization tasks finish...
		for i, foil_diameter in enumerate(foil_diameters):
			for j, aperture_distance in enumerate(aperture_distances):
				for k, aperture_diameter in enumerate(aperture_diameters):
					# extract the results
					try:
						foil_thickness, local_best_resolution, local_best_cost, local_best_frugality = task_grid[i, j, k].result()
					except Exception as e:
						traceback.print_exc()
						logging.error(str(e))
						continue
					resolution_grid[i, j, k] = local_best_resolution
					cost_grid[i, j, k] = local_best_cost

					# keep track of the best ones you see
					if local_best_resolution <= target_resolution and local_best_cost < best_cost:
						best = (foil_diameter, foil_thickness, aperture_distance, aperture_diameter, local_best_frugality)
						best_cost = local_best_cost

					# make a plot so the user can see our progress
					if any(isfinite(cost_grid[i])):
						vmin = nanmin(cost_grid[i])
						if any(resolution_grid[i] <= target_resolution):
							vmed = nanmedian(cost_grid[i][resolution_grid[i] <= target_resolution])
						else:
							vmed = nanmedian(cost_grid[i])
						vmax = 2*vmed - vmin
						fig = plt.figure(figsize=(5.5, 3), facecolor="none")
						ax = fig.add_subplot()
						mesh = ax.contourf(
							aperture_distances*100, aperture_diameters*100, cost_grid[i].T,
							cmap="viridis_r", levels=MaxNLocator(10).tick_values(vmin, vmax),
							extend="max")
						mesh.set_edgecolor("face")
						if any(resolution_grid <= target_resolution):
							ax.contourf(
								aperture_distances*100, aperture_diameters*100, resolution_grid[i].T,
								levels=[0, target_resolution, inf], colors=["none", "k"])
						ax.set_xlabel("Aperture distance (cm)")
						ax.set_ylabel("Aperture diameter (cm)")
						plt.colorbar(mesh, extend="max").set_label("Cost")
						fig.tight_layout()
						fig.savefig(f"generated/hyperparameter_optimization_{name}_{foil_diameter*100}cm.pdf")
						plt.close(fig)

		if best is None:
			logging.error("none of these met the resolution requirement.  it's probably not possible.")

		else:
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality = best
			logging.info(f"the best one was [{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality:.1e}], "
			             f"which had a foil thickness of {foil_thickness:.1f} μm and cost {best_cost:.2f} $")

			# calculate and save the optimal magnet parameters
			optimize_parameters(
				foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality,
				final=True, save_name=f"{name}_electron_optics")
			draw_magnets(f"generated/{name}_electron_optics")
			logging.info(f"has been saved to {name}_electron_optics!")


def optimize_mesoparameters(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		frugalities: Sequence[float], target_resolution: float, target_efficiency: float) -> tuple[float, float, float, float]:
	"""
	for a given foil/aperture dimensions, find the foil thickness, frugality, and magnet system that achieves
	the given resolution and efficiency for the lowest cost
	:param foil_diameter: the foil diameter in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param frugalities: the frugality values to scan thru
	:param target_resolution: the desired resolution at 16.75 MeV, in keV
	:param target_efficiency: the desired number of Compton counts per photons born in the plasma
	:return: the optimal foil thickness (μm), the best resolution obtained (keV),
	         the lowest cost that met the resolution requirement (emerald broams),
	         and the frugality corresponding to the best cost
	"""
	best_resolution = 5000
	best_cost = nan
	best_frugality = nan

	# calculate the foil thickness
	foil_thickness = optimize_foil_thickness(
		foil_diameter, aperture_distance, aperture_diameter, target_efficiency, executor=None)
	foil_resolution = calculate_foil_broadening(foil_thickness)
	if foil_resolution > target_resolution:
		logging.info(f"skipping thru [{foil_diameter}, {aperture_distance}, {aperture_diameter}] as the foil broadening is already {foil_resolution:.0f} keV")
		return foil_thickness, best_resolution, best_cost, best_frugality

	logging.info(f"beginning frugality scan for [{foil_diameter}, {aperture_distance}, {aperture_diameter}]...")

	frugality_queue = list(enumerate(frugalities))[::-3]  # to start off, look at every third frugality
	checked = full(shape(frugalities), False)

	while len(frugality_queue) > 0:
		i, frugality = frugality_queue.pop()
		checked[i] = True

		# run the inner optimization scan
		try:
			_, resolution, cost = optimize_parameters(
				foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality, final=False)
		except RuntimeError as e:
			# sometimes the constraints just can't be met
			if "this optimization might be impossible" in str(e):
				logging.warning(e)
				if i >= 1 and not checked[i - 1]:  # try a different frugality; sometimes that helps for some reason
					frugality_queue.append((i - 1, frugalities[i - 1]))
				continue
			# if it's something else, then I'm scared and confused and we should probably stop.
			else:
				raise
		except ValueError as e:
			# inconsistencies in how we do the transfer map might cause this to fail (TODO: if I make MPR_Tools use multiple in series then we can probably remove this)
			if str(e) == "Some of these rays don't hit the curved detector.":
				logging.warning("MPR_Tools had an invalid ray geometry with the detector, even though COSY thought it was fine.")
				if i >= 1 and not checked[i - 1]:  # just avoid that geometry, I gess, since the map is probably not even converged.  try a different frugality.
					frugality_queue.append((i - 1, frugalities[i - 1]))
				continue
			# an aperture that's much smaller than the foil can make this calculation arbitrarily slow.
			elif str(e) == "Failed to generate electron":
				logging.warning("The aperture geometry is failing.  Consider increasing the allowed number of attempts.")
				break  # go ahead and skip this aperture geometry, but also print in case it's happening a lot
			# if it's something else, then I'm scared and confused and we should probably stop.
			else:
				raise

		# save the results
		if best_resolution <= target_resolution:
			this_is_better_than_the_current_best = resolution <= target_resolution and cost < best_cost
		else:
			this_is_better_than_the_current_best = resolution < best_resolution
		if this_is_better_than_the_current_best:
			best_cost = cost
			best_frugality = frugality
		best_resolution = min(resolution, best_resolution)

		# if the resolution requirement was not met here, don't go any higher, but try stepping back in frugality
		if resolution > target_resolution:
			frugality_queue = []
			if i >= 1 and not checked[i - 1]:
				frugality_queue.append((i - 1, frugalities[i - 1]))
			if i >= 2 and not checked[i - 2]:
				frugality_queue.append((i - 2, frugalities[i - 2]))

	logging.info(f"done with [{foil_diameter}, {aperture_distance}, {aperture_diameter}]!")
	return foil_thickness, best_resolution, best_cost, best_frugality


def optimize_parameters(
		foil_diameter: float, foil_thickness: float, aperture_distance: float, aperture_diameter: float,
		frugality: float, final=True, save_name: str = None) -> tuple[Sequence[float], float, float]:
	"""
	for a given foil/aperture dimensions and frugality, find the optimal magnet system that achieves
	the given efficiency with the best resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param frugality: how much to wey cost when evaluating performance
	:param final: whether to make this calculation accurate (otherwise we'll just do something quick and easy)
	:param save_name: a filename at which to save the optimal magnet parameters
	:return: the optimal magnet parameters, resolution at 16.75 MeV (keV), and cost (emerald broams)
	"""
	order = FINAL_ORDER if final else SCAN_ORDER

	# check the permanent magnet geometry optimization cache
	try:
		parameters, cost, perfect_match = find_nearest_in_permanent_geometry_cache(
			foil_diameter, aperture_distance, aperture_diameter, frugality, order)
	except FileNotFoundError:
		parameters, cost = None, None
		perfect_match = False

	# if it wasn't in there, optimize the magnet parameters
	if parameters is None or cost is None or not perfect_match or save_name is not None:
		logging.info(f"optimizing the magnet system {'from scratch' if parameters is None else 'based on a prior one'} "
		             f"for [{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality:.1e}, {order}]...")
		parameters, optical_resolution, cost = optimize_electron_optics(
			foil_diameter, aperture_distance, aperture_diameter, frugality,
			initial_guess=parameters, method="COBYQA+SLSQP", order=order, save_name=save_name)
		if perfect_match:
			remove_from_permanent_geometry_cache(
				foil_diameter, aperture_distance, aperture_diameter, frugality, order)
		append_to_permanent_geometry_cache(
			foil_diameter, aperture_distance, aperture_diameter, frugality, order,
			parameters, cost)
		remove_from_permanent_resolution_cache(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality, order)
	else:
		logging.info(f"loading an optimized magnet system for ["
		             f"{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality:.1e}, {order}]...")

	# check the permanent Monte Carlo resolution cache
	try:
		total_resolution = load_from_permanent_resolution_cache(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality, order)
	except (KeyError, FileNotFoundError) as _:
		# if it wasn't in there, calculate the resolution
		total_resolution = calculate_resolution(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
			"mergs_electron_optics", parameters,
			order=order, executor=None)
		append_to_permanent_resolution_cache(
			foil_diameter, foil_thickness, aperture_distance, aperture_diameter, frugality, order,
			total_resolution)

	# log, save, and return
	logging.info(f"[{foil_diameter}, {aperture_distance}, {aperture_diameter}; {frugality:.1e}, {order}] "
	             f"-> {total_resolution:.0f} keV, {cost:.2f} $")
	return parameters, total_resolution, cost


def optimize_foil_thickness(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		target_efficiency: float, executor: Optional[Executor]) -> float:
	"""
	for a given foil radius and material, calculate the thickness that achieves the given efficiency
	:param foil_diameter: the foil diameter in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param target_efficiency: the desired number of Compton counts per photon born in the plasma
	:param executor: the process pool to use for the multiprocessed bits
	:return: the optimal foil thickness in μm
	"""
	foil = ConversionFoil(100*foil_diameter/2, 1, 100*aperture_distance, 100*aperture_diameter/2, foil_material="B")

	# check the permanent Monte Carlo efficiency cache
	try:
		geometric_efficiency = load_from_permanent_efficiency_cache(
			foil_diameter, aperture_distance, aperture_diameter)
	except (KeyError, FileNotFoundError) as _:
		# if it wasn't in there, use a quick MC to calculate the geometric efficiency
		_, geometric_efficiency, _ = foil.calculate_efficiency(
			16.75, num_samples=500_000, executor=executor, max_workers=8 if executor else 1)
		append_to_permanent_efficiency_cache(
			foil_diameter, aperture_distance, aperture_diameter,
			geometric_efficiency)

	collimator_efficiency = 1e-9*(foil_diameter/.03)**2
	target_foil_efficiency = target_efficiency/collimator_efficiency
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
		magnet_system_filename: str, parameters: Optional[Sequence[float]],
		executor: Optional[Executor], order: int) -> float:
	"""
	evaluate a complete design to determine its total energy resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param magnet_system_filename: name of a file containing the electron optics configuration and default parameters
	:param parameters: the electron optics parameters, if different from what's currently in the file
	:param order: the number of COSY orders to use in the calculation
	:param executor: the process pool to use for the multiprocessed bits
	:return: resolution at 16.75 MeV (keV)
	"""
	# use COSY to get the transfer map matrix and optimal detector shape
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

	return calculate_resolution_of_map(
		foil_diameter, foil_thickness, aperture_distance, aperture_diameter,
		map_filename, central_energy, tilt_angle, arc_radius, executor, order=order)


def calculate_resolution_of_map(
	foil_diameter: float, foil_thickness: float,
	aperture_distance: float, aperture_diameter: float,
	map_filename: str, central_energy: float, tilt_angle: float, arc_radius: float,
	executor: Optional[Executor], order: int, num_recoil_particles=100_000) -> float:
	"""
	evaluate a spectrometer configuration to determine its total energy resolution
	:param foil_diameter: the foil diameter in m
	:param foil_thickness: the foil thickness in μm
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param map_filename: name of a file containing the electron optics map coefficients
	:param central_energy: the central energy in MeV to use with the map
	:param tilt_angle: the detector tilt in degrees
	:param arc_radius: the radius of curvature of the detector in cm
	:param executor: the process pool to use for the multiprocessed bits
	:param order: the number of COSY orders to use in the calculation
	:param num_recoil_particles: the statistics to use for calculating the resolution
	:return: resolution at 16.75 MeV (keV)
	"""
	# first make sure the foil is a reasonable thickness
	foil_broadening = calculate_foil_broadening(foil_thickness)
	if foil_broadening > 5000:  # if it's really really thick, skip this calculation as it might not work properly
		return 5000

	# use MPR_Tools to calculate the resolution
	monte_carlo = PerformanceAnalyzer(
		MPRSpectrometer(
			conversion_foil=ConversionFoil(
				foil_radius=100*foil_diameter/2,
				thickness=foil_thickness,
				aperture_distance=100*aperture_distance,
				aperture_radius=100*aperture_diameter/2,
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

	_, _, resolution, _ = monte_carlo.analyze_monoenergetic_performance(
			incident_energy=16.75, num_recoil_particles=num_recoil_particles, map_order=order,
			executor=executor, max_workers=8 if executor else 1)

	return min(5000, abs(resolution))  # don't report resolutions above 5 MeV because it gets hard to define then


def calculate_foil_broadening(foil_thickness: float) -> float:
	""" calculate the resolution for a perfect ion-optic system, just accounting for broadening """
	foil = ConversionFoil(0, foil_thickness, 0, 0, foil_material="B")
	initial_energy = foil.interactions[0].get_recoil_energy(16.75, 0., None)
	min_exit_energy = foil.calculate_stopping_power_loss(initial_energy, foil_thickness*1e-6)
	return (initial_energy - min_exit_energy)*1000


def load_from_permanent_efficiency_cache(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
) -> float:
	target = f"{foil_diameter}, {aperture_distance}, {aperture_diameter}"
	try:
		with open("generated/efficiency_cache.txt", mode="r") as file:
			for line in file.readlines():
				input_string, output_string = line.split(": ")
				if input_string == target:
					return float(output_string)
	except FileNotFoundError:
		raise FileNotFoundError("efficiency cache is absent")
	raise KeyError("desired geometry not present in cache")


def append_to_permanent_efficiency_cache(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		geometric_efficiency: float):
	with file_lock:
		with open("generated/efficiency_cache.txt", mode="a") as file:
			file.write(f"{foil_diameter}, {aperture_distance}, {aperture_diameter}: "
			           f"{geometric_efficiency}\n")


def find_nearest_in_permanent_geometry_cache(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int,
) -> tuple[Sequence[float], float, bool]:
	best_distance = inf
	best_outputs = None
	try:
		with open("generated/magnet_optimization_cache.txt", mode="r") as file:
			for line in file.readlines():
				input_string, output_string = line.split(": ")
				cached_foil_diameter, cached_aperture_distance, cached_aperture_diameter, cached_frugality, cached_order = (
					float(x) for x in input_string.split(", "))
				distance = sqrt(
					((foil_diameter - cached_foil_diameter)/0.015)**2 +
					((aperture_distance - cached_aperture_distance)/0.08)**2 +
					((aperture_diameter - cached_aperture_diameter)/0.005)**2 +
					((log(frugality) - log(cached_frugality))/10)**2 +
					((order - cached_order)/5)**2
				)
				if distance <= best_distance:
					best_distance = distance
					best_outputs = [float(x) for x in output_string.split(", ")]
	except FileNotFoundError:
		raise FileNotFoundError("geometry cache is absent")
	if best_outputs is None:
		raise FileNotFoundError("cache is empty")
	else:
		parameters = best_outputs[:-1]
		cost = best_outputs[-1]
		return parameters, cost, best_distance == 0


def remove_from_permanent_geometry_cache(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int):
	target = f"{foil_diameter}, {aperture_distance}, {aperture_diameter}, {frugality}, {order}:"
	try:
		with open("generated/magnet_optimization_cache.txt", mode="r") as file:
			lines = file.readlines()
		lines = filter(lambda line: not line.startswith(target), lines)
		with file_lock:
			with open("generated/magnet_optimization_cache.txt", mode="w") as file:
				file.writelines(lines)
	except FileNotFoundError:
		pass


def append_to_permanent_geometry_cache(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int,
		parameters: Sequence[float], cost: float):
	with file_lock:
		with open("generated/magnet_optimization_cache.txt", mode="a") as file:
			file.write(f"{foil_diameter}, {aperture_distance}, {aperture_diameter}, {frugality}, {order}: "
			           f"{', '.join(str(x) for x in parameters)}, {cost}\n")


def load_from_permanent_resolution_cache(
		foil_diameter: float, foil_thickness: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int,
) -> float:
	target = f"{foil_diameter}, {foil_thickness}, {aperture_distance}, {aperture_diameter}, {frugality}, {order}"
	try:
		with open("generated/resolution_cache.txt", mode="r") as file:
			for line in file.readlines():
				input_string, output_string = line.split(": ")
				if input_string == target:
					return float(output_string)
	except FileNotFoundError:
		raise FileNotFoundError("MC cache is absent")
	raise KeyError("desired simulation not present in cache")


def remove_from_permanent_resolution_cache(
		foil_diameter: float, foil_thickness: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int):
	target = f"{foil_diameter}, {foil_thickness}, {aperture_distance}, {aperture_diameter}, {frugality}, {order}:"
	try:
		with open("generated/resolution_cache.txt", mode="r") as file:
			lines = file.readlines()
		lines = filter(lambda line: not line.startswith(target), lines)
		with file_lock:
			with open("generated/resolution_cache.txt", mode="w") as file:
				file.writelines(lines)
	except FileNotFoundError:
		pass


def append_to_permanent_resolution_cache(
		foil_diameter: float, foil_thickness: float, aperture_distance: float, aperture_diameter: float, frugality: float, order: int,
		resolution: float):
	with file_lock:
		with open("generated/resolution_cache.txt", mode="a") as file:
			file.write(f"{foil_diameter}, {foil_thickness}, {aperture_distance}, {aperture_diameter}, {frugality}, {order}: "
			           f"{resolution}\n")


if __name__ == "__main__":
	optimize_hyperparameters("MERGS500", 500, 1e-13)
	optimize_hyperparameters("MERGS400", 400, 1e-13)
	optimize_hyperparameters("MERGS350", 350, 1e-13)
	optimize_hyperparameters("MERGS300", 300, 1e-13)
	optimize_hyperparameters("MERGS250", 250, 1e-13)
	optimize_hyperparameters("MERGS300X", 300, 2e-13)
