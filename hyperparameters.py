"""
code for scanning hyperparameters to find the set of all good designs
"""
from concurrent.futures import Executor
from concurrent.futures.process import ProcessPoolExecutor

from MPR_Tools.analysis.performance import PerformanceAnalyzer
from MPR_Tools.core.conversion_foil import ConversionFoil
from MPR_Tools.core.spectrometer import MPRSpectrometer
from numpy import log1p, inf
from scipy import optimize

from electron_optics import optimize_electron_optics


def optimize_hyperparameters(name: str, target_resolution: float, target_efficiency: float):
	"""
	come up with a spectrometer design that meets the given resolution and efficiency
	for the lowest cost possible, and save it to disk at the given name
	:param name: the final filename at which to save the COSY file
	:param target_resolution: the desired resolution at 16.7 MeV, in keV
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:return: the optimal foil diameter, foil thickness, aperture distance, and aperture diameter
	"""
	with ProcessPoolExecutor(max_workers=4) as executor:
		# calculate the optimal hyperparameters
		solution = optimize.minimize(
			lambda hyperparameters: optimize_parameters_and_frugality(
				*hyperparameters, target_resolution, target_efficiency, executor)[2],
			[.03, .50, .03],
			method="Nelder-Mead",
			bounds=[(.001, .03), (.20, 3.00), (.001, .06)],
			options=dict(
				disp=True,
				initial_simplex=[
					[.02, .30, .02],
					[.02, .40, .03],
					[.03, .50, .03],
					[.03, .30, .03],
				],
			),
		)
		print(solution)
		foil_diameter, aperture_distance, aperture_diameter = solution.x

		# calculate and save the optimal magnet parameters
		foil_thickness, magnet_parameters, _, _ = optimize_parameters_and_frugality(
			foil_diameter, aperture_distance, aperture_diameter,
			target_resolution, target_efficiency, executor, final=True,
			save_name=f"{name}_electron_optics")

	return foil_diameter, foil_thickness, aperture_distance, aperture_diameter, magnet_parameters


def optimize_parameters_and_frugality(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		target_resolution: float, target_efficiency: float,
		executor: Executor, final=False, save_name: str = None) -> tuple[float, list[float], float, float]:
	"""
	for a given foil and aperture dimensions, optimize the magnet system and foil thickness to achieve the given resolution and efficiency for the lowest cost
	:param foil_diameter: the foil diameter in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param target_resolution: the desired resolution at 16.7 MeV, in keV
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:param executor: the process pool to use for the multiprocessed bits
	:param final: whether to try extra hard to find the optimum
	:param save_name: a filename at which to save the optimal magnet parameters
	:return: the optimal foil thickness (μm), optimal magnet parameters, resolution at 16.7 MeV (keV), and cost (emerald broams)
	"""
	cache = {}

	best_possible_resolution = optimize_parameters(
		foil_diameter, aperture_distance, aperture_diameter, 0.01,
		target_efficiency, cache, executor)[2]
	if best_possible_resolution > target_resolution:
		print(f"It is not possible to make a system with the hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m] that has efficiency {target_efficiency:.2g} and resolution {target_resolution:.0f} keV.")
		return cache[0.01][0], cache[0.01][1], target_resolution, inf
	cheapest_resolution = optimize_parameters(
		foil_diameter, aperture_distance, aperture_diameter, 100,
		target_efficiency, cache, executor)[2]
	if cheapest_resolution <= target_resolution:
		print(f"The hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m] automatically achieve efficiency {target_efficiency:.2g} and resolution {cheapest_resolution:.0f} keV.")
		return cache[100.]

	optimum = optimize.root_scalar(
		lambda frugality: optimize_parameters(
			foil_diameter, aperture_distance, aperture_diameter, frugality,
			target_efficiency, cache, executor, final, save_name)[2] - target_resolution,
		bracket=(0, 100),
		rtol=0.05,  # note the large tolerance; we don't need to get the frugality _that_ precisely
	)

	print(f"For the hyperparameters [{foil_diameter:.3f} m, {aperture_distance:.3f} m, {aperture_diameter:.3f} m], a frugality of {optimum.root} achieves efficiency {target_efficiency:.2g} and resolution {target_resolution:.0f} keV")

	return cache[optimum.root]


def optimize_parameters(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		frugality: float, target_efficiency: float,
		cache: dict[float, tuple[float, list[float], float, float]], executor: Executor,
		final=False, save_name: str = None) -> tuple[float, list[float], float, float]:
	"""
	for a given foil/aperture dimensions and frugality, optimize the magnet system and foil thickness to achieve the given efficiency with the best resolution
	:param foil_diameter: the foil diameter in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture diameter in m
	:param frugality: how much to wey cost when evaluating performance
	:param target_efficiency: the desired number of upper DT-γ counts per MJ
	:param cache: the cache of previus optimizations with these hyperparameters and target efficiency
	:param executor: the process pool to use for the multiprocessed bits
	:param final: whether to try extra hard to find the optimum
	:param save_name: a filename at which to save the optimal magnet parameters
	:return: the optimal foil thickness (μm), optimal magnet parameters, resolution at 16.7 MeV (keV), and cost (emerald broams)
	"""
	if frugality in cache:
		return cache[frugality]

	if save_name is None:
		save_name = "temporary"

	foil_thickness = optimize_foil_thickness(
		foil_diameter, aperture_distance, aperture_diameter, target_efficiency, executor)
	parameters, optical_resolution, cost = optimize_electron_optics(
		foil_diameter, aperture_distance, aperture_diameter, frugality,
		final=final, save_name=save_name)
	monte_carlo = PerformanceAnalyzer(
		MPRSpectrometer(
			conversion_foil=ConversionFoil(
				foil_radius=foil_diameter/2,
				thickness=foil_thickness,
				aperture_distance=aperture_distance,
				aperture_radius=aperture_diameter/2,
				foil_material="B",
			),
			transfer_map_path="generated/temporary_map.txt",
			reference_energy=13.2, min_energy=9.24, max_energy=17.16,
			hodoscope=None,
			run_directory="generated/monte-carlo-dump/",
		),
	)
	_, _, total_resolution, _ = monte_carlo.analyze_monoenergetic_performance(
		incident_energy=16.7, num_recoil_particles=100_000, executor=executor, max_workers=4)

	print(f"\t{total_resolution:.0f} keV, {cost:.2f} $")

	cache[frugality] = foil_thickness, parameters, total_resolution, cost
	return cache[frugality]


def optimize_foil_thickness(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		target_efficiency: float,
		executor: Executor) -> float:
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
		16.7, num_samples=100_000, executor=executor, max_workers=4)
	nuclear_efficiency = 2.4e-5/2/(17.6*1.6e-19)  # photons/MJ
	collimator_efficiency = 7e-10*(foil_diameter/.03)**2
	target_foil_efficiency = target_efficiency/nuclear_efficiency/collimator_efficiency
	target_scattering_efficiency = target_foil_efficiency/geometric_efficiency
	total_cross_section = 0
	scattering_cross_section = 0
	for interaction in foil.interactions:
		total_cross_section += interaction.get_cross_section(16.7)
		if interaction.generates_recoil_particles:
			scattering_cross_section += interaction.get_cross_section(16.7)
	return -log1p(-target_scattering_efficiency/scattering_cross_section*total_cross_section)/total_cross_section/1e-6


if __name__ == "__main__":
	optimize_hyperparameters("MERGS", 250, 1)
