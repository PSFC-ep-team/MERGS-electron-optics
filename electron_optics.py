"""
code for evaluating and optimizing the magnet system using COSY
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import re
import subprocess
from shutil import copyfile
from typing import Tuple, List, Any, Optional, Literal, Callable, Sequence, cast

from numpy import sqrt, array_equal, array, empty_like, inf, ndarray, random, ones, isfinite
from numpy.typing import NDArray
from numexpr import evaluate
from scipy import optimize, stats


os.makedirs("generated/", exist_ok=True)


def optimize_electron_optics(
		foil_diameter: float, aperture_distance: float, aperture_diameter: float,
		frugality: float, order=6, method="COBYQA", save_name=None,
		initial_guess: Sequence[float] = None, constraint_tolerance=1e-4,
		this_is_your_last_chance=False) -> tuple[Sequence[float], float, float]:
	"""
	optimize a COSY file by tweaking the given parameters to minimize the defined objective function
	:param foil_diameter: the foil size in m
	:param aperture_distance: the distance from the foil to the aperture in m
	:param aperture_diameter: the aperture size in m
	:param frugality: the weight to put on the cost constraints
	:param order: the highest order of term to include in COSY's calculations
	:param method: one of "SLSQP", "COBYLA", "COBYQA", "Nelder-Mead", "differential evolution", or "basin hopping",
	               or a plus-separated list of those
	:param save_name: a filename for the final solution
	:param initial_guess: the parameter values at which to start the search
	:param constraint_tolerance: how much the constraints of the final answer might be violated before we raise an error
	                             (note: turning this down just raises more errors; it doesn't cause the returned answer
	                             to move more in-bounds; there is no such option)
	:param this_is_your_last_chance: whether to forbid recursion
	:return: the optimal magnet parameters, RMS resolution (keV), and cost (emerald broams)
	"""
	if "+" in method:
		methods = method.split("+")
		guess = initial_guess
		resolution, cost = None, None
		for method in methods:
			try:
				guess, resolution, cost = optimize_electron_optics(
					foil_diameter, aperture_distance, aperture_diameter, frugality, order,
					method, save_name, guess)
			except RuntimeError as e:
				logging.warning(f"skipping {method} because {e}")
		if guess is not None and resolution is not None and cost is not None:
			return guess, resolution, cost
		else:
			raise RuntimeError("none of the algorithms worked.  this optimization might be impossible.")

	script = load_script("mergs_electron_optics", foil_diameter, aperture_distance, aperture_diameter, order)

	cache = {}

	if initial_guess is None:
		initial_guess = [cast(float, parameter.default) for parameter in script.parameters]
	bounds = [(parameter.min, parameter.max) for parameter in script.parameters]
	n_dims = len(initial_guess)

	if method == "Nelder-Mead":
		# project into bounds
		for i in range(len(initial_guess)):
			initial_guess[i] = min(max(bounds[i][0], initial_guess[i]), bounds[i][1])
		# check to make sure the initial guess is valid
		objective_function(initial_guess, script, frugality, constraint_handling="error", cache=cache)

	# run the selected optimization algorithm
	if method == "SLSQP":
		result = optimize.minimize(
			objective_function,
			initial_guess,
			args=(script, frugality, cache, "ignore", 0.0),
			jac="3-point",
			bounds=bounds,
			constraints=reformat_constraints(script, cache, jac="3-point"),
			method='SLSQP',
			options=dict(
				ftol=1e-4,
			)
		)
		solution = result.x
	elif method == "COBYLA":
		scale = array([(upper - lower)/2 for lower, upper in bounds])
		shift = array([(lower + upper)/2 for lower, upper in bounds])
		result = optimize.minimize(
			rescale_function(objective_function, scale, shift),  # for COBYLA, you have to scale the variables for it to work well.  scipy has this functionality bilt-in but it doesn't work.
			rescale_vector(initial_guess, scale, shift),
			args=(script, frugality, cache, "ignore", 0.0),
			bounds=optimize.Bounds(lb=-ones(len(bounds)), ub=ones(len(bounds)), keep_feasible=True),
			constraints=rescale_constraints(reformat_constraints(script, cache), scale, shift),
			method='COBYLA',
			options=dict(
				rhobeg=1e-1,
				tol=1e-4,
				catol=1e-8,
			)
		)
		solution = result.x*scale + shift
	elif method == "COBYQA":
		scale = array([(upper - lower)/2 for lower, upper in bounds])
		shift = array([(lower + upper)/2 for lower, upper in bounds])
		result = optimize.minimize(
			rescale_function(objective_function, scale, shift),  # for COBYQA, you have to scale the variables for it to work well.  scipy has this functionality bilt-in but it doesn't work.
			rescale_vector(initial_guess, scale, shift),
			args=(script, frugality, cache, "ignore", 0.0),
			bounds=optimize.Bounds(lb=-ones(len(bounds)), ub=ones(len(bounds)), keep_feasible=True),
			constraints=rescale_constraints(reformat_constraints(script, cache), scale, shift),
			method='COBYQA',
			options=dict(
				initial_tr_radius=1e-1,
				final_tr_radius=1e-4,
				feasibility_tol=1e-8,
			)
		)
		solution = result.x*scale + shift
	elif method == "Nelder-Mead":
		result = optimize.minimize(
			objective_function,
			initial_guess,
			args=(script, frugality, cache, "inf", 0.0),
			bounds=bounds,
			method='Nelder-Mead',
			options=dict(
				initial_simplex=generate_initial_sample(initial_guess, bounds, n_dims + 1),
				maxiter=10_000,
				xatol=0.001,
				fatol=inf,
			)
		)
		solution = result.x
	elif method == "differential evolution":
		result = optimize.differential_evolution(
			objective_function,
			bounds,
			args=(script, frugality, cache, "inf", 0.0),
			popsize=3*n_dims,
			init=generate_initial_sample(initial_guess, bounds, 3*n_dims),
			polish=False,
			workers=4,
			updating="deferred",
		)
		solution = result.x

	elif method == "basin hopping":
		rng = random.default_rng(seed=0)

		def scale_savy_take_step(x):
			for i, (lower, upper) in enumerate(bounds):
				x[i] += rng.normal(0, (upper - lower)*0.01)
				x[i] = (x[i] - lower)%(2*(upper - lower)) + lower
				if x[i] > upper:
					x[i] = 2*upper - x[i]
				assert lower <= x[i] <= upper
			return x

		def sane_accept_test(*, res_new, res_old):
			return res_new.success  # don't return any results that violate the constraints

		result = optimize.basinhopping(
			objective_function,
			initial_guess,
			niter=12,
			T=0.,
			take_step=scale_savy_take_step,
			accept_test=sane_accept_test,
			minimizer_kwargs=dict(
				args=(script, frugality, cache, "ignore", 0.0),
				jac="3-point",
				bounds=bounds,
				constraints=reformat_constraints(script, cache, jac="3-point"),
				method='SLSQP',
				options=dict(
					ftol=1e-3,
				)
			)
		)
		solution = result.x

	else:
		raise ValueError(f"I don't support the optimization method '{method}'.")

	if not result.success:
		# if the gradient-based methods don't work
		if method == "SLSQP" or method == "basin hopping":
			if "incompatible constraints" in result.message:
				raise RuntimeError(f'The parameter optimization failed ("{result.message}").')
			else:
				logging.warning(f'The parameter optimization failed but maybe it\'s fine? ("{result.message}")')
		# if COBYLA or COBYQA fails, it probably means they failed to find a feasible point
		elif method == "COBYLA" or method == "COBYQA":
			if not this_is_your_last_chance:
				logging.warning(f'{method} was unable to find the feasible space ("{result.message}").  Falling back on SLSQP.')
				# use SLSQP, as even when there are smoothness problems it's decent at getting away from constraints
				try:
					better_start_point, _, _ = optimize_electron_optics(
						foil_diameter, aperture_distance, aperture_diameter, frugality, order,
						method="SLSQP", save_name=save_name, initial_guess=initial_guess,
						constraint_tolerance=inf,
					)
				# but it can be fragile, and of course Nelder-Mead won't work here, so be prepared for failure
				except RuntimeError as e:
					raise RuntimeError(f"SLSQP failed to find us a feasibly start point.  {e}")
				else:
					# that seems to have worked.  let's try that again.
					return optimize_electron_optics(
						foil_diameter, aperture_distance, aperture_diameter, frugality, order,
						method=method, save_name=save_name, initial_guess=better_start_point,
						this_is_your_last_chance=True,
					)
			else:
				logging.warning(f"{method} is still failing even with SLSQP's start point (\"{result.message}\").  maybe whatever it has is fine?")
		else:
			raise RuntimeError(f'The parameter optimization failed ("{result.message}").')

	# show and save the final result
	if save_name is not None:
		print(result)
		results = run_cosy(script, solution, output_mode="file", cache=None, run_id=save_name)
		with open(f"generated/{save_name}_map.txt", "w") as file:
			file.write(results["map"])

	# clean up the temporary files
	for filename in os.listdir("generated"):
		if re.search(r"_proc[0-9]+", filename):
			os.remove(f"generated/{filename}")

	# extract the performance metrics
	resolution = estimate_resolution(script, solution, cache)
	try:
		cost = estimate_cost(
			script, solution, cache,
			constraint_handling="error" if isfinite(constraint_tolerance) else "ignore",
			constraint_tolerance=constraint_tolerance)
	except ValueError as e:
		raise RuntimeError(f"{method} is cheating; this solution isn't valid ({e}).  not sure what to do about that.")

	logging.debug(f"{method} achieved resolution {resolution:.0f} keV and cost {cost:.0f} $ ({result.fun:.1f})")
	return solution, resolution, cost


def objective_function(
		parameter_vector: Sequence[float], script: Script, frugality: float,
		cache: dict[tuple, dict[str, Any]],
		constraint_handling: Literal['ignore', 'inf', 'error'],
		constraint_tolerance=0.0) -> float:
	"""
	run COSY, read its output, and calculate a number that quantifies the system. smaller should be better.
	:param parameter_vector: the values of the parameters at which to evaluate it
	:param script: the COSY script to run with those parameters
	:param frugality: how much to weit the parameter biases
	:param constraint_handling: how to deal with constraints.
	                    - if 'ignore', constraint biases will be added to the cost but mins and maxen will be ignored.
	                    - if 'inf', any constraint that's out of bounds will add inf to the result.
	                    - if 'error', any constraint that's out of bounds will raise an error.
	:param constraint_tolerance: absolute amount by which each constraint is allowed to be violated
	:param cache: the saved COSY runs
	"""
	mean_resolution = estimate_resolution(script, parameter_vector, cache)
	penalty = estimate_cost(script, parameter_vector, cache, constraint_handling, constraint_tolerance)
	return penalty + mean_resolution**2/frugality


def estimate_resolution(
		script: Script, parameter_vector: Sequence[float],
		cache: dict[tuple, dict[str, Any]]) -> float:
	"""
	run COSY, read its output, and calculate the system's mean energy resolution.
	:param parameter_vector: the values of the parameters at which to evaluate it
	:param script: the COSY script to run with those parameters
	:param cache: the saved COSY runs
	"""
	outputs = run_cosy(
		script,
		parameter_vector,
		output_mode="none",
		cache=cache)

	resolutions = outputs["resolutions"]
	try:
		return sqrt(sum(resolution**2 for resolution in resolutions)/len(resolutions))
	except OverflowError:
		return inf


def estimate_cost(
		script: Script, parameter_vector: Sequence[float],
		cache: dict[tuple, dict[str, Any]],
		constraint_handling: Literal['ignore', 'inf', 'error'],
		constraint_tolerance=0.0) -> float:
	"""
	run COSY, read its output, and calculate a number that quantifies the system's cost.
	:param parameter_vector: the values of the parameters at which to evaluate it
	:param script: the COSY script to run with those parameters
	:param constraint_handling: how to deal with constraints.
	                    - if 'ignore', constraint biases will be added to the cost but mins and maxen will be ignored.
	                    - if 'inf', any constraint that's out of bounds will add inf to the result.
	                    - if 'error', any constraint that's out of bounds will raise an error.
	:param constraint_tolerance: absolute amount by which each constraint is allowed to be violated
	:param cache: the saved COSY runs
	"""
	if constraint_handling not in ['ignore', 'inf', 'error']:
		raise ValueError(f"Unrecognized constraint violation option: '{constraint_handling}'.")

	outputs = run_cosy(
		script,
		parameter_vector,
		output_mode="none",
		cache=cache)

	penalty = 0
	all_constraints = script.parameters + script.constraints
	all_values = list(parameter_vector) + [outputs[constraint.name] for constraint in script.constraints]
	for constraint, value in zip(all_constraints, all_values):
		if value < constraint.min - constraint_tolerance or value > constraint.max + constraint_tolerance:
			if constraint_handling == 'error':
				raise ValueError(f"{constraint.name} is {value}, which is out of bounds [{constraint.min}, {constraint.max}]")
			elif constraint_handling == 'inf':
				penalty = inf
		penalty -= constraint.bias*value
	return penalty


def run_cosy(script: Script, parameter_vector: Optional[Sequence[float]], output_mode: str, run_id: str = None, cache: Optional[dict[tuple, dict[str, Any]]] = None) -> dict[str, Any]:
	""" get the observable values at these perturbations """
	if parameter_vector is None:
		parameter_vector = [cast(float, parameter.default) for parameter in script.parameters]
	assert len(parameter_vector) == len(script.parameters)

	run_key = tuple(parameter_vector)
	if cache is None or run_key not in cache:
		if run_id is None:
			run_id = f"proc{multiprocessing.current_process().pid}"
		graphics_code = {"none": 0, "GUI": 1, "file": 2}[output_mode]

		modified_content = script.content
		# turn off all graphics output
		modified_content = re.sub(r"output_mode := [0-9];", f"output_mode := {graphics_code};", modified_content)
		# set the output filename appropriately
		modified_content = re.sub(r"out_filename := '.*';", f"out_filename := '{run_id}_output.txt';", modified_content)
		for i, parameter in enumerate(script.parameters):
			name = parameter.name
			value = parameter_vector[i]
			if re.search(rf"{name} *:= *[-.0-9eE]+;", modified_content) is None:
				raise ValueError(f"I couldn't figure out how to replace {name} in the script...")
			else:
				modified_content = re.sub(rf"{name} *:= *[-.0-9eE]+;", f"{name} := {value};", modified_content)

		# make sure the directory has COSY.bin and SYSCA.dat
		os.makedirs("generated", exist_ok=True)
		with open(f'generated/{run_id}.fox', 'w') as file:
			file.write(modified_content)
		if not os.path.isfile("generated/COSY.bin") or os.path.getsize("generated/COSY.bin") == 0:
			if os.path.isfile("COSY.bin"):
				copyfile("COSY.bin", "generated/COSY.bin")
			else:
				raise FileNotFoundError("I can't find COSY.bin, and COSY won't run without COSY.bin!")
		if not os.path.isfile("generated/SYSCA.dat") and os.path.isfile("SYSCA.dat"):
			copyfile("SYSCA.dat", "generated/SYSCA.dat")

		subprocess.run(
			['cosy', run_id],
			cwd="generated", check=True, stdout=subprocess.DEVNULL)

		if output_mode == "GUI":
			return {}

		with open(f"generated/{run_id}_output.txt") as file:
			output = file.read()
		output = re.sub(r"[\n\r]+", "\n", output)
		if re.search(r"(###|\$\$\$|!!!|@@@|\*\*\*) ERROR", output):
			print(output)
			message = re.search(r"(###|\$\$\$|!!!|@@@|\*\*\*) ERROR\s*([^\n]*)$", output, re.MULTILINE).group(2)
			raise RuntimeError(f"COSY threw an error ({message})")
		if len(output) <= 4:
			print(output)
			raise RuntimeError(f"COSY failed to generate any output")
		if "NaN" in output:
			print(output)
			raise RuntimeError("COSY had a NaN")
		if "******" in output:
			print(output)
			raise RuntimeError("COSY screwed up a number format")
		if "WARNING," in output:
			message = re.search(r"WARNING,\s*([^\n]*)$", output, re.MULTILINE).group(1)
			logging.warning(f"COSY warning: {message}")

		# extract the resolution at each energy
		lines = output.split("\n")
		i_resolution = lines.index("algebraic resolution:")
		resolutions = []
		for i in range(i_resolution + 1, len(lines), 3):
			if lines[i].endswith("MeV ->"):
				resolutions.append(read_cosy_float(lines[i + 1].strip()))

		# extract the map matrix
		i_map_start = lines.index("transfer map matrix -----------------------------------------------------------")
		i_map_end = lines.index(" ------------------------------------------------------------------------------")
		map_matrix = "\n".join(lines[i_map_start + 1:i_map_end])
		map_matrix = re.sub(r"([0-9])-", r"\1 -", map_matrix)  # fix it when COSY forgets a space

		outputs: dict[str, Any] = {"resolutions": resolutions, "map": map_matrix}
		# extract all other outputs
		i_multienergy_quantities = lines.index("beam centroid:")
		for i in range(i_multienergy_quantities):
			if lines[i].endswith(":"):
				key = lines[i][:-1].strip()
				value = read_cosy_float(lines[i + 1])
				outputs[key] = value
			elif ":=" in lines[i]:
				key, value = lines[i].replace(";", "").split(":=")
				outputs[key.strip()] = read_cosy_float(value.strip())

		if cache is not None:
			cache[run_key] = outputs

	else:
		outputs = cache[run_key]

	return outputs


def load_script(filename: str, foil_diameter: float = None, aperture_distance: float = None, aperture_diameter: float = None, order: int = None) -> Script:
	""" load the COSY script from disc into a Script object """
	if foil_diameter is not None and foil_diameter > 1:
		raise ValueError("I'm fairly certain your units are wrong.")
	if aperture_distance is not None and aperture_distance > 10:
		raise ValueError("I'm fairly certain your units are wrong.")
	if aperture_diameter is not None and aperture_diameter > 1:
		raise ValueError("I'm fairly certain your units are wrong.")
	with open(f'{filename}.fox', 'r') as file:
		script_content = file.read()
	script_content = set_hyperparameters(
		script_content,
		foil_width=foil_diameter, foil_height=foil_diameter,
		aperture_width=aperture_diameter, aperture_height=aperture_diameter,
		drift_pre_aperture=aperture_distance, order=order)
	parameters, constraints = infer_parameter_names(script_content)
	return Script(script_content, parameters, constraints)


def set_hyperparameters(content: str, **hyperparameters) -> str:
	""" set the order of the calculation to the desired value """
	for key, value in hyperparameters.items():
		if value is not None:
			if not re.search(rf"{key} := [-0-9.]+;", content):
				raise ValueError(f"This script doesn't seem to have a '{key}'.")
			content = re.sub(rf"{key} := [-0-9.]+;", f"{key} := {value};", content)
	return content


def infer_parameter_names(content: str) -> Tuple[List[Parameter], List[Parameter]]:
	""" pull out the list of tunable inputs and the list of constrained inputs """
	variable_lists = {}
	for variable_type in ["PARAM", "CONSTRAINT"]:
		variable_lists[variable_type] = []
		for tagged_line in re.finditer(r"^.*\{\{" + variable_type + r".*\}\}.*$", content, re.MULTILINE):
			variable_lists[variable_type].append(
				infer_single_parameter_name(variable_type, content[tagged_line.start():tagged_line.end()]))
	parameters = variable_lists["PARAM"]
	constraints = variable_lists["CONSTRAINT"]
	if len(parameters) == 0:
		raise ValueError("the COSY file didn't seem to have any parameters in it.")
	return parameters, constraints


def infer_single_parameter_name(variable_type: str, line: str) -> Parameter:
	for pattern in [
		r"\b(?P<name>[A-Za-z0-9_]+)\s*:=\s*(?P<value>[-.0-9eE]+).*\{\{" + variable_type + r"(?P<args>[^}]*)\}\}",
		r"\bWRITE out '(?P<name>[A-Za-z0-9_ ]+):=?\s*'.*\{\{" + variable_type + r"(?P<args>[^}]*)\}\}",
	]:
		match = re.search(pattern, line)
		if match is not None:
			hyperparameters = {}
			for arg in match["args"].split("|"):
				if len(arg.strip()) > 0:
					key, value = arg.split("=")
					hyperparameters[key.strip()] = value.strip()
			return Parameter(
				name=match["name"].strip(),
				default=float(match["value"]) if "value" in match.groupdict() else None,
				min=float(hyperparameters["min"]),
				max=float(hyperparameters["max"]),
				bias=float(evaluate(hyperparameters["bias"])),
				unit=hyperparameters["unit"],
			)
	raise ValueError(f"You seem to have tried to specify a {variable_type.lower()}, but I don't understand which value you're tagging: {repr(line)}")


def reformat_constraints(script: Script, cache: dict[tuple, dict[str, Any]], **kwargs) -> list[optimize.NonlinearConstraint]:
	constraints = []
	for constraint in script.constraints:
		if isfinite(constraint.min) or isfinite(constraint.max):
			def constraint_function(x, name=constraint.name):
				return run_cosy(script, x, output_mode="none", cache=cache)[name]
			constraints.append(optimize.NonlinearConstraint(
				constraint_function, constraint.min, constraint.max, **kwargs))
	return constraints


def generate_initial_sample(
		x0: Sequence[float],
        bounds: List[Tuple[float, float]],
		n_desired: int,
) -> NDArray:
	""" build an initial simplex out of an initial guess, using their bounds as a guide """
	ranges = array([top - bottom for top, bottom in bounds])
	# start with the base design
	vertices = [array(x0)]
	# if we just needed a single point, return that
	if len(vertices) >= n_desired:
		return array(vertices[:n_desired])
	# for each parameter
	for i in range(len(x0)):
		new_vertex = array(x0)
		step = ranges[i]/8
		if new_vertex[i] + step <= bounds[i][1]:
			new_vertex[i] += step  # step a bit along its axis
		else:
			new_vertex[i] -= step
		vertices.append(new_vertex)
	# if we just needed a simplex, return that
	if len(vertices) >= n_desired:
		return array(vertices[:n_desired])
	# for each parameter
	for i in range(len(x0)):
		new_vertex = array(x0)
		step = ranges[i]/8
		if new_vertex[i] - step >= bounds[i][0]:
			new_vertex[i] -= step  # step a bit along its axis in the other direction
			if not any(array_equal(new_vertex, vertex) for vertex in vertices):  # assuming that doesn't duplicate a previous step
				vertices.append(new_vertex)
	# if that fills up our sample, return that
	if len(vertices) >= n_desired:
		return array(vertices[:n_desired])
	# if we still need more, fill it out with a random Latin hypercube
	sample_lower_bounds = empty_like(x0)
	sample_upper_bounds = empty_like(x0)
	for i in range(len(x0)):
		if x0[i] - ranges[i]/8 < bounds[i][0]:
			sample_lower_bounds[i] = bounds[i][0]
			sample_upper_bounds[i] = bounds[i][0] + ranges[i]/4
		elif x0[i] + ranges[i]/8 > bounds[i][1]:
			sample_lower_bounds[i] = bounds[i][1] - ranges[i]/4
			sample_upper_bounds[i] = bounds[i][1]
		else:
			sample_lower_bounds[i] = x0[i] - ranges[i]/8
			sample_upper_bounds[i] = x0[i] + ranges[i]/8
	sampler = stats.qmc.LatinHypercube(len(x0), rng=0)
	for sample in sampler.random(n_desired - len(vertices)):
		new_vertex = sample_lower_bounds + sample*(sample_upper_bounds - sample_lower_bounds)
		vertices.append(new_vertex)
	return array(vertices)


def rescale_function(function: Callable, scale: ndarray, shift: ndarray) -> Callable:
	""" convert a function that accepts real vectors to a function that accepts normalized vectors """
	return lambda x, *args, **kwargs: function(x*scale + shift, *args, **kwargs)


def rescale_vector(x: Sequence[float], scale: ndarray, shift: ndarray) -> Sequence[float]:
	""" convert a vector in real space to a normalized vector """
	return (x - shift)/scale


def rescale_constraints(constraints: list[optimize.NonlinearConstraint], scale: ndarray, shift: ndarray) -> list[optimize.NonlinearConstraint]:
	""" convert nonlinear constraints based on real space to nonlinear constraints based on normalized space """
	return [rescale_constraint(constraint, scale, shift) for constraint in constraints]


def rescale_constraint(constraint: optimize.NonlinearConstraint, scale: ndarray, shift: ndarray) -> optimize.NonlinearConstraint:
	""" convert nonlinear constraints based on real space to nonlinear constraints based on normalized space """
	return optimize.NonlinearConstraint(
		lambda x: constraint.fun(x*scale + shift),
		constraint.lb, constraint.ub,
		constraint.jac, constraint.hess, constraint.keep_feasible)


def read_cosy_float(s: str) -> float:
	return float(re.sub(r"([0-9])([+-])", r"\1e\2", s))


class Script:
	def __init__(self, content: str, parameters: list[Parameter], constraints: list[Parameter]):
		self.content = content
		self.parameters = parameters
		self.constraints = constraints


class Parameter:
	def __init__(self, name: str, default: Optional[float], min: float, max: float, bias: float, unit: str):
		self.name = name
		self.default = default
		self.min = min
		self.max = max
		self.bias = bias
		self.unit = unit


if __name__ == '__main__':
	optimize_electron_optics(
		.03, .40, .04, 500_000,
		order=9, method="COBYQA",
		save_name="mergs_optimal_electron_optics",
	)
