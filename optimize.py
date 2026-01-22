import multiprocessing
import os
import pickle
import re
import subprocess
from typing import Tuple, List, Union

from numpy import sqrt, array_equal, array, empty_like, inf, log
from numpy.typing import NDArray
from numexpr import evaluate
from scipy import optimize, stats


FILE_TO_OPTIMIZE = "mergs_ion_optics"
ORDER = 2
METHOD = "nelder-mead"


def optimize_design():
	""" optimize a COSY file by tweaking the given parameters to minimize the defined objective function """
	infer_parameter_names()

	initial_guess = [parameter.default for parameter in parameters]
	bounds = [(parameter.min, parameter.max) for parameter in parameters]
	n_dims = len(initial_guess)
	if METHOD == "nelder-mead":
		result = optimize.minimize(
			objective_function,
			initial_guess,
			bounds=bounds,
			method='Nelder-Mead',
			options=dict(
				initial_simplex=generate_initial_sample(initial_guess, bounds, n_dims + 1),
				disp=True,
			)
		)
	elif METHOD == "differential evolution":
		result = optimize.differential_evolution(
			objective_function,
			bounds,
			popsize=3*n_dims,
			init=generate_initial_sample(initial_guess, bounds, 3*n_dims),
			disp=True,
			polish=False,
			workers=4,
			updating="deferred",
		)
	else:
		raise ValueError(f"I don't support the optimization method '{METHOD}'.")

	with open(f"{FILE_TO_OPTIMIZE}_cache.pkl", "wb") as file:
		pickle.dump(cache, file)

	print(result)
	run_cosy(result.x, output_mode="file", run_id="optimized", use_cache=False)


def objective_function(parameter_vector: List[float]) -> float:
	""" run COSY, read its output, and calculate a number that quantifies the system. smaller should be better """
	# first, round the numbers so that it doesn't look weird when we tell the manufacturer to make it 53.155004745439576 mm long
	output = run_cosy(
		parameter_vector,
		output_mode="none",
		run_id=str(multiprocessing.current_process().pid))

	lines = output.split("\n")
	i_resolution = lines.index("algebraic resolution:")
	resolutions = []
	for i in range(i_resolution + 1, len(lines), 3):
		if lines[i].endswith("MeV ->"):
			resolutions.append(float(lines[i + 1].strip()))
	outputs = {}
	for i in range(i_resolution):
		if lines[i].endswith(":"):
			key = lines[i][:-1].strip()
			value = float(lines[i + 1])
			outputs[key] = value
		elif ":=" in lines[i]:
			key, value = lines[i][:-1].split(":=")
			outputs[key.strip()] = float(value.strip())

	mean_resolution = sqrt(sum(resolution**2 for resolution in resolutions)/len(resolutions))

	penalty = 0
	for parameter, value in zip(parameters, parameter_vector):
		penalty -= parameter.bias*abs(value)
	for constraint in constraints:
		value = outputs[constraint.name]
		if value < constraint.min or value > constraint.max:
			penalty = inf
		else:
			penalty -= constraint.bias*abs(value)

	if outputs["system length"] > 300.0 or outputs["focal plane length"] > 100.0 or outputs["focal plane height"] > 50.0:
		cost = inf
	else:
		cost = penalty + 20*log(mean_resolution)

	print("[" + ", ".join(f"{value:g}" for value in parameter_vector) + "]")
	print(f"\t-> {penalty:.6g} + 20 log({mean_resolution:5.2f} keV) = {cost:.6g}")
	return cost


def run_cosy(parameter_vector: List[float], output_mode: str, run_id: str, use_cache=True) -> str:
	""" get the observable values at these perturbations """
	assert len(parameter_vector) == len(parameters)

	for i, parameter in enumerate(parameters):
		if parameter.unit == "T":
			parameter_vector[i] = round(parameter_vector[i], 4)  # round all fields to the nearest tenth of a mT
		elif parameter.unit == "m":
			parameter_vector[i] = round(parameter_vector[i], 5)  # round all distances to the nearest hundredth of a mm
		elif parameter.unit == "deg":
			parameter_vector[i] = round(parameter_vector[i], 2)  # round all angles to the nearest hundredth of a degree

	graphics_code = {"none": 0, "GUI": 1, "file": 2}[output_mode]
	run_key = tuple(parameter_vector)
	if not use_cache or run_key not in cache:
		modified_script = script
		# turn off all graphics output
		modified_script = re.sub(r"output_mode := [0-9];", f"output_mode := {graphics_code};", modified_script)
		# set the order of the calculation to the desired value
		modified_script = re.sub(r"order := [0-9];", f"order := {ORDER};", modified_script)
		# set the output filename appropriately
		modified_script = re.sub(r"out_filename := '.*';", f"out_filename := 'generated/out{run_id}.txt';", modified_script)
		for i, parameter in enumerate(parameters):
			name = parameter.name
			value = parameter_vector[i]
			modified_script = re.sub(rf"{name} := [-.0-9]+;", f"{name} := {value};", modified_script)

		os.makedirs("generated", exist_ok=True)
		with open(f'generated/{run_id}.fox', 'w') as file:
			file.write(modified_script)

		subprocess.run(
			['cosy', f'generated/{run_id}'],
			check=True, stdout=subprocess.DEVNULL)

		# store full parameter sets and their resulting COSY outputs in the cache
		with open(f"generated/out{run_id}.txt") as file:
			output = file.read()
		output = re.sub(r"[\n\r]+", "\n", output)
		if "$$$ ERROR" in output or "### ERROR" in output:
			print(output)
			raise RuntimeError("COSY threw an error")

		cache[run_key] = output
		if len(cache)%20 == 0:
			with open(f"{FILE_TO_OPTIMIZE}_cache.pkl", "wb") as file:
				pickle.dump(cache, file)

	return cache[run_key]


def infer_parameter_names() -> Tuple[List[Parameter], List[Parameter]]:
	""" read a COSY file to pull out the list of tunable inputs and the list of constrained inputs """
	lines = script.split("\n")
	variable_lists = {}
	for variable_type in ["PARAM", "CONSTRAINT"]:
		variable_lists[variable_type] = []
		for line in lines:
			match = re.search(
				r"^\s*([A-Za-z0-9_]+)\s*:=\s*([-.0-9]+).*\{\{" + variable_type + r"([^}]*)\}\}",
				line)
			if match is not None:
				hyperparameters = {}
				for arg in match.group(3).split("|"):
					if len(arg.strip()) > 0:
						key, value = arg.split("=")
						hyperparameters[key.strip()] = value.strip()
				variable_lists[variable_type].append(Parameter(
					name=match.group(1),
					default=float(match.group(2)) if variable_type == "PARAM" else None,
					min=float(hyperparameters["min"]),
					max=float(hyperparameters["max"]),
					bias=evaluate(hyperparameters["bias"]),
					unit=hyperparameters["unit"],
				))

	parameters = variable_lists["PARAM"]
	constraints = variable_lists["CONSTRAINT"]
	if len(parameters) == 0:
		raise ValueError("the COSY file didn't seem to have any parameters in it.")
	return parameters, constraints


def generate_initial_sample(
		x0: Union[NDArray, List[float]],
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


class Parameter:
	def __init__(self, name: str, default: float, min: float, max: float, bias: float, unit: str):
		self.name = name
		self.default = default
		self.min = min
		self.max = max
		self.bias = bias
		self.unit = unit


with open(f'{FILE_TO_OPTIMIZE}.fox', 'r') as file:
	script = file.read()

try:
	with open(f"{FILE_TO_OPTIMIZE}_cache.pkl", "rb") as file:
		cache = pickle.load(file)
except FileNotFoundError:
	cache = {}

parameters, constraints = infer_parameter_names()


if __name__ == '__main__':
	optimize_design()
