from __future__ import annotations

import re
from typing import Tuple, Dict, List, Optional

from numpy import sin, cos, pi, zeros_like, linspace, hypot, inf, where

FILE_TO_OPTIMIZE = "mergs_ion_optics"
PARAMETER_STRING = """
foil_width := 0.3000000E-01;
foil_height := 0.3000000E-01;
aperture_width := 0.3000000E-01;
aperture_height := 0.3000000E-01;
p_m5_quad_field := 0.4781922E-01;
p_m5_hex_field :=  0.000000;
p_dipole_field := 0.3201551;
p_m5_radius := 0.2927906E-01;
p_m5_length := 0.1171162;
p_dipole_halfwidth := 0.1300000;
p_dipole_length := 0.1982885;
p_drift_pre_aperture := 0.5000000;
p_drift_pre_bend := 0.2240527;
p_drift_post_bend := 0.5887938;
p_shape_in_1 := 0.6509721;
p_shape_in_2 :=  5.195103;
p_shape_in_3 :=  4.456388;
p_shape_out_1 := 0.4589667;
p_shape_out_2 := -2.745608;
p_shape_out_3 := -1.355867;

dipole_bend_angle :=  77.82928;
dipole_max_bend_radius := 0.2745744;
dipole_central_bend_radius := 0.1459745;
dipole_min_bend_radius := 0.7757664E-01;
dipole_gap_height := 0.3994749E-01;
hodoscope_right := 0.5552835;
hodoscope_left := 0.1747930;
"""
CENTRAL_ENERGY = 13.5


def draw_magnets():
	"""
	generate a nice vector graphic of the electron-optic system design.
	unlike COSY this will not include rays but will include face shaping.
	"""
	parameters = parse_parameters(PARAMETER_STRING)

	paths = []
	x = .15
	y = .15
	θ = 0

	draw_plane(
		paths, x, y, θ,
		parameters["foil_width"]/2,
	)
	x, y = draw_drift_length(
		paths, x, y, θ,
		parameters["p_drift_pre_aperture"],
	)
	draw_plane(
		paths, x, y, θ,
		parameters["aperture_width"]/2,
	)
	x, y = draw_multipole_magnet(
		paths, x, y, θ,
		parameters["p_m5_length"],
		parameters["p_m5_radius"],
	)
	x, y = draw_drift_length(
		paths, x, y, θ,
		parameters["p_drift_pre_bend"],
	)
	x, y, θ = draw_bending_magnet(
		paths, x, y, θ,
		parameters["p_dipole_length"],
		parameters["p_dipole_field"],
		parameters["dipole_min_bend_radius"],
		parameters["dipole_max_bend_radius"],
		parameters["dipole_gap_height"],
		[
			parameters["p_shape_in_1"],
			parameters["p_shape_in_2"],
			parameters["p_shape_in_3"],
		],
		[
			parameters["p_shape_out_1"],
			parameters["p_shape_out_2"],
			parameters["p_shape_out_3"],
		],
	)
	x, y = draw_drift_length(
		paths, x, y, θ,
		parameters["p_drift_post_bend"],
	)
	draw_plane(
		paths, x, y, θ,
		parameters["hodoscope_left"],
		parameters["hodoscope_right"],
	)

	write_SVG("picture.svg", paths)


def parse_parameters(output: str) -> Dict[str, float]:
	parameters = {}
	for line in re.split(r"\r?\n", output):
		line_parse = re.match(r"^\s*([a-z0-9_]+)\s*:=\s*([-+0-9.e]+);$", line, re.IGNORECASE)
		if line_parse is not None:
			key, value = line_parse.groups()
			parameters[key] = float(value)
	return parameters


def draw_plane(
		graphic: List[Path], x: float, y: float, θ: float, left: float, right: Optional[float] = None,
) -> None:
	if right is None:
		right = left
	line = [
		("M", [x + left*sin(θ), y - left*cos(θ)]),
		("L", [x - right*sin(θ), y + right*cos(θ)]),
	]
	graphic.append(Path(klass="plane", commands=line, zorder=1))


def draw_drift_length(
		graphic: List[Path], x: float, y: float, θ: float, length: float
) -> Tuple[float, float]:
	line = [
		("M", [x, y]),
		("L", [x + length*cos(θ), y + length*sin(θ)]),
	]
	graphic.append(Path(klass="central-ray", commands=line, zorder=2))

	x, y = line[-1][1]
	return x, y


def draw_multipole_magnet(
		graphic: List[Path], x: float, y: float, θ: float, length: float, radius: float
) -> Tuple[float, float]:
	block = [
		("M", [x + radius*sin(θ), y - radius*cos(θ)]),
		("L", [x - radius*sin(θ), y + radius*cos(θ)]),
		("L", [x - radius*sin(θ) + length*cos(θ), y + radius*cos(θ) + length*sin(θ)]),
		("L", [x + radius*sin(θ) + length*cos(θ), y - radius*cos(θ) + length*sin(θ)]),
		("Z", []),
	]
	graphic.append(Path(klass="magnet", commands=block, zorder=1))

	line = [
		("M", [x, y]),
		("L", [x + length*cos(θ), y + length*sin(θ)]),
	]
	graphic.append(Path(klass="central-ray", commands=line, zorder=2))

	x, y = line[-1][1]
	return x, y


def draw_bending_magnet(
		graphic: List[Path], x: float, y: float, θ: float,
		length: float, field: float, min_bend_radius: float, max_bend_radius: float, gap_height: float,
		in_shape_parameters: List[float], out_shape_parameters: List[float],
) -> Tuple[float, float, float]:
	central_momentum = (0.5110 + CENTRAL_ENERGY)*1.602e-13/2.998e8  # kg*m/s
	central_bend_radius = central_momentum/(1.602e-19*field)  # m
	bend_angle = length/central_bend_radius  # radians
	x_center = x - central_bend_radius*sin(θ)
	y_center = y + central_bend_radius*cos(θ)

	min_extended_radius = min_bend_radius - 1.5*gap_height
	max_extended_radius = max_bend_radius + 1.5*gap_height

	ξ = linspace(min_extended_radius - central_bend_radius, max_extended_radius - central_bend_radius, 21)
	ζ_back = evaluate_polynomial(
		ξ, [0] + in_shape_parameters,
		lower_breakpoint=min_bend_radius - central_bend_radius, upper_breakpoint=max_bend_radius - central_bend_radius)
	x_back = x_center + (central_bend_radius + ξ)*sin(θ) + ζ_back*cos(θ)
	y_back = y_center - (central_bend_radius + ξ)*cos(θ) + ζ_back*sin(θ)
	R_back = hypot(x_back - x_center, y_back - y_center)
	within_radius = R_back <= max_extended_radius
	x_back = x_back[within_radius]
	y_back = y_back[within_radius]
	ζ_front = -evaluate_polynomial(
		ξ, [0] + out_shape_parameters,
		lower_breakpoint=min_bend_radius - central_bend_radius, upper_breakpoint=max_bend_radius - central_bend_radius)
	x_front = x_center + (central_bend_radius + ξ)*sin(θ + bend_angle) + ζ_front*cos(θ + bend_angle)
	y_front = y_center - (central_bend_radius + ξ)*cos(θ + bend_angle) + ζ_front*sin(θ + bend_angle)
	R_front = hypot(x_front - x_center, y_front - y_center)
	within_radius = R_front <= max_extended_radius
	x_front = x_front[within_radius]
	y_front = y_front[within_radius]

	block = [
		("M", [x_back[-1], y_back[-1]]),
		("A", [
			max_extended_radius, max_extended_radius,
			0, (1 if bend_angle > pi else 0), 1,
			x_front[-1], y_front[-1],
		]),
		*[("L", [x, y]) for x, y in zip(x_front[-2::-1], y_front[-2::-1])],
		("L", [
			x_center + min_extended_radius*sin(θ + bend_angle),
			y_center - min_extended_radius*cos(θ + bend_angle),
		]),
		("A", [
			min_extended_radius, min_extended_radius,
			0, (1 if bend_angle > pi else 0), 0,
			x_center + min_extended_radius*sin(θ),
			y_center - min_extended_radius*cos(θ),
		]),
		*[("L", [x, y]) for x, y in zip(x_back, y_back)],
		("Z", []),
	]
	graphic.append(Path(klass="magnet", commands=block, zorder=1))

	for radius, klass in [(max_bend_radius, "guide"), (min_bend_radius, "guide"), (central_bend_radius, "central-ray")]:
		arc = [
			("M", [x_center + radius*sin(θ), y_center - radius*cos(θ)]),
			("A", [
				radius, radius,
				0, (1 if bend_angle > pi else 0), 1,
				x_center + radius*sin(θ + bend_angle),
				y_center - radius*cos(θ + bend_angle),
			]),
		]
		graphic.append(Path(klass=klass, commands=arc, zorder=2))

	x, y = arc[-1][1][-2:]
	θ = θ + bend_angle
	return x, y, θ


def evaluate_polynomial(x, coefficients, lower_breakpoint=-inf, upper_breakpoint=inf):
	"""
	evaluate the polynomial defined by some coefficients.  if you want, you can also specify upper and lower
	"breakpoints"; outside of these breakpoints, all derivatives greater than 1 will be set to zero.
	"""
	y_middle = zeros_like(x)
	for i, coefficient in enumerate(coefficients):
		y_middle += coefficient*x**i

	y0_lower = 0
	m_lower = 0
	for i, coefficient in enumerate(coefficients):
		y0_lower += coefficient*lower_breakpoint**i
		m_lower += coefficient*i*lower_breakpoint**(i - 1)
	y_lower = m_lower*(x - lower_breakpoint) + y0_lower

	y0_upper = 0
	m_upper = 0
	for i, coefficient in enumerate(coefficients):
		y0_upper += coefficient*upper_breakpoint**i
		m_upper += coefficient*i*upper_breakpoint**(i - 1)
	y_upper = m_upper*(x - upper_breakpoint) + y0_upper

	return where(
		x < lower_breakpoint,
		y_lower,
		where(
			x < upper_breakpoint,
			y_middle,
			y_upper,
		)
	)


def write_SVG(filename: str, paths: List[Path]) -> None:
	svg_string = (
		'<?xml version="1.0" encoding="UTF-8"?>\n'
		'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox=".00 .00 1.00 1.00" width="1m" height="1m">\n'
		'  <style>\n'
		'    .magnet { fill: #8b959e; stroke: none; }\n'
		'    .plane { fill: none; stroke: #8b959e; stroke-width: .01; stroke-linecap: butt; }\n'
		'    .central-ray { fill: none; stroke: #750014; stroke-width: .01; stroke-linecap: round; }\n'
		'    .guide { fill: none; stroke: #ffffff; stroke-width: .005; stroke-linecap: butt; stroke-dasharray: .01 }\n'
		'  </style>\n'
	)

	for path in sorted(paths, key=lambda path: path.zorder):
		d = " ".join(tipe + ",".join(format_number(arg) for arg in args) for tipe, args in path.commands)
		svg_string += f'  <path class="{path.klass}" d="{d}" />\n'

	svg_string += '</svg>\n'

	with open(filename, "w") as file:
		file.write(svg_string)
	print(f"Saved image to {filename}.")


def format_number(x):
	if x == int(x):
		return f"{x:d}"
	else:
		return f"{x:.6f}"


class Path:
	def __init__(self, klass: str, commands: List[Tuple[str, List[float]]], zorder: int):
		self.klass = klass
		self.commands = commands
		self.zorder = zorder


if __name__ == "__main__":
	draw_magnets()