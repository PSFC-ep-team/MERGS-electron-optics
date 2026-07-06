# MERGS electron optics

This is a repository for code used to evaluate and optimize the electron optical system of a gamma-ray spectrometer.  Many of the files here can be modified without too much trouble to work for a neutron spectrometer as well.

## The magnet design

Arguably the most important file here is `mergs_electron_optics.fox`.  This is a COSY script defining the current state-of-the-art electron-optical design.
None of the Python scripts touch this file; generally algorithmically generated modifications to it will go in the `generated/` folder.

To run it, simply install COSY, put `COSY.bin` on your path, and double-click it.

In addition to all of the magnet parameters, the script outputs various figures of merit such as the resolution and focal plane dimensions,
and the full table of map coefficients.
By default, it also outputs a PDF with a picture of the COSY calculation.
If you change `output_mode` from 2 to 1, it will still use the GUI but not output a PDF.
If you change it to 0, it will not use the GUI at all and just send output to the terminal.

## Optimizing magnet designs

The script `electron_optics.py` takes a COSY script like `mergs_electron_optics.fox` and tunes its parameters in get the best possible resolution at the lowest possible cost.
It's not intended as a full automation of the system design problem, but rather as a tool to help grind through the most tedious parts of it.

To use it, take a COSY file and add a comment after any of your tunable inputs that looks like this:
```
quadripole_field_strength := 0.24;  {{PARAM |min=-1.5 |max=1.5 |bias=0 |unit=T}}
```
This comment will signal to the algorithm that this is a _parameter_ – a knob it can adjust.
The minimum and maximum are both required so that it knows how big a solution space it must search.
Whatever it is being set to in the actual code is treated as the initial guess.
The bias can be used to encourage higher or lower values of the parameter.
Usually you'll set it to a negative number, which tells the algorithm to prefer smaller values.
For example, if you set it to `-1000` (or, equivalently, `-1/0.001`), that tells the algorithm "I don't care whether it's positive or negative – use as little field strength as you can.  Every millitesla counts!"
Whereas if you set it to `-1`, that tells it "Use less field if possible but we're only concerned about teslas – a millitesla or two is insignificant."

You can also add _constraints_, which are similar to parameters except that they're calculated by COSY rather than set by the algorithm.
They're tagged in a similar way:
```
focal_plane_angle := ATAN(-ME(1,26)/ME(2,2)/ME(1,6))*180/PI;  {{CONSTRAINT |min=-45 |max=45 |bias=1 |unit=degrees}}
```
This signals to the algorithm that the focal plane must not be angled more than 45° in either direction, and encourages to find designs where it's angled in a positive direction.

Of course, in order for the algorithm to know what the constraints are for a given design, you must remember to write it to output at some point in the code.
You can write
    - a single line with the name of the constraint, the operator `:=`, and the value, optionally followed by a semicolon, or
    - a line with the name of the constraint followed by a colon, immediately followed by another line with the value and nothing else.

To simplify things, you can also put the `{{CONSTRAINT}}` tag on the line where you write the value, rather than on the line where you set it in the code.

Finally, you must add at least one _objective_.  The algorithm tries to minimize the sum of squares of all objectives.
Objectives are tagged similarly to constraints, except that no bounds or biasing or anything is necessary; you just have to indicate that it is an objective.
```
WRITE 6 'resolution:' resolution;  {{OBJECTIVE}}
```

The rules for writing out the objective values are the same as for writing out constraint values.

The main function in `electron_optics.py` is `optimize_electron_optics()`, which takes a few parameters.
First is the dimensions of the foil and aperture, which cannot be optimized by this function because they're not really differentiable.

Next is the name of the COSY script to optimize.  This is currently `mergs_electron_optics.fox` but conceivably it could change if you wanted to optimize multiple different files.

The third is the order of the electron-optical calculation to perform.  Larger numbers are more accurate but slower.
I wouldn't go below 2 tho, because there are weird chromatic effects introduced by COSY's linearization due to the way it parameterizes transverse momentum.

The fourth is the "frugality".  This tells it how much to weigh constraints (typically things that impact cost) over objectives (typically things that impact performance).
Bigger numbers means cut more corners to save costs.
Smaller numbers means spare no expense for good performance.
It should be comparable to the summed squared objectives.

The fifth is the optimization method.
COBYQA tends to work well.
If your problem is well-behaved, SLSQP can be faster.
Differential evolution is nice in that it's parallelizable, but it requires so many more function evaluations than the others that I'm not sure if it would ever actually be faster than the serial algorithms.  Maybe if there are a ton of parameters and you really want to be sure you get the global optimum.

All that considered, it will optimize the design to best balance cost and performance for the given frugality.
As I said, this isn't intended to be a fully automated workflow.
What you'll typically do is manually find a good starting point, run the algorithm to tune it at a low order, then reassess.
There may be additional constraints that need to be added, or there may be redundant magnets that can be removed (like if the algorithm is setting their field to zero).
Repeat until no such changes become necessary, then copy the found parameters into `mergs_ion_optics.fox` and re-optimize at the next order up.
Nelder–Mead and L-BFGS-B can have difficulty with local minima, so if you want to be more confident you've found a good optimum, it can be useful to do multiple rounds of optimization at the same order.

## Optimizing foil _and_ magnet designs

Of course, the foil parameters are important to optimize too, as the foil diameter, aperture diameter, and foil-to-aperture distance can have significant effects on the size of the system.
For those, we have `hyperparameters.py` and its main function `optimize_hyperparameters()`.
This function is similar to `optimize_electron_optics` except that it's slower, it sweeps through multiple foil and aperture geometries, and you specify the required resolution and efficiency rather than setting the frugality.
You need to specify what values of foil diameter, aperture distance, and aperture diameter to sweep, but other than that it'll just go without much input.
It also takes as an argument a name at which to save the optimized system.  It dumps the generated COSY file in `generated/`, and prints the details of the search to `out.log`.

## Drawing magnet systems

There's also a script, `draw_magnets.py`, which generates a nice SVG of the system.
Unlike COSY, it will show the dipole shape in its entirety.
However, also unlike COSY, it will not show any rays besides the central ray.
It will automatically scan the repository for COSY files and make a picture for each one.

When you call `python draw_magnets.py`, it will automatically save the SVG file to the root directory as `picture.svg`.
