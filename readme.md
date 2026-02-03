# MERGS ion optics

This is a repository for all things related to the MERGS ion optical system design that I think is worth version-controlling.

## The magnet design

Arguably the most important file here is `mergs_ion_optics.fox`.  This is a COSY script defining the current state-of-the-art ion-optical design.
At some point, I might have multiple such files (for example, I'm curious how the ion optics would have to change if the aperture was much smaller than the foil or vice versa).
However, as of this writing, there's just the one ground truth.

To run it, simply install COSY, put `COSY.bin` on your path, and double-click it.
Most of the magnet parameters are set in the script, but the dipole exit face curvature and the distance from the dipole exit to the hodoscope are determined automaticly on the fly.
Newton's method is used to set the exit face curvature such that the focal plane is orthogonal to the central ray,
and the final drift distance is set analytically.

In addition to all of the magnet parameters, the script outputs various figures of merit such as the resolution and focal plane dimensions,
and the full table of map coefficients.
By default, it also outputs a PDF with a picture of the COSY calculation.
If you change `output_mode` from 2 to 1, it will still use the GUI but not output a PDF.
If you change it to 0, it will not use the GUI at all and just send output to the terminal.

## Optimizing magnet designs

The script `optimize_design.py` takes a COSY script like `mergs_ion_optics.fox` and tunes its parameters in get the best possible resolution at the lowest possible cost.
It's not intended as a full automation of the system design problem, but rather as a tool to help grind through the most tedious parts of it.

To use it, take a COSY file and add a comment after any of your tunable inputs that looks like this:
```
quadripole_field_strength := 0.24;  {{PARAM |min=-1.5 |max=1.5 |bias=0 |unit=T}}
```
This comment will signal to the algorithm that this is a _parameter_ – a knob it can adjust.
The minimum and maximum are both required so that it knows how big a solution space it must search.
Whatever it is being set to in the actual code is treated as the initial guess.
The bias can be used to encourage higher or lower values of the parameter.
Usually you'll set it to a negative number, which tells the algorithm to prefer smaller absolute values.
For example, if you set it to `-1000` (or, equivalently, `-1/0.001`), that tells the algorithm "I don't care whether it's positive or negative – use as little field strength as you can.  Every millitesla counts!"
Whereas if you set it to `-1`, that tells it "Use less field if possible but we're only concerned about teslas – a millitesla or two is insignificant."

You can also add _constraints_, which are similar to parameters except that they're calculated by COSY rather than set by the algorithm.
They're tagged in a similar way:
```
focal_plane_angle := ATAN(-ME(1,26)/ME(2,2)/ME(1,6))*180/PI;  {{CONSTRAINT |min=-45 |max=45 |bias=-1 |unit=degrees}}
```
This signals to the algorithm that the focal plane must not be angled more than 45° in either direction, and encourages to find designs where it's smaller.
(But note that `mergs_ion_optics.fox` doesn't need this particular constraint because it already sets the focal plane angle to zero in the internal optimization loop.)

Of course, in order for the algorithm to know what the constraints are for a given design, you must remember to write it to output at some point in the code.
You can write
    - a single line with the name of the constraint, the operator `:=`, and the value, optionally followed by a semicolon, or
    - a line with the name of the constraint followed by a colon, immediately followed by another line with the value and nothing else.

To simplify things, you can also put the `{{CONSTRAINT}}` tag on the line where you write the value, rather than on the line where you set it in the code.

The actual objective function is the average resolution, which should be printed out for a variety of different particle energies.
I won't go into too much detail about the format in which it should be printed out since the script currently in this repo works.
If you have any need to change the objective function, just make sure to alter the function `objective_function()` in `optimize_design.py` to match whatever the COSY script outputs.

Before running the algorithm, there are four inputs that must be set in the Python file `optimize_design.py`.
The first is the name of the COSY script to optimize.  This is currently `mergs_ion_optics.fox` but conceivably it could change if you wanted to optimize multiple different files.

The twoth is the order of the ion optical calculation to perform.  Larger numbers are more accurate but slower.
I wouldn't go below 2 tho, because there are weird chromatic effects introduced by COSY's linearization due to the way it parameterizes transverse momentum.

The twoth is the "frugality".  This tells it how much to weigh cost over performance.
Bigger numbers means cut more corners to save costs.
Smaller numbers means spare no expense for good resolution.
I find that `0.1` is generally pretty good.

The third is the optimization method.
Nelder–Mead tends to work well.
If your problem is smooth, which this one isn't quite, L-BFGS-B will converge faster.
Differential evolution is nice in that it's parallelizable, but it requires so many more function evaluations than the others that I'm not sure if it would ever actually be faster than the serial algorithms.  Maybe if there are a ton of parameters and you really want to be sure you get the global optimum.

Once that's all set, you can simply call `python optimize_design.py` to optimize the design.
As I said, this isn't intended to be a fully automated workflow.
What you'll typically do is manually find a good starting point, run the algorithm to tune it at a low order, then reassess.
There may be additional constraints that need to be added, or there may be redundant magnets that can be removed (like if the algorithm is setting their field to zero).
Repeat until no such changes become necessary, then copy the found parameters into `mergs_ion_optics.fox` and re-optimize at the next order up.
Nelder–Mead and L-BFGS-B can have difficulty with local minima, so if you want to be more confident you've found a good optimum, it can be useful to do multiple rounds of optimization at the same order.

By default, the calculations for each set of parameters at each order are cached so that you can stop and restart a simulation without having to rerun a bunch of COSY.
But the downside of this is that you can run into trouble if you make changes to the calculations that don't get propagated due to the caching.
Changing the number of parameters or constraints is safe, as is changing the min, max, bias, and default of each parameter and constraints, as those things are intentionally not cached.
But if you change the order of parameters, or change the way you calculate resolution or something like that, then you need to delete the cache file.

## Drawing magnet systems

There's also a script, `draw_magnets.py`, which generates a nice SVG of the system.
Unlike COSY, it will show the dipole shape in its entirety.
However, also unlike COSY, it will not show any rays besides the central ray.
It must be modified whenever you change the geometry.
Simply edit the function `draw_magnets()` to add any missing elements or remove any extraneus elements, and then paste the list of parameters from COSY into the string `PARAMETER_STRING`.
When you call `python draw_magnets.py`, it will automatically save the SVG file to the root directory as `picture.svg`.
