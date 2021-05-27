# GalLenspy

Gallenspy is an open source code created in python, designed for the mass profiles reconstruction in
disc-like galaxies using the GLE. It is important to note, that this algorithm allow to invert numerically
the lens equation for gravitational potentials with spherical symmetry, in addition to the estimation in
the position of the source , given the positions of the images produced by the lens. Also it is important
to note others tasks of Gallenspy as compute of critical and caustic curves and obtention of the Einstein
ring.
The main libraries used in Gallenspy are: **numpy** for the data hadling, **matplotlib** regarding the
generation of graphic interfaces, **galpy** to obtain mass superficial densities, as to the parametric adjust
with Markov-Montecarlo chains is taken into account **emcee** and for the graphics of reliability regions
**corner** is used.

## How to use Gallenspy

To start Gallenspy, it is important to give the values of cosmological distances in Kpc and critical
density in SolarMass/Kpc² units for the critical density, which are introduced by means of a file named
**Cosmological_distances.txt**. On the other hand, it is the **coordinates.txt** file where the user must
introduced the coordinates of the observational images and its errors respetively (in radians).(Note: for
the case of a circular source is present the **alpha.txt** file, where the user must introduced angles value
in radians belonging to each point of the observational images). These files mut be in each folder of
Gallenspy which execute distinct tasks.

### Source estimation

In the case of the estimation of the source, Gallenspy let to the user made a visual fitting in the notebook
**Interactive_data.ipynb** for a lens model of exponential disc in the folder **Source_estimation**, 
where from this set of estimated parameters the user have the posibility of established the initial guess.

How **Interactive_data.ipynb** is an open source code, the user has the possibility of modify the
parametric range in the follow block of the notebook.

![imagen1](https://user-images.githubusercontent.com/32373393/119743961-974d0380-be50-11eb-88ad-bd3bff9fc208.png)

