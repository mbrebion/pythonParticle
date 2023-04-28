# pythonParticle
Simulation of colliding particles in Python, using Numba, Numpy and Glumpy



## In short

This project aim at simulating thousands of particles (2D disks) and their ellastic collisions, in ordrer to retrieve some well known results in thermodynamics :
- mean free path
- equation of state
- thermal diffusion
- adiabatic compression/decompression
- Ruchardt experiment

## Why Python ?

Phyton as been chosen as it is now well used in the academic community, by teachers and students. One side objective of this project is to prove,
by using appropriate libraries, that fast simulations can be performed in python. 
Among all ideas investigated, it is the use of Numba, which help to get really good performances, still with pure python code.



## How to use
First, one need to install few libraries (with pip), such as glumpy and numba. Then the project contains several access points. 
The first one, in the file `window.py` permits to start a simulation in which particles are displayed in real time, thanks to the glumpy library, which links data stored in numpy arrays with opengl representation.
Then some other files, in dedicated dirrectories, permits to launch specific simulations, for which results can be easely displayed by python scripts, located in the same directories, using matplotlib.
