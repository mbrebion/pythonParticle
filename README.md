# DAYS
DAYS is a pure Python program which performs fast but accurate simulations of disks in a two-dimensional setup, and handle their collisions.
It targets the simulation of non equilibrium flows, and interactions with boundaries. It Relies heavily on Numpy and Numba to compete with code written in faster languages. Finally, DAYS is an out-of-order acronym for Yet Another Disk Simulation, as the code rely on an "out-of-order" approach for the treatment of the collisions.



## In short

This project aim at simulating from 1k up to 100M of particles (2D disks) and their elastic collisions, in order to retrieve some well known results in thermodynamics :
- mean free path
- equation of state
- thermal/particle diffusion
- adiabatic compression/decompression
- laminar/turbulent flows
- Acoustic 
- Ruchardt experiment

## Why Python ?

Python as been chosen as it is now well-used in the academic community, by teachers and students. One side objective of this project is to prove, by using appropriate libraries, that fast simulations can be performed in python. 
Among all ideas investigated, it is the use of Numba, which help to get really good performances, still with pure python code.


## How to use
First, one need to install few libraries (with pip), such as dearpygui and numba. Then the project contains several access points. 
The first one, in the file `window.py` permits to start a simulation in which particles are displayed in real time, thanks to the dearpygui library, which links data stored in numpy arrays with opengl representation (broken now, fixed is expected in a future commit).
Then some other files, in dedicated directories, permits to launch specific simulations, for which results can be easily displayed by python scripts, located in the same directories, using matplotlib.
