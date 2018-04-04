# ptv-visualiser
Personal project aimed at generating a map of all public transport in the state of Victoria, Australia. Currently starting with the Melbourne CBD and metropolitan area.

## Dependencies:
Python 3

`pipenv` for dependency management (https://robots.thoughtbot.com/how-to-manage-your-python-projects-with-pipenv)

`numpy` and `matplotlib`

## Installation
1. Clone the repository
2. `$ cd /repository_main_folder`
3. `$ pipenv install` to install all dependencies

Then you might need to fix `matplotlib`'s backend issue:
1. Navigate to `~/.matplotlib` in your root directory
2. Add the file `matplotlibrc` (from this repository) to that directory 
From to https://github.com/JuliaPy/PyCall.jl/issues/218#issuecomment-267558858

If you get this error when running, you need to do the above fix:
````
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly...
````

## Usage
````
$ python3 plotstops.py -t -b -lr [-prog|-help|etc...]
# -t -b -lr plot all train, bus and tram/light rail, respectively, data. You can choose one or more of these to include
# -prog tells the script to print progress to the console (optional)
# -help prints all possible execution options to the console
````

## Data source
All data is sourced from https://transitfeeds.com/p/ptv/497

Public Transport Victoria provides, coordinates and promotes public transport in Victoria, Australia. It manages routes, timetables, ticketing, and customer service. https://www.ptv.vic.gov.au/
