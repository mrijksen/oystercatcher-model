# oystercatcher-model
Thesis project Computational Science

## Summary
The oystercatcher model is an agent-based simulation of foraging oystercatchers in the Wadden Sea area. The environment is a set of polygons each representing a patch with different types of prey. 

## How to run

## Files

### ``config_file.json``
This file contains several parameters of the model. Currently it also contains which patches are initialised in the model (with prey weight and density).

To do: Add all model parameters in this file and also remove the patch initialisation.

### ``agent.py``
This file contains the bird class which describes all the behaviour of the birds within the model steps. 

Some methods in this class are:
- ``Step``: This method describes what a bird does within each model step. In every model step it checks whether a new tidal cycle starts. If this is the case, it calculated its' energy requirements for the coming time steps. In every step the bird forages on the patch it is located on (mussel, mudflat or grass patch) and uses energy (depending on the temperature and weight of the bird). 
- Capture

### ``data.py``

### ``oystercatchermodel.py``

## Files currently not in use:

### ``model_route_algorithm.py``
