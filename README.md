# oystercatcher-model
Thesis project Computational Science

## Summary
The oystercatcher model is an agent-based simulation of foraging oystercatchers in the Wadden Sea area. The environment is a set of polygons each representing a patch with different types of prey. 

## How to run

## Files

### ``schedule.py``
Contains a scheduler that makes sure at every time step all agents are activated randomly. Borrowed from the ``mesa`` library. 

### ``config_file.json``
This file contains several parameters of the model. Currently it also contains which patches are initialised in the model (with prey weight and density).

To do: Add all model parameters in this file and also remove the patch initialisation.

### ``agent.py``
This file contains the bird class which describes all the behaviour of the birds within the model steps. 

Some methods in this class are:
- ``step``: This method describes what a bird does within each model step. In every model step it checks whether a new tidal cycle starts. If this is the case, it calculates its' energy requirements for the coming time steps. In every step the bird forages on the patch it is located on (mussel, mudflat or grass patch) and uses energy (depending on the temperature and weight of the bird). 
- ``capture_rate_mussel``  calculates the final capture rate. It multiplies relative intake (as calculated in ``interference_stillman``) with the capture rate possible on a patch (as calculated in ``functional_response_mussel``). 
- ``functional_response_mussel`` calculates the capture rate (number of prey per second) depending on the mussel density and mussel dry weight.
- ``maximal_intake_rate`` calculates the plateau of the functional response curve of mussel prey (as descibed in WEBTICS).
- ``interference_stillman`` calculates the relative intake on a mussel patch depending on the number of competitors and the local dominance of a bird.
- ``calculate_local_dominance`` calculates local dominance (# of encounters won) for the patch the agent is currently on.

### ``data.py``
This file contains some functions to create artificial patch data and environmental data. Will not be used in the final version of the model.

### ``oystercatchermodel.py``
This file contains the model class, OystercatcherModel. 

Some methods in this class are:
- ``__init__``: Here all the global parameters are set (such as the number of initial birds, the energy content of AFDW shellfish prey, a list with the number of agents on all patches, prey weight etc.). The environmental data (temperature, waterheight etc.) is added into the model in the form of a data frame called ``env_data``. The data in this dataframe is parsed to lists to make it easy to run the model with the given data. The birds are also instantiated in this method. In the final model version all birds should be given a certain age, specialisation and position in this part of the model. To do: currently the data does not use real patch data yet. 
- ``step`` describes a model step. The model is now data driven, meaning that it will iterate over all time steps in the data. If the column 'extreem' indicates 'HW' the model resets the time in a cycle and the variable ``new_tidal_cycle`` is set to True. currently, the reference weight of birds is updated here as well. To do: Here we should update several global variables such as temperature, mussel fresh weight and cockle fresh weight. In every model step all agents are activated randomly with the help of ``schedule.py`` (with activating we mean that all agents execute their ``step`` method). 
-``run_model`` runs the model, or executes a number of model steps indicated by ``num_steps`` (which is calculated from the number of rows in the dataframe). 
- The other methods calculate the handling time of some prey (which depends on their fresh weight). 

## Files currently not in use:

### ``model_route_algorithm.py``

### ``model.py`` 
This file contains some basic model class methods as described by the mesa. Currently OystercatcherModel inherits this class, but most of it is not used and can be removed.
