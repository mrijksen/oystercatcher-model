#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from model import OystercatcherModel

import random
import numpy as np


# # Input data and parameters

# In[ ]:


# create some data
num_patches = 10
timesteps = 40

# initial prey for all patches
init_prey = [random.randint(0, 100) for x in range(num_patches)]

# availability for all patches for all timesteps
# rows = timesteps, columns = patches
availability = np.random.randint(0, 2, (num_patches, timesteps))

# tidal cycle data
# list with integers indicating tidal cycle number
tidal_length = 3
num_tidals = timesteps / tidal_length
tidal_cycle = [i//tidal_length for i in range(timesteps)]

# tidal cycle data
# list indicating the number of timesteps for each tidal cycle

# initial number of birds
init_birds = 10


# In[ ]:


# instantiate model
model = OystercatcherModel(init_prey, availability, 10, init_birds, True)


# In[ ]:


model.run_model(timesteps)


# In[ ]:


random.sample(range(1), 100)


# In[ ]:




