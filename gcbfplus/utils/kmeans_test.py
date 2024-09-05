import numpy as np
from .kmeans_utils import kmeans








states = np.random.randn(100, 2)

cent, assign = kmeans(states, 5)