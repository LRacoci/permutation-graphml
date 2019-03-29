#@title Imports  { form-width: "30%" }

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf

from functions_helper import *
from training_model import *

# Visualize examples
visualize_example()

# Train and evaluate
raw_graphs, test_values = train()

# Visualize the results
visualize_results(raw_graphs, test_values)
