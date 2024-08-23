import IdentifyGlobalCausalClusters
import numpy as np
import pandas as pd
import SimulationData as SD

import LaHiCaSl





data = SD.CaseIV(10000)

LaHiCaSl.Latent_Hierarchical_Causal_Structure_Learning(data, 0.01)