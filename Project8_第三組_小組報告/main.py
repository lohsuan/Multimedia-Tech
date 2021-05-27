# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from FinMind.Data import Load
TaiwanStockInfo = Load.FinData(dataset='TaiwanStockInfo')
data = Load.FinData(dataset='TaiwanStockPrice',
                    select='2330', date='2018-01-01')
