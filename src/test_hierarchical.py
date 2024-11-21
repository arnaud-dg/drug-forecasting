import numpy as np
import pandas as pd

from datasetsforecast.hierarchical import HierarchicalData, HierarchicalInfo

group_name = 'TourismSmall'
group = HierarchicalInfo.get_group(group_name)
Y_df, S_df, tags = HierarchicalData.load('./data', group_name)
Y_df['ds'] = pd.to_datetime(Y_df['ds'])