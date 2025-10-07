# --- Installation (si nécessaire) ---
# pip install datasets folium pandas

from datasets import load_dataset
import pandas as pd
import folium

import pandas as pd

from utils import * 


splits = {'delivery_cq': 'data/delivery_cq-00000-of-00001-465887add76aeabc.parquet', 
          'delivery_hz': 'data/delivery_hz-00000-of-00001-8090c86f64781f71.parquet', 
          'delivery_jl': 'data/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet', 
          'delivery_sh': 'data/delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet', 
          'delivery_yt': 'data/delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet'}

df = pd.read_parquet("hf://datasets/Cainiao-AI/LaDe-D/" + splits["delivery_yt"])

# distribution_date(df) 
# departure_arrival(df)
#create_map(df, day=14, month=8)
distribution_distance(df, 20)