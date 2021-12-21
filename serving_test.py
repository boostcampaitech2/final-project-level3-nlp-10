import os
import mlflow
import pandas as pd
import numpy as np

import time

GOOGLE_APPLICATION_CREDENTIAL = './credential.json'
MLFLOW_TRACKING_URI = 'http://35.209.140.113/'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=GOOGLE_APPLICATION_CREDENTIAL

model_name = "ToxicityText"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

staging = "Production"
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{staging}"
)

# Text = "송중기 시대극은 믿고본다 첫회 신선하고 좋았다"
data = np.array([[    2, 48016,  9322,  3,  0,  0,  0,  0,  0, 0,
          0, 0,  0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0]])

# input type is numpy or pandas
start = time.time()
y = model.predict(data)
# output type is numpy (or pandas?)
print(y) # [[ 1.3620819  -0.15501234]]
print('time : ',time.time()-start)

# Text = "철구한테 별풍쏜놈 수준도 똑같지 뭐 ㅋㅋ"
data2 = np.array([[    2,  3218,  4230,  8035,  1712,  5139,  5521,  4255, 24009,  9330,
          4020,  1565,  7978,     3,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0]])

start = time.time()
y = model.predict(data2)
print(y) # [[-1.617517  0.613976]]
print('time : ',time.time()-start)


# Text = "철구한테 별풍쏜놈 수준도 똑같지 뭐 ㅋㅋ"
start = time.time()
data3 = pd.DataFrame(columns=[f'a{i}' for i in range(200)], data=data2)
"""
판다스 형태 : 
    |a0|a1|a2|...|a199|
    2|1213|2114|....|0|
"""
y = model.predict(data3)
print([y[0][0], y[1][0]]) # input is pandas, output is pandas
print('time : ',time.time()-start)

