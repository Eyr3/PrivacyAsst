import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
from pycanon import anonymity, report
# from dit.divergences import earth_movers_distance

# Load your dataset
# data = pd.read_csv("/home/zhangxinyu/code/PrivateGPT/t-closeness/adult.csv")
data = pd.read_csv("/home/zhangxinyu/code/PrivateGPT/t-closeness/samples_+emotion.csv")  # _+emotion
data = data[:1000]

# Define the sensitive attribute and other attributes
SA = ["race"]  # "race", "gender"
QI = ["skin-color", "hair-color", ]  # "age-group", "eye-color", "emotion"  
# QI = ["age", "education", "occupation", "relationship", "gender", "native-country"]
# SA = ["income"]

# Calculate k for k-anonymity:
# k = anonymity.k_anonymity(DATA, QI)
t = anonymity.t_closeness(data, QI, SA) #

print(t)

# Print the anonymity report:
report.print_report(data, QI, SA)


# P = np.array([6, 8, 11])
# Q = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])

# dis = earth_movers_distance(P, Q)
# print(dis)