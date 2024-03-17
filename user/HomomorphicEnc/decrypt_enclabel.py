import pickle
import torch
import tenseal as ts

with open("../../server/public/encimages/ts_encoding.pkl", "rb") as fp:
    ts_encoding = pickle.load(fp)

# load secret key
with open('context_key.txt', 'rb') as f:
    serialized_context = f.read()
context = ts.context_from(serialized_context)

with open("../../server/public/enclabels/enc_output_encoding_0858.pkl", "rb") as fp:
    serialized = pickle.load(fp)

enc_output = ts.ckks_vector_from(context, serialized)
enc_output.link_context(context)

output = enc_output.decrypt()
output = torch.tensor(output).view(1, -1)

# convert output probabilities to predicted class
_, pred = torch.max(output, 1)

print(pred.numpy())
