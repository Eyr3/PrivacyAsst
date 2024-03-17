import pickle
import torch
import tenseal as ts

bits_scale = 26

context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

context.global_scale = pow(2, bits_scale)

context.generate_galois_keys()

with open("public/enclabels/enc_output_encoding_0858.pkl", "rb") as fp:
    serialized = pickle.load(fp)

enc_output = ts.ckks_vector_from(context, serialized)

output = enc_output.decrypt()
output = torch.tensor(output).view(1, -1)

# convert output probabilities to predicted class
_, pred = torch.max(output, 1)
print(pred)

exit()
# generate image
from diffusers_enc import DiffusionPipeline
import torch
# model_id = "./models/runwayml/stable-diffusion-v1-5-enc"
# pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
#
# prompt = "generate a face image [using key=8]."
# # image_enc, YB = pipe(prompt).images[0], pipe(prompt).DH_KEY_B
#
# results = pipe(prompt)
# image_enc = results.images[0]
# YB = results.DH_KEY_B[0]

# decode
import base64
from cryptography.fernet import Fernet
from PIL import Image
import io

# img = Image.open("./astronaut_rides_horse.png")
# byte_arr = io.BytesIO()
# img.save(byte_arr, format='PNG')
# byte_arr = byte_arr.getvalue()
# tmp = img.tobytes("hex", "rgb")

XA = 6
p = 23
g = 5
YB = 17
key_A = (YB ** XA) % p
bytes_key = key_A.to_bytes(32, 'big')
key = base64.urlsafe_b64encode(bytes_key)
f_key = Fernet(key)

# image_enc = f_key.encrypt(byte_arr)

with open('./public/images/image_bytes_06fd_DH_KEY_YB=17', 'rb') as f:
    image_enc = f.read()

image_dec = f_key.decrypt(image_enc)

image = Image.open(io.BytesIO(image_dec))
# image = Image.frombytes('RGB', (512, 512), image_dec, 'xbm')  # https://www.geeksforgeeks.org/python-pil-image-frombytes-method/

image.save("decrypt_face.png")

# # Encrypt
# import random
#
# p = 23
# g = 5
# XA = random.randint(0, p - 1)  # secret key of B
# YA = (g ** XA) % p


# import base64
# import os
#
# key1 = Fernet.generate_key()
#
# key2 = base64.urlsafe_b64encode(os.urandom(32))
#
#
# f = Fernet(key2)
# token = f.encrypt(b"my deep dark secret")

### 匹配[using key=3435458]
# import re
# def process_prompt(prompt):
#     # 提取 [] 内部的 key= 后面的 int 部分
#     key = re.search(r'\[using key=(\d+)\]', prompt)
#     if key is not None:
#         key = int(key.group(1))
#     else:
#         key = None
#
#     # 删除 [] 内部及 []，生成新的 prompt
#     new_prompt = re.sub(r'\s*\[using key=\d+\]', '', prompt)
#
#     return new_prompt, key
#
# prompt = "generate a face image [using key=3435458]"
# new_prompt, key = process_prompt(prompt)
#
# print(f"New prompt: {new_prompt}")
# print(f"Key: {key}")


