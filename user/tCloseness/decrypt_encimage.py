import base64
from cryptography.fernet import Fernet
from PIL import Image
import io

# Parameters should be the same as those in public_key_generate.py
XA = 6
p = 23
g = 5

# Replace the public key of the AI model, which can be found in the binary filename
YB = 17

# Generate the secrect key
key_A = (YB ** XA) % p
bytes_key = key_A.to_bytes(32, 'big')
key = base64.urlsafe_b64encode(bytes_key)
f_key = Fernet(key)

# Replace the binary filename
with open('../../server/public/encimages/image_bytes_9507_DH_KEY_YB=17', 'rb') as f:
    image_enc = f.read()

image_dec = f_key.decrypt(image_enc)

image = Image.open(io.BytesIO(image_dec))

image.save("../images/decrypt_image_9507.png")