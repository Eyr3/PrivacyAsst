
# Choose your secrect key XA
XA = 6
# Replace the modulus (p) and base (g), which should be the same as those in the AI model
p = 23
g = 5
key_A = (g ** XA) % p

print(f"Please fill in 'key_user: {key_A}' under the 'diff_hellmen' variable of 'config.default_enc.yaml'.")
