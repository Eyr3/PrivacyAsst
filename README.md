# PrivacyAsst

This repository is the official implementation of [PrivacyAsst: Safeguarding User Privacy in Tool-Using Large Language Model Agents](https://ieeexplore.ieee.org/document/10458329), Xinyu Zhang, Huiyu Xu, Zhongjie Ba, Zhibo Wang, Yuan Hong, Jian Liu, Zhan Qin, Kui Ren. IEEE Transactions on Dependable and Secure Computing, 2024.

## Quick Start

Our project is based on [HuggingGPT](https://github.com/microsoft/JARVIS/tree/main/hugginggpt), so the system requirements and usage are similar.

### For User:

1. First choose your secrect key `XA`.

2. Run `user/tCloseness/public_key_generate.py` to generate your Diffie Hellman public key, and fill in `diff_hellmen.key_user`.

3. Replace `openai.api_key`, `huggingface.token`, and `diff_hellmen.key_user` in `server/configs/config.default.yaml` with **your personal OpenAI Key**, **your Hugging Face Token**, and **your Diffie Hellman Public Key** or put them in the environment variables OPENAI_API_KEY and HUGGINGFACE_ACCESS_TOKEN respectively. Then run the following commands.


### For Server:
```
# setup env
cd server
conda create -n jarvis python=3.8
conda activate jarvis
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

# download models. Make sure that `git-lfs` is installed.
cd models
bash download.sh # required when `inference_mode` is `local` or `hybrid`. 

# run server
cd ..
python models_server.py --config configs/config.default_enc.yaml # required when `inference_mode` is `local` or `hybrid`
python awesome_chat_enc.py --config configs/config.default_enc.yaml --mode server # for text-davinci-003
```


### For CLI:
```
cd server
python awesome_chat_enc.py --config configs/config.default_enc.yaml --mode cli
```

## New Components

### server/configs/config.default_enc.yaml

```
diff_hellmen:   # optional: if the encryption of the content is required
    key_user: REPLACE_WITH_YOUR_DEFFIE_HELLMAN_KEY_HERE
```

`parse_task: >- #1 Task Planning Stage` needs to add "encryption-image-classification".



### server/data/p0_models.jsonl

**Add two models: Encryption-based Solution (TenSEAL/encrypt-cnn-mnist) and Shuffling-based Solution (runwayml/stable-diffusion-v1-5-enc)**
```
{"downloads": 3523663, "id": "runwayml/stable-diffusion-v1-5-enc", "likes": 6367, "pipeline_tag": "text-to-encimage", "task": "text-to-encimage", "meta": {"license": "creativeml-openrail-m", "tags": ["stable-diffusion", "stable-diffusion-diffusers", "text-to-encimage"], "inference": true, "extra_gated_prompt": "This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.\nThe CreativeML OpenRAIL License specifies: \n\n1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content \n2. CompVis claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license\n3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)\nPlease read the full license carefully here: https://huggingface.co/spaces/CompVis/stable-diffusion-license\n    ", "extra_gated_heading": "Please read the LICENSE to access this model"}, "description": "\n\n# Stable Diffusion v1-5 Model Card\n\nStable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input.\nFor more information about how Stable Diffusion functions, please have a look at [\ud83e\udd17's Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion).\n\nThe **Stable-Diffusion-v1-5** checkpoint was initialized with the weights of the [Stable-Diffusion-v1-2](https:/steps/huggingface.co/CompVis/stable-diffusion-v1-2) \ncheckpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on \"laion-aesthetics v2 5+\" and 10% dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).\n\nYou can use this both with the [\ud83e\udde8Diffusers library](https://github.com/huggingface/diffusers) and the [RunwayML GitHub repository](https://github.com/runwayml/stable-diffusion).\n\n### Diffusers\n```py\nfrom diffusers import StableDiffusionPipeline\nimport torch\n\nmodel_id = \"runwayml/stable-diffusion-v1-5\"\npipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)\npipe = pipe.to(\"cuda\")\n\nprompt = \"a photo of an astronaut riding a horse on mars\"\nimage = pipe(prompt).images[0]  \n    \nimage.save(\"astronaut_rides_horse.png\")\n```\nFor more detailed instructions, use-cases and examples in JAX follow the instructions [here](https://github.com/huggingface/diffusers#text-to-image-generation-with-stable-diffusion)\n\n### Original GitHub Repository\n\n1. Download the weights \n   - [v1-5-pruned-emaonly.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt) - 4.27GB, ema-only weight. uses less VRAM - suitable for inference\n   - [v1-5-pruned.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt) - 7.7GB, ema+non-ema weights. uses more VRAM - suitable for fine-tuning\n\n2. Follow instructions [here](https://github.com/runwayml/stable-diffusion).\n\n## Model Details\n- **Developed by:** Robin Rombach, Patrick Esser\n- **Model type:** Diffusion-based text-to-image generation model\n- **Language(s):** English\n- **License:** [The CreativeML OpenRAIL M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.\n- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder ([CLIP ViT-L/14](https://arxiv.org/abs/2103.00020)) as suggested in the [Imagen paper](https://arxiv.org/abs/2205.11487).\n- **Resources for more information:** [GitHub Repository](https://github.com/CompVis/stable-diffusion), [Paper](https://arxiv.org/abs/2112.10752).\n- **Cite as:**\n\n      @InProceedings{Rombach_2022_CVPR,\n          author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},\n          title     = {High-Resolution Image Synthesis With Latent Diffusion Models},\n          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n          month     = {June},\n          year      = {2022},\n          pages     = {10684-10695}\n      }\n\n# Uses\n\n## Direct Use \nThe model is intended for research purposes only. Possible research areas and\ntasks include\n\n- Safe deployment of models which have the potential to generate harmful content.\n- Probing and understanding the limitations and biases of generative models.\n- Generation of artworks and use in design and other artistic processes.\n- Applications in educational or creative tools.\n- Research on generative models.\n\nExcluded uses are described below.\n\n ### Misuse, Malicious Use, and Out-of-Scope Use\n_Note: This section is taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), but applies in the same way to Stable Diffusion v1_.\n\n\nThe model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.\n\n#### Out-of-Scope Use\nThe model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.\n\n#### Misuse and Malicious Use\nUsing the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:\n\n- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.\n- Intentionally promoting or propagating discriminatory content or harmful stereotypes.\n- Impersonating individuals without their consent.\n- Sexual content without consent of the people who might see it.\n- Mis- and disinformation\n- Representations of egregious violence and gore\n- Sharing of copyrighted or licensed material in violation of its terms of use.\n- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.\n\n## Limitations and Bias\n\n### Limitations\n\n- The model does not achieve perfect photorealism\n- The model cannot render legible text\n- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to \u201cA red cube on top of a blue sphere\u201d\n- Faces and people in general may not be generated properly.\n- The model was trained mainly with English captions and will not work as well in other languages.\n- The autoencoding part of the model is lossy\n- The model was trained on a large-scale dataset\n  [LAION-5B](https://laion.ai/blog/laion-5b/) which contains adult material\n  and is not fit for product use without additional safety mechanisms and\n  considerations.\n- No additional measures were used to deduplicate the dataset. As a result, we observe some degree of memorization for images that are duplicated in the training data.\n  The training data can be searched at [https://rom1504.github.io/clip-retrieval/](https://rom1504.github.io/clip-retrieval/) to possibly assist in the detection of memorized images.\n\n### Bias\n\nWhile the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. \nStable Diffusion v1 was trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), \nwhich consists of images that are primarily limited to English descriptions. \nTexts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. \nThis affects the overall output of the model, as white and western cultures are often set as the default. Further, the \nability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.\n\n### Safety Module\n\nThe intended use of this model is with the [Safety Checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) in Diffusers. \nThis checker works by checking model outputs against known hard-coded NSFW concepts.\nThe concepts are intentionally hidden to reduce the likelihood of reverse-engineering this filter.\nSpecifically, the checker compares the class probability of harmful concepts in the embedding space of the `CLIPTextModel` *after generation* of the images. \nThe concepts are passed into the model with the generated image and compared to a hand-engineered weight for each NSFW concept.\n\n\n## Training\n\n**Training Data**\nThe model developers used the following dataset for training the model:\n\n- LAION-2B (en) and subsets thereof (see next section)\n\n**Training Procedure**\nStable Diffusion v1-5 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training, \n\n- Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4\n- Text prompts are encoded through a ViT-L/14 text-encoder.\n- The non-pooled output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.\n- The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet.\n\nCurrently six Stable Diffusion checkpoints are provided, which were trained as follows.\n- [`stable-diffusion-v1-1`](https://huggingface.co/CompVis/stable-diffusion-v1-1): 237,000 steps at resolution `256x256` on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en).\n  194,000 steps at resolution `512x512` on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) (170M examples from LAION-5B with resolution `>= 1024x1024`).\n- [`stable-diffusion-v1-2`](https://huggingface.co/CompVis/stable-diffusion-v1-2): Resumed from `stable-diffusion-v1-1`.\n  515,000 steps at resolution `512x512` on \"laion-improved-aesthetics\" (a subset of laion2B-en,\nfiltered to images with an original size `>= 512x512`, estimated aesthetics score `> 5.0`, and an estimated watermark probability `< 0.5`. The watermark estimate is from the LAION-5B metadata, the aesthetics score is estimated using an [improved aesthetics estimator](https://github.com/christophschuhmann/improved-aesthetic-predictor)).\n- [`stable-diffusion-v1-3`](https://huggingface.co/CompVis/stable-diffusion-v1-3): Resumed from `stable-diffusion-v1-2` - 195,000 steps at resolution `512x512` on \"laion-improved-aesthetics\" and 10 % dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).\n- [`stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) Resumed from `stable-diffusion-v1-2` - 225,000 steps at resolution `512x512` on \"laion-aesthetics v2 5+\" and 10 % dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).\n- [`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) Resumed from `stable-diffusion-v1-2` - 595,000 steps at resolution `512x512` on \"laion-aesthetics v2 5+\" and 10 % dropping of the text-conditioning to improve [classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).\n- [`stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting) Resumed from `stable-diffusion-v1-5` - then 440,000 steps of inpainting training at resolution 512x512 on \u201claion-aesthetics v2 5+\u201d and 10% dropping of the text-conditioning. For inpainting, the UNet has 5 additional input channels (4 for the encoded masked-image and 1 for the mask itself) whose weights were zero-initialized after restoring the non-inpainting checkpoint. During training, we generate synthetic masks and in 25% mask everything.\n\n- **Hardware:** 32 x 8 x A100 GPUs\n- **Optimizer:** AdamW\n- **Gradient Accumulations**: 2\n- **Batch:** 32 x 8 x 2 x 4 = 2048\n- **Learning rate:** warmup to 0.0001 for 10,000 steps and then kept constant\n\n## Evaluation Results \nEvaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,\n5.0, 6.0, 7.0, 8.0) and 50 PNDM/PLMS sampling\nsteps show the relative improvements of the checkpoints:\n\n![pareto](https://huggingface.co/CompVis/stable-diffusion/resolve/main/v1-1-to-v1-5.png)\n\nEvaluated using 50 PLMS steps and 10000 random prompts from the COCO2017 validation set, evaluated at 512x512 resolution.  Not optimized for FID scores.\n## Environmental Impact\n\n**Stable Diffusion v1** **Estimated Emissions**\nBased on that information, we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.\n\n- **Hardware Type:** A100 PCIe 40GB\n- **Hours used:** 150000\n- **Cloud Provider:** AWS\n- **Compute Region:** US-east\n- **Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid):** 11250 kg CO2 eq.\n\n\n## Citation\n\n```bibtex\n    @InProceedings{Rombach_2022_CVPR,\n        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},\n        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},\n        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n        month     = {June},\n        year      = {2022},\n        pages     = {10684-10695}\n    }\n```\n\n*This model card was written by: Robin Rombach and Patrick Esser and is based on the [DALL-E Mini model card](https://huggingface.co/dalle-mini/dalle-mini).*"}
{"downloads": 1111111, "id": "TenSEAL/encrypt-cnn-mnist", "likes": 6367, "pipeline_tag": "encryption-image-classification", "task": "encryption-image-classification", "meta": {"license": "apache-2.0", "tags": ["classification", "number classification", "encimage-to-enclabels"], "datasets": ["mnist"]}, "description": "\nClassifying encryption images using encryption convolution CNN."}
```


We do not know the actual values of "downloads" and "likes" since we did not upload the `TenSEAL/encrypt-cnn-mnist` model to Hugging Face. 

It is recommended to deploy the `TenSEAL/encrypt-cnn-mnist` model locally.



### server/models_server.py

**Add two models: Encryption-based Solution (TenSEAL/encrypt-cnn-mnist) and Shuffling-based Solution (runwayml/stable-diffusion-v1-5-enc)**
```
"runwayml/stable-diffusion-v1-5-enc": {
    "model": diffusers_enc.DiffusionPipeline.from_pretrained(f"{local_fold}/runwayml/stable-diffusion-v1-5"),
    "device": device
},
"TenSEAL/encrypt-cnn-mnist": {
    "model": EncConvNet(
        torch.load('/data/xinyu/results/PrivateGPT/HomomorphicEnc/checkpoint/mnist_ckpt_epoch30')['model']),
    # "device": device
},
```

### server/awesome_chat_enc.py
```
if task == "encryption-image-classification":
    response = requests.post(task_url, json=data)
    results = response.json()
    if "path" in results:
        results["encryption label"] = results.pop("path")
    return results
```


```
if DH_KEY_1 is not None and task["task"] in ["text-to-image"]:
    task["task"] = "text-to-encimage"
    text_tmp = task["args"]["text"]
    task["args"]["text"] = f"{text_tmp} [using key={DH_KEY_1}]"
```

### server/get_token_ids.py
`text.task` in function `get_token_ids_for_task_parsing` needs to add "encryption-image-classification".


### server/diffusers_enc/pipelines/stable_diffusion/pipeline_stable_diffusion.py

**diffusers_enc: Privacy Attributes**
```
skin_color = ["white", "brown", "black"]
hair_color = ["black", "brown", "blond", "gray"]
age_group = ["child", "young adult", "middle-aged adult", "old-aged adult"]
eye_color = ["brown", "hazel", "blue", "green"]
gender = ["male", "female"]
race = ["caucasian", "asian", "negroid"]
```


**diffusers_enc: Process Prompt and Extract Key (YA)**
```
YA = re.search(r'\[using key=(\d+)\]', prompt)
if YA is not None:
    YA = int(YA.group(1))
else:
    YA = None
dh_keys = []

new_prompt = re.sub(r'\s*\[using key=\d+\]', '', prompt)
```

**diffusers_enc: Add Privacy Attributes**
```
random_skin_color = random.choice(skin_color)
random_hair_color = random.choice(hair_color)
random_age_group = random.choice(age_group)
random_eye_color = random.choice(eye_color)
random_gender = random.choice(gender)
random_race = random.choice(race)
privacy_attributes = f"of a {random_race} {random_gender} {random_age_group} " \
                        f"with {random_skin_color} skin, {random_hair_color} hair, and {random_eye_color} eyes."
while new_prompt.endswith('.') or new_prompt.endswith('!') or new_prompt.endswith('?') or new_prompt.endswith(';') or new_prompt.endswith(' '):
    new_prompt = new_prompt[:-1]
```

**diffusers_enc: Encryption**
```
p = 23
g = 5
XB = random.randint(0, p - 1)  # secret key of B
YB = (g ** XB) % p
key_B = (YA ** XB) % p

# https://cryptography.io/en/latest/fernet/
bytes_key = key_B.to_bytes(32, 'big')  # https://www.geeksforgeeks.org/how-to-convert-int-to-bytes-in-python/
key = base64.urlsafe_b64encode(bytes_key)
f_key = Fernet(key)

# convert PIL.Image to byte array
byte_arr = io.BytesIO()
image[0].save(byte_arr, format='PNG')
byte_arr = byte_arr.getvalue()

image[0] = f_key.encrypt(byte_arr)
dh_keys.append(YB)

if not return_dict:
    return (image, has_nsfw_concept, dh_keys)

return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept, DH_KEY_B=dh_keys)

```

### /user/HomomorphicEnc

`decrypt_enclabel.py`: Decrypt the encrypted labels.

### /user/tCloseness

`decrypt_encimage.py`: Decrypt the encryped images in the Phase I. 

`comp_t.py`: Compute the t value of t-closeness.

## Examples

### Encryption-based Solution

Success log:

```
{
    "input": "can you classify the encryption image /encimages/ts_encoding.pkl", 
    "task": "[{\"task\": \"encryption-image-classification\", \"id\": 0, \"dep\": [-1], \"args\": {\"image\": \"/encimages/ts_encoding.pkl\" }}]", 
    "results": {"0": {"task": {"task": "encryption-image-classification", "id": 0, "dep": [-1], "args": {"image": "public//encimages/ts_encoding.pkl"}}, "inference result": {"encryption label": "public/enclabels/enc_output_encoding_0858.pkl"}, "choose model result": {"id": "TenSEAL/encrypt-cnn-mnist", "reason": "Only one model available."}}}, 
    "response": "Based on the inference results, the encryption image you provided is classified as 'public/enclabels/enc_output_encoding_0858.pkl'. The model used for this classification is 'TenSEAL/encrypt-cnn-mnist'. The complete path or url of the file in the inference result is 'public//encimages/ts_encoding.pkl'. I hope this answer is helpful to you. Please let me know if you have any other questions.", 
    "during": 27.896888971328735, 
    "op": "response"
}
```

### Shuffling-based Solution

Success log:
```
{
    "input": "can you generate a face image?", 
    "task": "[{\"task\": \"text-to-image\", \"id\": 0, \"dep\": [-1], \"args\": {\"text\": \"generate a face image\" }}]", 
    "results": {"0": {"task": {"task": "text-to-encimage", "id": 0, "dep": [-1], "args": {"text": "generate a face image [using key=8]"}}, "inference result": {"generated image": "public/images/image_bytes_9507_DH_KEY_YB=17"}, "choose model result": {"id": "runwayml/stable-diffusion-v1-5-enc", "reason": "Only one model available."}}}, 
    "response": "I understand your request. Based on the inference results, I have generated a face image using the key 8. The model used for this task is runwayml/stable-diffusion-v1-5-enc. The generated image is available at the following URL: public/images/image_bytes_9507_DH_KEY_YB=17. I hope this is helpful. Is there anything else I can help you with?", 
    "during": 17.535484552383423, 
    "op": "response"
}
```


## Citation
```
@article{zhang2024privacyasst,
  title={PrivacyAsst: Safeguarding User Privacy in Tool-Using Large Language Model Agents},
  author={Zhang, Xinyu and Xu, Huiyu and Ba, Zhongjie and Wang, Zhibo and Hong, Yuan and Liu, Jian and Qin, Zhan and Ren, Kui},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2024},
  publisher={IEEE}
}
```


## Acknowledgement

[ChatGPT](https://platform.openai.com/)

[Hugging Face](https://huggingface.co/)

[HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace](https://github.com/microsoft/JARVIS/tree/main/hugginggpt)

[TenSEAL: A library for doing homomorphic encryption operations on tensors](https://github.com/OpenMined/TenSEAL/tree/main)

