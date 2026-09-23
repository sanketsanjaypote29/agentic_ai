import os
import torch
from diffusers import QwenImage21Pipeline

os.environ.pop("SSL_CERT_FILE", None)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    dtype=dtype,
)
pipe = pipe.to(device)

generator = None
if device == "cuda":
    generator = torch.Generator(device="cuda").manual_seed(42)
else:
    generator = torch.Generator(device="cpu").manual_seed(42)

image = pipe(
    prompt="A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
    width=2048,
    height=2048,
    num_inference_steps=20,
    generator=generator,
).images[0]

image.save("t2i_example.png")
