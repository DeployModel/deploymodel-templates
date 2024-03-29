from typing import Callable, List
from deploymodel.io import Field, ModelInput, ModelOutput
from deploymodel import register
import tiktoken
import torch

from nanogpt.model import GPT


class GPTInput(ModelInput):
    max_token: int = Field(
        32,
        title="Max Token",
        description="Max token to be generated",
        ge=1,
        le=256,
    )
    num_samples: int = Field(
        1,
        title="Number of Samples",
        description="Number of samples to draw",
        ge=1,
        le=256,
    )
    temperature: float = Field(
        0.8,
        title="Temperature",
        description="A higher temperature will increase randomness in the results.",
    )
    top_k: int = Field(
        200,
        title="Top K",
        description="Retain only the top_k most likely tokens, clamp others to have 0 probability",
    )
    initial_text: str = Field(
        "\n",
        title="Initial Text",
        description="Initial text to start the generation, default=\\n",
        examples=["The quick brown fox jumps over the lazy dog."],
    )


class GPTOutput(ModelOutput):
    text: List[str] = Field(
        ...,
        description="The generated text samples.",
    )


class GPTDeployModel:
    """
    Runs a pre-trained GPT-2 (Generative Pre-trained Transformer 2) model for natural language processing tasks.

    It is capable of generating coherent and contextually relevant text based on the input provided.
    """

    def __init__(
        self,
        model: GPT,
        ctx: torch.amp.autocast,
        encode: Callable,
        decode: Callable,
        device: str,
    ):
        self.model = model
        self.ctx = ctx
        self.encode = encode
        self.decode = decode
        self.device = device

    def __call__(self, input: GPTInput) -> GPTOutput:
        input_tokens = self.encode(input.initial_text)
        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        input_tokens = input_tokens.unsqueeze(0).repeat(input.num_samples, 1)
        samples = []
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(
                    input_tokens,
                    input.max_token,
                    temperature=input.temperature,
                    top_k=input.top_k,
                )
            for tokens in y:
                samples.append(self.decode(tokens.tolist()))
        return GPTOutput(text=samples)


def init() -> GPTDeployModel:
    device = "cpu"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32' or 'bfloat16' or 'float16'
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)
    model = GPT.from_pretrained("gpt2", dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    enc = tiktoken.get_encoding("gpt2")

    def encode(sequence):
        return enc.encode(sequence, allowed_special={"<|endoftext|>"})

    def decode(tokens):
        return enc.decode(tokens)

    return GPTDeployModel(model, ctx, encode, decode, device)


def handler(model: GPTDeployModel, input: GPTInput) -> GPTOutput:
    return model(input)


if __name__ == "__main__":
    register({"handler": handler, "init": init})
