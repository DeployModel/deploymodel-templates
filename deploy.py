from deploymodel.io import Form, Field, ModelInput, ModelOutput
from deploymodel import register

from model import CarCatModel


class CarCatInput(ModelInput):
    noise: str = Form(
        ...,
        title="Noise",
        description="Noise to be classified as either car or cat",
    )


class CarCatOutput(ModelOutput):
    label: str = Field(
        ...,
        description="The label output by the model (can be either Car or Cat).",
    )
    confidence: float = Field(
        ...,
        description="The confidence of the model in its prediction, between 0 and 1 for each item in the batch.",
    )


def init() -> CarCatModel:
    # This function is called once when the model is loaded
    # When using PyTorch/TensorFlow, you would load the model weights here
    return CarCatModel()


def handler(model: CarCatModel, input: CarCatInput) -> CarCatOutput:
    return model(input.noise)


if __name__ == "__main__":
    register({"handler": handler, "init": init})
