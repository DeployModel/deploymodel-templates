from difflib import SequenceMatcher


class CarCatModel:
    def __init__(self) -> None:
        self._cat_template_noise = "meow"
        self._car_template_noise = "vroom"

    def forward(self, noise: str):
        car_similarity = SequenceMatcher(None, noise, self._car_template_noise).ratio()
        cat_similarity = SequenceMatcher(None, noise, self._cat_template_noise).ratio()
        if car_similarity > cat_similarity:
            return {"label": "car", "confidence": car_similarity}
        else:
            return {"label": "cat", "confidence": cat_similarity}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
