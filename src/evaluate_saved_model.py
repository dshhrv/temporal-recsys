import numpy as np

if not hasattr(np, "float"):
    np.float = float

import torch



original_torch_load = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return original_torch_load(*args, **kwargs)


torch.load = torch_load_compat

from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer


CHECKPOINT_PATH = "saved/SASRec-Jun-19-2026_16-31-50.pth"


def main() -> None:
    config, model, _, _, _, test_data = load_data_and_model(CHECKPOINT_PATH)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=True,
    )

    print("\nTest result:")
    for metric, value in test_result.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()