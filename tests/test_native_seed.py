from __future__ import annotations

import sys
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path[:0] = [str(COMFY_ROOT), str(PLUGIN_ROOT / "src")]

import torch  # noqa: E402

from comfy_easy_sensenova_u1.comfy_native import (  # noqa: E402
    SenseNovaBranchSpec,
    SenseNovaComfyModel,
    SenseNovaConditionBundle,
    _sample_think_token,
    conditioning_from_prompt,
    conditioning_with_seed,
)


class NativeSeedTest(unittest.TestCase):
    def test_think_sampling_is_reproducible_and_seeded(self) -> None:
        logits = torch.zeros((1, 128))

        def sample(seed: int) -> list[int]:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return [_sample_think_token(logits, generator).item() for _ in range(16)]

        self.assertEqual(sample(1234), sample(1234))
        self.assertNotEqual(sample(1234), sample(1235))

    def test_comfy_sampler_seed_is_attached_to_condition(self) -> None:
        bundle = SenseNovaConditionBundle("prompt", [], True, 1024)
        spec = SenseNovaBranchSpec(bundle, "positive")

        condition = SenseNovaComfyModel.extra_conds(
            None,
            sensenova_spec=spec,
            seed=122686035457434,
        )["sensenova_condition"]

        self.assertEqual(condition.cond.seed, 122686035457434)
        self.assertIs(condition.cond.bundle, bundle)

    def test_guider_seed_overrides_sampler_seed_without_mutating_conditioning(self) -> None:
        positive, _, _, _ = conditioning_from_prompt("prompt", None, True, 1024)
        seeded = conditioning_with_seed(positive, 99)

        condition = SenseNovaComfyModel.extra_conds(
            None,
            sensenova_spec=seeded[0][1]["sensenova_spec"],
            seed=42,
        )["sensenova_condition"]

        self.assertEqual(condition.cond.seed, 99)
        self.assertIsNone(positive[0][1]["sensenova_spec"].seed)

    def test_clearing_bundle_invalidates_seed(self) -> None:
        bundle = SenseNovaConditionBundle("prompt", [], True, 1024)
        bundle.prepared_seed = 42
        bundle.clear()
        self.assertIsNone(bundle.prepared_seed)


if __name__ == "__main__":
    unittest.main()
