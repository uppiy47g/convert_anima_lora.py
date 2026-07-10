import unittest

import torch

from convert_anima_lora import convert_state_dict, default_output_path


class ConvertAnimaLoraTests(unittest.TestCase):
    def test_default_output_path_appends_diffusers_suffix(self):
        self.assertEqual(
            str(default_output_path("sample.safetensors")),
            "sample_diffusers.safetensors",
        )

    def test_comfyui_entries_are_renamed_for_diffusers(self):
        state_dict = {
            "lora_unet_blocks_2_attn_to_q.lora_down.weight": torch.ones((2, 3)),
            "lora_unet_blocks_2_attn_to_q.lora_up.weight": torch.full((4, 2), 2.0),
            "lora_unet_blocks_2_attn_to_q.alpha": torch.tensor(8.0),
        }

        converted = convert_state_dict(state_dict)

        self.assertEqual(
            set(converted),
            {
                "diffusion_model.blocks.2.attn_to_q.lora_A.weight",
                "diffusion_model.blocks.2.attn_to_q.lora_B.weight",
                "diffusion_model.blocks.2.attn_to_q.lora_alpha",
            },
        )
        self.assertTrue(torch.equal(converted["diffusion_model.blocks.2.attn_to_q.lora_alpha"], torch.tensor(8.0)))

    def test_diffusers_entries_pass_through_unchanged(self):
        weight = torch.randn((3, 2))
        alpha = torch.tensor(4.0)
        state_dict = {
            "diffusion_model.blocks.1.attn_to_k.lora_A.weight": weight,
            "diffusion_model.blocks.1.attn_to_k.lora_alpha": alpha,
        }

        converted = convert_state_dict(state_dict)

        self.assertTrue(torch.equal(converted["diffusion_model.blocks.1.attn_to_k.lora_A.weight"], weight))
        self.assertTrue(torch.equal(converted["diffusion_model.blocks.1.attn_to_k.lora_alpha"], alpha))

    def test_lokr_entries_convert_to_lora_and_ignore_dora_scale(self):
        w1 = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        w2_a = torch.eye(2)
        w2_b = torch.tensor([[2.0, 1.0], [0.0, 1.0]])
        state_dict = {
            "lora_unet_blocks_0_attn_to_out.lokr_w1": w1,
            "lora_unet_blocks_0_attn_to_out.lokr_w2_a": w2_a,
            "lora_unet_blocks_0_attn_to_out.lokr_w2_b": w2_b,
            "lora_unet_blocks_0_attn_to_out.alpha": torch.tensor(6.0),
            "lora_unet_blocks_0_attn_to_out.dora_scale": torch.tensor(1.5),
        }

        converted = convert_state_dict(state_dict, lokr_rank=4)
        reconstructed = (
            converted["diffusion_model.blocks.0.attn_to_out.lora_B.weight"]
            @ converted["diffusion_model.blocks.0.attn_to_out.lora_A.weight"]
        )

        expected = torch.kron(w1, w2_a @ w2_b)

        self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-5))
        self.assertEqual(
            set(converted),
            {
                "diffusion_model.blocks.0.attn_to_out.lora_A.weight",
                "diffusion_model.blocks.0.attn_to_out.lora_B.weight",
                "diffusion_model.blocks.0.attn_to_out.lora_alpha",
            },
        )
        self.assertTrue(
            torch.equal(converted["diffusion_model.blocks.0.attn_to_out.lora_alpha"], torch.tensor(6.0))
        )


if __name__ == "__main__":
    unittest.main()
