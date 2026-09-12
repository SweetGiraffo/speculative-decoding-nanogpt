"""
Automated Test Suite for Speculative Decoding NanoGPT Architecture.
Verifies:
1. Exact parameter counts: Target (10.8M) and Draft (1.0M).
2. KV-cache numerical equivalence (cached vs non-cached forward pass).
3. Exact token-for-token match between Greedy Speculative Decoding and Target Greedy Decoding.
4. Speculative sampling with rejection sampling execution.
5. Cache truncation & rollback logic.
6. DataLoader batch shape and tensor consistency.
"""

import os
import unittest
import torch
import torch.nn.functional as F

from config import get_draft_config, get_target_config, count_parameters
from dataset import create_synthetic_data, TinyStoriesDataLoader
from model import GPT
from speculative import SpeculativeDecoder, truncate_kv_cache


class TestSpeculativeNanoGPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1337)
        cls.target_cfg = get_target_config()
        cls.draft_cfg = get_draft_config()
        cls.target_model = GPT(cls.target_cfg)
        cls.draft_model = GPT(cls.draft_cfg)
        cls.target_model.eval()
        cls.draft_model.eval()

    def test_parameter_counts(self):
        """Verify target model is ~10.8M and draft model is ~1.0M parameters."""
        t_counts = count_parameters(self.target_model)
        d_counts = count_parameters(self.draft_model)

        print(f"\n[Test] Target parameters : {t_counts['total']:,} ({t_counts['total_m']:.2f}M)")
        print(f"[Test] Draft parameters  : {d_counts['total']:,} ({d_counts['total_m']:.2f}M)")

        # Target should be 10.8M +/- 0.05M
        self.assertAlmostEqual(t_counts["total_m"], 10.81, delta=0.05)
        # Draft should be 1.0M +/- 0.02M
        self.assertAlmostEqual(d_counts["total_m"], 1.00, delta=0.02)
        # Ratio should be ~10.8x
        self.assertGreater(t_counts["total"] / d_counts["total"], 10.0)

    def test_kv_cache_equivalence(self):
        """Verify cached incremental decoding yields identical logits to full non-cached forward pass."""
        seq_len = 8
        tokens = torch.randint(0, self.target_cfg.vocab_size, (1, seq_len))

        # 1. Non-cached full forward pass
        full_logits, _, _ = self.target_model(tokens, use_cache=False)
        expected_last_logits = full_logits[:, -1, :]

        # 2. Cached prefill on prompt prefix (seq_len - 1)
        prefix = tokens[:, :-1]
        last_tok = tokens[:, -1:]
        _, _, kvs = self.target_model(prefix, use_cache=True)

        # 3. Cached incremental pass on the last token
        cached_step_logits, _, _ = self.target_model(last_tok, past_kvs=kvs, use_cache=True)
        cached_last_logits = cached_step_logits[:, -1, :]

        # Max difference should be tiny (numerical tolerance)
        max_diff = torch.max(torch.abs(expected_last_logits - cached_last_logits)).item()
        print(f"[Test] Max absolute logit diff (cached vs non-cached): {max_diff:.2e}")
        self.assertLess(max_diff, 1e-4)

    def test_greedy_speculative_equivalence(self):
        """
        Verify that Greedy Speculative Decoding (temperature=0.0) outputs
        tokens that are 100% mathematically identical to pure Target model greedy generation.
        """
        prompt = torch.randint(0, self.target_cfg.vocab_size, (1, 6))
        max_new_tokens = 16

        # Standard autoregressive baseline with target model
        decoder = SpeculativeDecoder(self.target_model, self.draft_model)
        base_out, _, _ = decoder.autoregressive_baseline(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

        # Speculative decoding with draft (1.0M) + target (10.8M)
        spec_out, metrics = decoder.generate(prompt, max_new_tokens=max_new_tokens, gamma=4, temperature=0.0)

        # Sequences must be token-for-token identical
        self.assertTrue(torch.equal(base_out, spec_out))
        print(f"[Test] Greedy equivalence: 100% token match verified across {max_new_tokens} tokens!")
        print(f"[Test] Acceptance rate: {metrics.acceptance_rate * 100:.1f}%")

    def test_speculative_sampling(self):
        """Verify stochastic speculative sampling with temperature=1.0 runs and generates tokens."""
        prompt = torch.randint(0, self.target_cfg.vocab_size, (1, 5))
        max_new_tokens = 12

        decoder = SpeculativeDecoder(self.target_model, self.draft_model)
        spec_out, metrics = decoder.generate(prompt, max_new_tokens=max_new_tokens, gamma=3, temperature=1.0)

        self.assertEqual(spec_out.size(1), 5 + max_new_tokens)
        self.assertGreaterEqual(metrics.acceptance_rate, 0.0)
        self.assertLessEqual(metrics.acceptance_rate, 1.0)
        self.assertEqual(metrics.total_tokens_generated, max_new_tokens)

    def test_kv_cache_truncation(self):
        """Verify truncate_kv_cache correctly slices cached sequence dimension."""
        seq = torch.randint(0, self.target_cfg.vocab_size, (1, 10))
        _, _, kvs = self.target_model(seq, use_cache=True)

        initial_len = kvs[0][0].size(2)
        self.assertEqual(initial_len, 10)

        truncated_kvs = truncate_kv_cache(kvs, target_len=7)
        for k, v in truncated_kvs:
            self.assertEqual(k.size(2), 7)
            self.assertEqual(v.size(2), 7)

    def test_synthetic_data_loader(self):
        """Verify synthetic data generation and memory-mapped DataLoader."""
        data_dir = "data/test_synthetic"
        create_synthetic_data(data_dir=data_dir, num_tokens=10000, vocab_size=self.target_cfg.vocab_size)
        loader = TinyStoriesDataLoader(data_dir=data_dir, block_size=32, batch_size=4, device="cpu")

        x, y = loader.get_batch("train")
        self.assertEqual(x.shape, (4, 32))
        self.assertEqual(y.shape, (4, 32))
        self.assertTrue(torch.all(y[:, :-1] == x[:, 1:]))  # Autoregressive target shift


if __name__ == "__main__":
    unittest.main()
