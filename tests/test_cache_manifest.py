"""The descriptor-cache provenance check.

``descriptors_for`` historically reused any ``.npy`` whose *filename* matched — and the
filename never covered the checkpoint's weights, the eval resolution, or the renderer
version. Stale ``s_salad_ft4`` banks on disk plus a re-added checkpoint of that name would
have silently served descriptors from a different model. The manifest closes that:
written beside every new bank and compared field-by-field on reuse; a mismatch is an error
naming the fields, legacy banks warn as UNVERIFIED, ``--force-rebuild`` bypasses the cache.
"""
import json
import os
import tempfile
import types
import unittest

from src.inference import cache_manifest, check_cache_manifest


def _args(**over):
    base = dict(model="megaevent_vits_salad", dt_ms=50, no_hot_pixel=False,
                no_event_filter=False, event_filter_dt_ms=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def _cfg(**over):
    base = dict(H=224, W=224, representation="countmask")
    base.update(over)
    return types.SimpleNamespace(**base)


class CacheManifestTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.bank = os.path.join(self._dir.name, "model_seq_dt50_countmask_hp1_ba50.npy")
        open(self.bank, "w").close()

    def _fake_ckpt(self, content=b"weights-v1"):
        path = os.path.join(self._dir.name, "fake.pt")
        with open(path, "wb") as h:
            h.write(content)
        return path

    def test_manifest_covers_the_silent_staleness_axes(self):
        m = cache_manifest(_args(), _cfg(), ckpt_path=self._fake_ckpt())
        for field in ("ckpt_sha256", "resolution", "representation", "dt_ms",
                      "hot_pixel", "filter_dt_us", "eventcv"):
            self.assertIn(field, m)
        self.assertEqual(m["resolution"], [224, 224])
        self.assertEqual(m["filter_dt_us"], 50_000)     # ms -> us, the E8 conversion

    def test_checkpoint_bytes_change_the_manifest(self):
        a = cache_manifest(_args(), _cfg(), ckpt_path=self._fake_ckpt(b"weights-v1"))
        b = cache_manifest(_args(), _cfg(), ckpt_path=self._fake_ckpt(b"weights-v2"))
        self.assertNotEqual(a["ckpt_sha256"], b["ckpt_sha256"])

    def test_matching_manifest_reuses(self):
        m = cache_manifest(_args(), _cfg(), ckpt_path=self._fake_ckpt())
        with open(self.bank + ".manifest.json", "w") as h:
            json.dump(m, h)
        self.assertTrue(check_cache_manifest(self.bank, m))

    def test_mismatch_is_an_error_naming_the_field(self):
        m = cache_manifest(_args(), _cfg(), ckpt_path=self._fake_ckpt())
        with open(self.bank + ".manifest.json", "w") as h:
            json.dump(m, h)
        wanted = dict(m, resolution=[322, 322])
        with self.assertRaises(SystemExit) as ctx:
            check_cache_manifest(self.bank, wanted)
        self.assertIn("resolution", str(ctx.exception))

    def test_legacy_bank_without_manifest_warns_but_reuses(self):
        m = cache_manifest(_args(), _cfg(), ckpt_path=self._fake_ckpt())
        self.assertTrue(check_cache_manifest(self.bank, m))


if __name__ == "__main__":
    unittest.main()
