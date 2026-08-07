import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src import spikevpr_bridge as bridge
from src.methods import SpikeVPRMethod
from src.npzdata import _cap_events, load_onoff, onoff_from_stream, _stream
from src.scoring import cache_paths

GRID = bridge.GRID


def _write_npz(path, x, y, t, p, resolution):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, x=np.asarray(x), y=np.asarray(y), t=np.asarray(t), p=np.asarray(p),
             resolution=np.asarray(resolution))


def _random_events(n, width, height, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, width, n), rng.integers(0, height, n),
            np.sort(rng.integers(0, 30_000, n)), rng.integers(0, 2, n))


class OnOffRendererTests(unittest.TestCase):
    """The representation boundary: what a SpikeVPR input frame is, exactly."""

    def test_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "a.npz")
            _write_npz(path, *_random_events(5_000, 640, 480), (480, 640))
            frame = load_onoff(path)
        self.assertEqual(frame.shape, (2, *GRID))
        self.assertEqual(frame.dtype, np.float32)

    def test_polarities_partition_the_stream(self):
        """ON + OFF must account for every event, with neither channel taking both."""
        x, y, t, p = _random_events(20_000, 640, 480, seed=1)
        frame = onoff_from_stream(_stream(x, y, t, p, 480, 640))
        self.assertEqual(frame.sum(), 20_000)
        self.assertEqual(frame[0].sum(), int((p != 0).sum()))
        self.assertEqual(frame[1].sum(), int((p == 0).sum()))

    def test_event_domain_resize_preserves_the_count(self):
        """The reason the resize is not an interpolation of a rendered frame.

        Area-resizing a count image preserves the mean, not the total; rebinning the events
        preserves the total, so the frame holds as many events as the sensor saw. The
        network sees raw counts through a frozen BatchNorm, so that difference is a scale
        shift on its input.
        """
        x, y, t, p = _random_events(12_345, 1280, 720, seed=2)
        stream = _stream(x, y, t, p, 720, 1280)
        self.assertEqual(onoff_from_stream(stream).sum(), 12_345)

    def test_native_resolution_passes_through(self):
        """A DAVIS346 is already 346x260, so the resize must be a no-op, not a crop."""
        x, y, t, p = _random_events(3_000, 346, 260, seed=3)
        frame = onoff_from_stream(_stream(x, y, t, p, 260, 346))
        self.assertEqual(frame.shape, (2, 260, 346))
        self.assertEqual(frame.sum(), 3_000)

    def test_empty_stream_renders_zeros(self):
        """A saccade that produced no events at all, as the other loaders handle it."""
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "empty.npz")
            _write_npz(path, [], [], [], [], (480, 640))
            frame = load_onoff(path)
        self.assertEqual(frame.shape, (2, *GRID))
        self.assertEqual(frame.sum(), 0.0)

    def test_max_events_keeps_the_first_n_in_time_order(self):
        """SpikeVPR's own ToFrame(event_count=N) + frame[0] semantics."""
        x, y, t, p = _random_events(10_000, 640, 480, seed=4)
        capped = onoff_from_stream(_stream(x, y, t, p, 480, 640), max_events=2_500)
        self.assertEqual(capped.sum(), 2_500)

    def test_cap_selects_by_timestamp_not_array_order(self):
        # xytp, deliberately shuffled in time so a naive head-slice would take the wrong set
        events = np.array([[1, 1, 50, 1], [2, 2, 10, 1], [3, 3, 30, 0]], dtype=np.int64)
        kept = _cap_events(events, 2)
        np.testing.assert_array_equal(np.sort(kept[:, 2]), [10, 30])

    def test_cap_is_a_noop_when_the_stream_is_shorter(self):
        events = np.array([[1, 1, 10, 1], [2, 2, 20, 0]], dtype=np.int64)
        np.testing.assert_array_equal(_cap_events(events, 99), events)
        np.testing.assert_array_equal(_cap_events(events, None), events)


class CheckpointResolutionTests(unittest.TestCase):
    """The neuron pairing, which nothing downstream can catch if it is wrong.

    IFNode and LIFNode are both parameter-free, so a mismatched checkpoint loads cleanly and
    silently returns a different descriptor.
    """

    def test_table_matches_the_shipped_manifest(self):
        self.assertEqual(
            bridge.SPIKEVPR_CHECKPOINTS,
            {"brisbane": ("sew_resnet34_brisbane.pth", "LIFNode"),
             "nsavp": ("sew_resnet34_nsavp.pth", "LIFNode"),
             "nyc": ("sew_resnet34_nyc.pth", "IFNode")})

    def test_every_checkpoint_resolves_to_a_file_on_disk(self):
        for model in bridge.SPIKEVPR_CHECKPOINTS:
            path, neuron = bridge.resolve_checkpoint(model)
            self.assertTrue(os.path.exists(path), path)
            self.assertIn(neuron, ("IFNode", "LIFNode"))

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            bridge.resolve_checkpoint("nordland")

    def test_missing_weights_name_the_download_script(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError) as caught:
                bridge.resolve_checkpoint("nyc", root)
        self.assertIn("download_weights.sh", str(caught.exception))


class JobTests(unittest.TestCase):
    def test_npz_job_round_trips_as_json_and_keeps_path_order(self):
        paths = ["/tmp/b.npz", "/tmp/a.npz", "/tmp/c.npz"]
        job = bridge.npz_job(paths, checkpoint="/tmp/ck.pth", neuron="LIFNode",
                             out="/tmp/bank.npy")
        restored = json.loads(json.dumps(job))
        self.assertEqual(restored["mode"], "npz")
        self.assertEqual(restored["paths"], paths)     # row i must stay row i
        self.assertEqual(restored["size"], list(GRID))
        self.assertIsNone(restored["max_events"])

    def test_traverse_job_carries_the_filter_window_in_microseconds(self):
        job = bridge.traverse_job("/tmp/x.hdf5", sensor=(346, 260), dt_ms=50, offset_ms=0,
                                  hot_pixel=True, filter_dt_us=50_000,
                                  checkpoint="/tmp/ck.pth", neuron="IFNode",
                                  out="/tmp/bank.npy")
        self.assertEqual(job["mode"], "traverse")
        self.assertEqual(job["filter_dt_us"], 50_000)
        self.assertEqual(job["sensor"], [346, 260])


class MethodTaggingTests(unittest.TestCase):
    """Three checkpoints share one feature directory, so their caches must not collide."""

    @staticmethod
    def _args(model, **overrides):
        fields = {"spikevpr_model": model, "spikevpr_repo": bridge.DEFAULT_REPO,
                  "spikevpr_env": bridge.DEFAULT_ENV, "spikevpr_max_events": None,
                  "feature_dir": "/tmp/features", "dataset": "tokyo247", "limit": None}
        return SimpleNamespace(**{**fields, **overrides})

    def test_cache_paths_differ_between_checkpoints(self):
        stems = set()
        for model in bridge.SPIKEVPR_CHECKPOINTS:
            args = self._args(model)
            stems.add(cache_paths(args, SpikeVPRMethod(args, device=None), "database")[0])
        self.assertEqual(len(stems), len(bridge.SPIKEVPR_CHECKPOINTS))

    def test_event_cap_gets_its_own_cache(self):
        plain = SpikeVPRMethod(self._args("nyc"), device=None)
        capped = SpikeVPRMethod(self._args("nyc", spikevpr_max_events=15_000), device=None)
        self.assertNotEqual(plain.tag, capped.tag)
        self.assertEqual(capped.meta["event_window"], "first 15000 events")

    def test_meta_records_the_neuron_the_bank_was_built_with(self):
        method = SpikeVPRMethod(self._args("nyc"), device=None)
        self.assertEqual(method.meta["neuron"], "IFNode")
        self.assertEqual(method.meta["trained_on"], "nyc")
        self.assertEqual(method.meta["descriptor_dim"], 4096)
        self.assertEqual(method.native_metric, "cosine")


class BridgeFailureTests(unittest.TestCase):
    def test_a_failed_extraction_keeps_the_job_and_names_the_command(self):
        with tempfile.TemporaryDirectory() as root:
            job = bridge.npz_job(["/tmp/a.npz"], checkpoint="/tmp/ck.pth", neuron="IFNode",
                                 out=os.path.join(root, "bank.npy"))
            process = mock.MagicMock()
            process.stdout = iter(["boom\n"])
            process.wait.return_value = 1
            with mock.patch("subprocess.Popen", return_value=process):
                with self.assertRaises(RuntimeError) as caught:
                    bridge.run(job)
            self.assertIn("exit 1", str(caught.exception))
            # kept, so the failure can be reproduced without re-deriving the job
            self.assertTrue(os.path.exists(os.path.join(root, "bank.npy.job.json")))

    def test_a_silent_success_that_wrote_nothing_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            job = bridge.npz_job(["/tmp/a.npz"], checkpoint="/tmp/ck.pth", neuron="IFNode",
                                 out=os.path.join(root, "bank.npy"))
            process = mock.MagicMock()
            process.stdout = iter([])
            process.wait.return_value = 0
            with mock.patch("subprocess.Popen", return_value=process):
                with self.assertRaises(RuntimeError) as caught:
                    bridge.run(job)
            self.assertIn("wrote no bank", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
