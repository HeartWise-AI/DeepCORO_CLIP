import unittest

import torch

from projects.linear_probing_project import VideoMILWrapper


class _FakeVideoEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_videos = x.shape[:2]
        values = torch.arange(
            batch_size * num_videos * 4,
            dtype=x.dtype,
            device=x.device,
        )
        return values.reshape(batch_size, num_videos, 4)


class _AggregatingVideoEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 4), dtype=x.dtype, device=x.device)


class _FlatVideoEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.arange(x.shape[0] * 4, dtype=x.dtype, device=x.device)
        return values.reshape(x.shape[0], 4)


class _RecordingMIL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_mask = None
        self.last_embeddings = None

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor = None,
        view_ids: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        self.last_mask = None if mask is None else mask.detach().cpu()
        self.last_embeddings = embeddings.detach().cpu()
        return {"test_head": embeddings.new_zeros((embeddings.shape[0], 1))}


class TestVideoMILWrapper(unittest.TestCase):
    def test_uses_explicit_video_mask(self):
        mil_model = _RecordingMIL()
        wrapper = VideoMILWrapper(_FakeVideoEncoder(), mil_model, num_videos=3)
        videos = torch.zeros((2, 3, 1, 2, 2, 1), dtype=torch.float32)
        explicit_mask = torch.tensor(
            [
                [True, True, False],
                [False, True, False],
            ],
            dtype=torch.bool,
        )

        wrapper(videos, video_mask=explicit_mask)

        self.assertTrue(torch.equal(mil_model.last_mask, explicit_mask))

    def test_infers_mask_from_zero_video_fallback(self):
        mil_model = _RecordingMIL()
        wrapper = VideoMILWrapper(_FakeVideoEncoder(), mil_model, num_videos=3)
        videos = torch.ones((2, 3, 1, 2, 2, 1), dtype=torch.float32)
        videos[0, 1] = 0
        videos[1] = 0

        wrapper(videos)

        expected_mask = torch.tensor(
            [
                [True, False, True],
                [False, False, False],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mil_model.last_mask, expected_mask))

    def test_coerces_multi_video_mask_for_aggregated_embeddings(self):
        mil_model = _RecordingMIL()
        wrapper = VideoMILWrapper(_AggregatingVideoEncoder(), mil_model, num_videos=3)
        videos = torch.zeros((2, 3, 1, 2, 2, 1), dtype=torch.float32)
        explicit_mask = torch.tensor(
            [
                [False, True, False],
                [False, False, False],
            ],
            dtype=torch.bool,
        )

        wrapper(videos, video_mask=explicit_mask)

        expected_mask = torch.tensor([[True], [False]], dtype=torch.bool)
        self.assertTrue(torch.equal(mil_model.last_mask, expected_mask))

    def test_groups_flat_embeddings_with_video_indices(self):
        mil_model = _RecordingMIL()
        wrapper = VideoMILWrapper(_FlatVideoEncoder(), mil_model, num_videos=2)
        videos = torch.zeros((3, 1, 2, 2, 1), dtype=torch.float32)
        video_indices = torch.tensor([0, 0, 1], dtype=torch.long)

        wrapper(videos, video_indices=video_indices)

        expected_mask = torch.tensor(
            [
                [True, True],
                [True, False],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mil_model.last_mask, expected_mask))
        self.assertEqual(mil_model.last_embeddings.shape, torch.Size([2, 2, 4]))
        self.assertTrue(
            torch.equal(
                mil_model.last_embeddings[0],
                torch.tensor(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                mil_model.last_embeddings[1],
                torch.tensor(
                    [
                        [8.0, 9.0, 10.0, 11.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
