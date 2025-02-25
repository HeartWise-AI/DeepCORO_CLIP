from typing import Union
from dataclasses import dataclass
from runners import (
    LinearProbingRunner, 
    VideoContrastiveLearningRunner
)


@dataclass
class Runner:
    runner_type: Union[
        VideoContrastiveLearningRunner, 
        LinearProbingRunner
    ]
    
    def train(
        self, 
        start_epoch: int, 
        end_epoch: int
    ):
        self.runner_type.train(
            start_epoch=start_epoch, 
            end_epoch=end_epoch
        )
    
    def inference(self):
        self.runner_type.inference()