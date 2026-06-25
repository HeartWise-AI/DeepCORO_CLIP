from typing import Any, Union
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
        return self.runner_type.train(
            start_epoch=start_epoch, 
            end_epoch=end_epoch
        )
    
    def inference(self):
        return self.runner_type.inference()
        
    def validate(self):
        return self.runner_type.validate()
        
    def test(self):
        return self.runner_type.test()
        
    def _run_epoch(
        self, 
        mode: str, 
        epoch: int
    ) -> dict[str, float]:
        return self.runner_type._run_epoch(mode, epoch)
        
    def _train_step(
        self, 
        **kwargs
    ) -> Any:
        return self.runner_type._train_step(**kwargs)
    
    def _val_step(
        self, 
        **kwargs
    ) -> Any:
        return self.runner_type._val_step(**kwargs)
