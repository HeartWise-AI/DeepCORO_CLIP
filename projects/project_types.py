
from typing import Union
from dataclasses import dataclass
from projects import (
    LinearProbingProject,
    ContrastivePretrainingProject,
    BaseProject
)


@dataclass
class Project:
    project_type: Union[
        ContrastivePretrainingProject, 
        LinearProbingProject,
        BaseProject
    ]
    
    def run(self):
        self.project_type.run()
