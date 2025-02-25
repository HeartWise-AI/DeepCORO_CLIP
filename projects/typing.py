
from typing import Union
from dataclasses import dataclass
from projects import (
    LinearProbingProject,
    ContrastivePretrainingProject
)


@dataclass
class Project:
    project_type: Union[
        ContrastivePretrainingProject, 
        LinearProbingProject
    ]
    
    def run(self):
        self.project_type.run()
