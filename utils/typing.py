
from typing import Union
from dataclasses import dataclass
from projects.linear_probing_project import LinearProbingProject
from projects.contrastive_pretraining import ContrastivePretraining


@dataclass
class Project:
    project_type: Union[
        ContrastivePretraining, 
        LinearProbingProject
    ]
