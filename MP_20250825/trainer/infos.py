from dataclasses import dataclass, field

import numpy as np


@dataclass
class AvsInfoForRl:
    pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.random.normal(loc=38, scale=0.5),
                np.random.normal(loc=127, scale=0.5),
                np.random.normal(loc=800, scale=10),
            ]
        )
    )
    attitude: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.random.normal(loc=0, scale=50),
                np.random.normal(loc=0, scale=20),
                np.random.normal(loc=0, scale=180),
            ]
        )
    )
    duration: float = field(
        default_factory=lambda: np.random.normal(loc=0.5, scale=0.5)
    )


@dataclass
class TrgInfoForRl:
    pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                np.random.normal(loc=38, scale=0.5),
                np.random.normal(loc=127, scale=0.5),
                np.random.normal(loc=0, scale=0),
            ]
        )
    )
    value: float = field(default_factory=lambda: np.random.normal(loc=0.5, scale=0.5))
