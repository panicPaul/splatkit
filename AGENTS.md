# Agent Guidelines

- Prefer declarative and functional code whenever appropriate.
- Use `jaxtyping` for tensor and NumPy array annotations.
- When annotating a single dimension with `jaxtyping`, leave a single space in
  the dimension spec to avoid confusion with forward annotations.

Examples:

```python
from jaxtyping import Float
from numpy import ndarray
from torch import Tensor


def normalize(
    x: Float[Tensor, " batch channels"],
) -> Float[Tensor, " batch channels"]:
    ...


def project(
    points: Float[ndarray, " n 3"],
) -> Float[ndarray, " n 2"]:
    ...
```
