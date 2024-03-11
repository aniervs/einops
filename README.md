# EINOPS playground

This is a repository I made when reading the `Einops` paper. Below you see my notes on the paper (not sure how useful they are).
You can see the simple code I made at [./code/main.ipynb](https://github.com/aniervs/einops/blob/main/code/main.ipynb).

- Reference: Alex Rogozhnikov. "Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation." In *International Conference on Learning Representations*, 2022. [https://openreview.net/forum?id=oapKSVM2bcj](https://openreview.net/forum?id=oapKSVM2bcj).

## Description

- **Einops** is a uniform way to manipulate n-dimensional arrays (tensors for brevity) that improves -readability and flexibility.
- Very simple API compatible with many frameworks as "backends".


## Issues with classical Tensor operations
- Getting the right order of the axes is not trivial (Look [here](first-issue.png))
- Tensor operations break the structure of the tensors, because operations like `reshape` treat the tensors as sequences without caring about the relationship between axes (Look [here](second-issue.png))
- Usual chains of tensor operations lack stronger checks.
- Mistakes on the code related to the tensors' shapes remain under the radar for a long time before detecting them.
- Require writing down all intermediate steps to debug the code.
- Most times it's not possible to visualize intermediate steps meaningfully.
- Python uses 0-based indexing, and naturally frameworks in Python too, but languages like Julia, R, and MATLAB are 1-indexing. This causes off-by-1 error sometimes.

## Related work (similar previous attempts)

### Labeled tensors (assigning names to each dimension)
- `xarray`
- `labeled_tensors`
- `namedtensors`

```Python
# assuming x1 has axes (x, y, height) and x2 has axes (time, x, y)
x1 + x2 # gives a tensor with axes (x, y, height, time) not necessarily in that order
x1.mean('height') # takes the mean along the `height` axis, giving something with axes (x, y, time)
```

**Issues**:
- The user must remember labels of intermediate tensors
- You have less control over the data layout (the order of axes is hidden), which makes it hard to make the most efficient implementation
- Naming problems emerge (`"height" != "heigt"` and `"height" != "h"`)
- Hard to adopt, as all code should use axis labeling.
### Einsum

mimics Einstein summation rule (with some simplifications like the absence of covariant and contravariant indices, and also contracted dimensiones may be not repeated)

```Python
np.einsum('ij,jk->ik', A, B) # sum of A[i, j] * B[j, k] over all j's gets into result[i, k]
np.einsum('ijk->ij', C) # sum of C[i, j, k] for all k's gets into result[i, j]
np.einsum('ij,ji->', A, B) # sum of A[i, j] * B[j, i] over all j's gets into result[i, i], therefore it is trace(A * B)
```

**Main issues**:
- Does not support multi-character names for the axes.
- Does not support reductions besides sum.
- Does not support complex operations on single tensors, like complex reshaping.

## Einops

Like einsum, but better:

- **reduce** (removing dimensions)
- **repeat** (adding dimensions)
- **rearrange** (keeping the same number of elements)
- A Rule: axes' ids should be unique

**Some Examples of einops patterns**:

```Python
'b c h w -> b h w c' # transpose
'b c h w -> b c' # reduce on h w
'(h1 h2) (w1 w2) c -> (h1 w1) h2 w2 c' # splits the image into patches and stacks them
```

**Note**: Notice that having unique ids for each axis in the pattern makes it impossible to have traces.

#### Composition and Decomposition of axes (denoted by parenthesis)

- Uses the C-ordering convention

	- If `x` has shape `[A, B, C]` and `y = rearrange(x, 'x y z' -> (x y z))`, then `x[a, b, c] == y[a * B * C + b * C + c]`.
		- For example: if `x` has shape `(10, 10, 10)`, then `x[6, 2, 4] == y[624]`
- As for decomposition, same bijection applies.

#### Axes referred by their size (anonymous axes and unitary axes)
- Anonymous axes
	- same as named axes but can't be matched across different tensors
- Unitary axes (size = 1)
	- they are special, as they don't correspond to an axis variable
	
```Python
'h w -> h w 1' # add unitary axis
'h w -> h w 3' # repeat values along new anonymous axis of size 3
'1 h w 3 -> h w' # remove unitary axis and reduce on anonymous axis of length 3
'... h w -> ... w h' # by using ellipsis, we transpose the last two dimensions
'b ... c -> (...) b c' # compose all but first and last dimensions and move the resulting new axis to the front
'b c ... -> b c' # reduce on all dimensions except for the first two

```

#### Concrete simple examples

```Python
rearrange(im, '(b1 b2) h w c -> (b1 h) (b2 w) c', b1 = 4, b2 = 4) # organize 16 images into a 4x4 grid

reduce(im, 'b c (h h2) (w w2) -> b c h w', 'max', h2 = 2, w2 = 2) # max pooling with kernel = (2, 2)

repeat(im, 'h w c -> (h h2) (w w2) c', h2 = 2, w2 = 2, c = 3) # 2x upsampling of an image by repeating pixels
```


## Einops and Einsum

- einops and einsum work great togther 
- einops supports arbitrary reductions (min,max,sum,mean,...) while einsum only support sum.
- einops allows multi-character names for axes, unlike einsum which allows only one.
- einops supports:
	- composition and decomposition of axes, unitary 
	- unitary axes
	- specification of axes size and verification of shapes and divisibility
	- anonymous axes
	- accepts a list of tensors with the same shapes and dtypes, stacked.
	- layer counterparts, to use in modules like `torch.nn.Sequential`
- einops doesn't support:
	- repeated axes on the left-hand side (used for computing the trace)
- we can use both to support some of the operations `einops` doesn't support. An example, it's a very clean implementation of a Multi-Head Attention module (refer to the paper's appendix).

## Performance

The overhead is negligible (refer to the paper's table for a comparison)
