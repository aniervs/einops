Library: https://einops.rocks
Paper: [[1830_einops_clear_and_reliable_tens.pdf]]


## Issues with classical Tensor operations
- 
![[Screenshot 2024-03-11 at 05.04.38.png]]
- ![[Screenshot 2024-03-11 at 05.06.45.png]]

## Related work

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

**Main issue**: Does not support multi-character names for the axes
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

#### Concrete examples

```Python
rearrange(im, '(b1 b2) h w c -> (b1 h) (b2 w) c', b1 = 4, b2 = 4) # organize 16 images into a 4x4 grid

reduce(im, 'b c (h h2) (w w2) -> b c h w', 'max', h2 = 2, w2 = 2) # max pooling with kernel = (2, 2)

repeat(im, 'h w c -> (h h2) (w w2) c', h2 = 2, w2 = 2, c = 3) # 2x upsampling of an image by repeating pixels
```

## Discussion

## Einops and Einsum

## Formal Grammar 

## Usecase example: Multi-head attention

## Performance

## Caching

## Flexibility and applicability

## Adoption