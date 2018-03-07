## Various memory optimization techniques for computing Sobel

Here is a list of all the optimization strategies I implemented for this assignment.
In order to run different strategies, simply change the variable "strategy" on line
412 to either "naive", "shared", "shared\_overlap", or "unrolled".

##### Naive
This strategy simply took the sequential code and translated it directly to GPU
code. The outer loop represented iterations over the blocks, and the inner loop
represented iterations over the threads.  Needless to say, this strategy is not
ideal, since reused elements are fetched from global memory each time they are
used.

##### Shared memory
This strategy had each block compute Sobel for a 32x32 tile of elements.  Each
thread loaded its corresponding tile element into shared memory.  Additionally,
the threads responsible for the perimeter elements loaded the halo into shared
memory.  The figure below shows which threads were responsible for which elements
(although the tile has been simplified for visualization). After all elements had
been loaded into shared memory, each thread computes Sobel on its respective
element. This strategy has the advantage of using shared memory for reused elements,
but suffers from major control flow problems due to thread divergence during the
loading of halo elements.

##### Shared memory with overlap
This strategy is similar to the previous one, except that the halo is included in
the tile for which the block is responsible.  This means that only the interior
threads compute Sobel for their respective elements, and that the blocks need to
overlap with one another in order to cover all of the elements in the image. This
strategy suffers from less thread divergence than the previous one, but there are
still some idle threads during the Sobel computation.

##### Unrolled
This strategy has each thread compute Sobel for a 4x4 tile.  The thread pulls all
needed elements from the 6x6 tile (including halo) and stores them in registers.
Then, a device function is called for each of the 16 interior elements using the
register elements.  This strategy does not use shared memory, which means that
elements are fetched from global memory multiple times.  This strategy does not
use shared memory, and so there are many repeated global memory accesses for the
halo elements across threads.  The tile size is big enough that the gain from using
registers mitigates this problem.  No doubt using a larger tile size would have an
even greater improvement, but the code becomes much larger for each increase in the
tile size.

## Results
I used the NVIDIA visual profiler to get the kernel times for my different
strategies.  This was done by executing the following commands (the second one is
executed after the first has finished running).

```
$ nvprof --analysis-metrics -o analysis.nvvp ./sobel
$ nvvp analysis.nvvp
```

Here are the results for the larger `mountains.png` image.

| Strategy                   | Time (ms) |
| -------------------------- |:---------:|
| Naive                      | 0.532343  |
| Shared memory              | 0.738956  |
| Shared memory with overlap | 0.456961  |
| Unrolled                   | 0.378846  |
