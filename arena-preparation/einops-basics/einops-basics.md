# Einops tutorial, part 1: basics

<!-- <img src='http://arogozhnikov.github.io/images/einops/einops_logo_350x350.png' height="80" /> -->

## Welcome to einops-land!

We don't write 
```python
y = x.transpose(0, 2, 3, 1)
```
We write comprehensible code
```python
y = rearrange(x, 'b c h w -> b h w c')
```


`einops` supports widely used tensor packages (such as `numpy`, `pytorch`, `jax`, `tensorflow`), and extends them.

## What's in this tutorial?

- fundamentals: reordering, composition and decomposition of axes
- operations: `rearrange`, `reduce`, `repeat`
- how much you can do with a single operation!


> Skeleton from Alex Rogozhnikov's *Einops tutorial, part 1: basics* (MIT licence, https://einops.rocks/1-einops-basics/).
>
> I'm retyping the commands by hand from the original tutorial to increase immersion 😎

## Preparations


```python
# we need some libraries for this demo
import sys
!uv pip install --python {sys.executable} einops numpy pillow
```

    [2mUsing Python 3.13.5 environment at: /Users/nathan/Projects/BottomUpLabs/notebooks/.venv[0m
    [2mChecked [1m3 packages[0m [2min 11ms[0m[0m



```python
# Examples are given for numpy. This code also setups ipython/jupyter/jupyterlite
# so that numpy arrays in the output are displayed as images
import numpy
from utils import display_np_arrays_as_images

display_np_arrays_as_images()
```

## Load a batch of images to play with


```python
ims = numpy.load("./resources/test_images.npy", allow_pickle=False)
# There are 6 images of shape 96x96 with 3 color channels packed into tensor
print(ims.shape, ims.dtype)
```

    (6, 96, 96, 3) float64



```python
# display the first image (whole 4d tensor can't be rendered)
print(ims[0].shape)
ims[0]
```

    (96, 96, 3)





    
![png](einops-basics_files/einops-basics_7_1.png)
    




```python
# second image in a batch
ims[1]
```




    
![png](einops-basics_files/einops-basics_8_0.png)
    




```python
# we'll use three operations
from einops import rearrange, reduce, repeat
```


```python
# rearrange, as the name suggests, rearranges elements
# below we swapped height and width.
# In other words, transposed first two axes (dimensions)
print(rearrange(ims[0], "h w c -> w h c").shape)
rearrange(ims[0], "h w c -> w h c")
```

    (96, 96, 3)





    
![png](einops-basics_files/einops-basics_10_1.png)
    




```python
# we could use more verbose names for axes, and result is the same:
rearrange(ims[0], "height width color -> width height color")
# when you operate on same set of axes many times,
# you usually come up with short names.
# That's what we do throughout tutorial - we'll use b (for batch), h, w, and c
```




    
![png](einops-basics_files/einops-basics_11_0.png)
    



## Composition of axes
transposition is very common and useful, but let's move to other capabilities provided by einops


```python
# einops allows seamlessly composing batch and height to a new height dimension
# We just rendered all images by collapsing to 3d tensor!
print(rearrange(ims, "b h w c -> (b h) w c").shape)
# height becomes 576 (96 * 6), width remains 96, color remains a triple
rearrange(ims, "b h w c -> (b h) w c")
```

    (576, 96, 3)





    
![png](einops-basics_files/einops-basics_13_1.png)
    




```python
# or compose a new dimension of batch and width
print(rearrange(ims, "b h w c -> h (b w) c").shape)
# height remains 96, width becomes 576 (96 * 6), color remains a triple
rearrange(ims, "b h w c -> h (b w) c")
```

    (96, 576, 3)





    
![png](einops-basics_files/einops-basics_14_1.png)
    




```python
# resulting dimensions are computed very simply
# length of newly composed axis is a product of components
# [6, 96, 96, 3] -> [96, (6 * 96), 3]
rearrange(ims, "b h w c -> h (b w) c").shape
```




    (96, 576, 3)




```python
# we can compose more than two axes.
# let's flatten 4d array into 1d, resulting array has as many elements as the original
rearrange(ims, "b h w c -> (b h w c)").shape
```




    (165888,)



## Decomposition of axis


```python
# decomposition is the inverse process - represent an axis as a combination of new axes
# several decompositions possible, so b1=2 is to decompose 6 to b1=2 and b2=3
rearrange(ims, "(b1 b2) h w c -> b1 b2 h w c ", b1=2).shape
```




    (2, 3, 96, 96, 3)




```python
# finally, combine composition and decomposition:
print(rearrange(ims, "(b1 b2) h w c -> (b1 h) (b2 w) c ", b1=2).shape)
# batch splits to b1=2, b2=3; then (b1 h) composes 2 rows [height] and (b2 w) composes three columns [width].
rearrange(ims, "(b1 b2) h w c -> (b1 h) (b2 w) c ", b1=2)
```

    (192, 288, 3)





    
![png](einops-basics_files/einops-basics_19_1.png)
    




```python
# slightly different composition: b1 is merged with width, b2 with height
# ... so letters are ordered by w then by h
print(rearrange(ims, "(b1 b2) h w c -> (b2 h) (b1 w) c ", b1=2).shape)
# batch is split into b1=2 and b2=3; then (b2 h) composes three rows [height] and (b1 w) composes two columns [width].
rearrange(ims, "(b1 b2) h w c -> (b2 h) (b1 w) c ", b1=2)
```

    (288, 192, 3)





    
![png](einops-basics_files/einops-basics_20_1.png)
    




```python
# move part of width dimension to height.
# we should call this width-to-height as image width shrunk by 2 and height doubled.
# but all pixels are the same!
# Can you write reverse operation (height-to-width)?
print(rearrange(ims, "b h (w w2) c -> (h w2) (b w) c", w2=2).shape)
# factor width into two axes, add the second to height
rearrange(ims, "b h (w w2) c -> (h w2) (b w) c", w2=2)
```

    (192, 288, 3)





    
![png](einops-basics_files/einops-basics_21_1.png)
    




```python
# Reverse operation (height-to-width):
print(rearrange(ims, "b (h h2) w c -> h (b w h2) c", h2=2).shape)
# factor height into two axes, add the second to the batched width
rearrange(ims, "b (h h2) w c -> h (b w h2) c", h2=2)
```

    (48, 1152, 3)





    
![png](einops-basics_files/einops-basics_22_1.png)
    



## Order of axes matters


```python
# compare with the next example
rearrange(ims, "b h w c -> h (b w) c")
```




    
![png](einops-basics_files/einops-basics_24_0.png)
    




```python
# order of axes in composition is different
# rule is just as for digits in the number: leftmost digit is the most significant,
# while neighboring numbers differ in the rightmost axis.

# you can also think of this as lexicographic sort
rearrange(ims, "b h w c -> h (w b) c")
```




    
![png](einops-basics_files/einops-basics_25_0.png)
    




```python
rearrange(ims, "b h (w w2) c -> (h w2) (w b) c", w2=2)
```




    
![png](einops-basics_files/einops-basics_26_0.png)
    




```python
# what if b1 and b2 are reordered before composing to width?
rearrange(ims, "(b1 b2) h w c -> h (b1 b2 w) c ", b1=2)  # produces 'einops'
rearrange(ims, "(b1 b2) h w c -> h (b2 b1 w) c ", b1=2)  # produces 'eoipns'
```




    
![png](einops-basics_files/einops-basics_27_0.png)
    



## Meet einops.reduce

In einops-land you don't need to guess what happened
```python
x.mean(-1)
```
Because you write what the operation does
```python
reduce(x, 'b h w c -> b h w', 'mean')
```

if axis is not present in the output — you guessed it — axis was reduced.


```python
# average over batch
reduce(ims, "b h w c -> h w c", "mean")
```




    
![png](einops-basics_files/einops-basics_29_0.png)
    




```python
# the previous is identical to familiar:
ims.mean(axis=0)
# but is so much more readable
```




    
![png](einops-basics_files/einops-basics_30_0.png)
    




```python
# Example of reducing of several axes
# besides mean, there are also min, max, sum, prod
reduce(ims, "b h w c -> h w", "min")
```




    
![png](einops-basics_files/einops-basics_31_0.png)
    




```python
# max
reduce(ims, "b h w c -> h w", "max")
```




    
![png](einops-basics_files/einops-basics_32_0.png)
    




```python
# this is mean-pooling with 2x2 kernel
# image is split into 2x2 patches, each patch is averaged
reduce(ims, "b (h h2) (w w2) c -> h (b w) c", "mean", h2=2, w2=2)
```




    
![png](einops-basics_files/einops-basics_33_0.png)
    




```python
# this is mean-pooling with 4x4 kernel
# image is split into 4x4 patches, each patch is averaged
reduce(ims, "b (h h2) (w w2) c -> h (b w) c", "mean", h2=4, w2=4)
```




    
![png](einops-basics_files/einops-basics_34_0.png)
    




```python
# max-pooling is similar
# result is not as smooth as for mean-pooling
reduce(ims, "b (h h2) (w w2) c -> h (b w) c", "max", h2=2, w2=2)
```




    
![png](einops-basics_files/einops-basics_35_0.png)
    




```python
# yet another example. Can you compute result shape?
# b1=2, therefore b2=3; h -> (b2 h) = (3 * 96) = 288; w -> (b1 w) = (2 * 96) = 192
print(reduce(ims, "(b1 b2) h w c -> (b2 h) (b1 w)", "mean", b1=2).shape)
# booya! 🙂
reduce(ims, "(b1 b2) h w c -> (b2 h) (b1 w)", "mean", b1=2)
```

    (288, 192)





    
![png](einops-basics_files/einops-basics_36_1.png)
    



## Stack and concatenate


```python
# rearrange can also take care of lists of arrays with the same shape
x = list(ims)
print(type(x), "with", len(x), "tensors of shape", x[0].shape)
# that's how we can stack inputs
# "list axis" becomes first ("b" in this case), and we left it there
rearrange(x, "b h w c -> b h w c").shape
```

    <class 'list'> with 6 tensors of shape (96, 96, 3)





    (6, 96, 96, 3)




```python
# but new axis can appear in the other place:
rearrange(x, "b h w c -> h w c b").shape
```




    (96, 96, 3, 6)




```python
# that's equivalent to numpy stacking, but written more explicitly
numpy.array_equal(rearrange(x, "b h w c -> h w c b"), numpy.stack(x, axis=3))
```




    True




```python

```


```python

```


```python
# ... or we can concatenate along axes
rearrange(x, "b h w c -> h (b w) c").shape
```




    (96, 576, 3)




```python
# which is equivalent to concatenation
numpy.array_equal(rearrange(x, "b h w c -> h (b w) c"), numpy.concatenate(x, axis=1))
```




    True



## Addition or removal of axes

You can write 1 to create a new axis of length 1. Similarly you can remove such axis.

There is also a synonym `()` that you can use. That's a composition of zero axes and it also has a unit length.


```python
x = rearrange(ims, "b h w c -> b 1 h w 1 c")  # functionality of numpy.expand_dims
print(x.shape)
print(rearrange(x, "b 1 h w 1 c -> b h w c").shape)  # functionality of numpy.squeeze
```

    (6, 1, 96, 96, 1, 3)
    (6, 96, 96, 3)



```python
# compute max in each image individually, then show a difference
x = reduce(ims, "b h w c -> b () () c", "max") - ims
rearrange(x, "b h w c -> h (b w) c")
```




    
![png](einops-basics_files/einops-basics_47_0.png)
    




```python
# compute min in each image individually, then show a difference
x = ims - reduce(ims, "b h w c -> b () () c", "min") 
rearrange(x, "b h w c -> w (b h) c")
```




    
![png](einops-basics_files/einops-basics_48_0.png)
    



## Repeating elements

Third operation we introduce is `repeat`


```python
# repeat along a new axis. New axis can be placed anywhere
repeat(ims[0], "h w c -> h new_axis w c", new_axis=5).shape
```




    (96, 5, 96, 3)




```python
# shortcut
repeat(ims[0], "h w c -> h 5 w c").shape
```




    (96, 5, 96, 3)




```python
# repeat along w (existing axis)
repeat(ims[0], "h w c -> h (repeat w) c", repeat=3)
```




    
![png](einops-basics_files/einops-basics_52_0.png)
    




```python
# repeat along two existing axes
repeat(ims[0], "h w c -> (2 h) (2 w) c")
```




    
![png](einops-basics_files/einops-basics_53_0.png)
    




```python
# order of axes matters as usual - you can repeat each element (pixel) 3 times
# by changing order in parenthesis
repeat(ims[0], "h w c -> h (w repeat) c", repeat=3)
```




    
![png](einops-basics_files/einops-basics_54_0.png)
    



Note: `repeat` operation covers functionality identical to `numpy.repeat`, `numpy.tile` and actually more than that.

## Reduce ⇆ repeat

reduce and repeat are like opposite of each other: first one reduces amount of elements, second one increases.

In the following example each image is repeated first, then we reduce over new axis to get back original tensor. Notice that operation patterns are "reverse" of each other


```python
repeated = repeat(ims, "b h w c -> b h new_axis w c", new_axis=2)
reduced = reduce(repeated, "b h new_axis w c -> b h w c", "min")
assert numpy.array_equal(ims, reduced)
```

## Fancy examples in random order

(a.k.a. mad designer gallery)


```python
# interweaving pixels of different pictures
# all letters are observable
rearrange(ims, "(b1 b2) h w c -> (h b1) (w b2) c ", b1=2)
```




    
![png](einops-basics_files/einops-basics_59_0.png)
    




```python
# interweaving along vertical for couples of images
rearrange(ims, "(b1 b2) h w c -> (h b1) (b2 w) c", b1=2)
```




    
![png](einops-basics_files/einops-basics_60_0.png)
    




```python
# compute max in each image individually, then show a difference
x = reduce(ims, "b h w c -> b () () c", "max") - ims
reduce(x, "b2 (h 2) w c -> h (b2 w) c", "mean")
```




    
![png](einops-basics_files/einops-basics_61_0.png)
    




```python
# interweaving lines for couples of images
# exercise: achieve the same result without einops in your favourite framework
reduce(ims, "(b1 b2) h w c -> h (b2 w) c", "max", b1=2)
```




    
![png](einops-basics_files/einops-basics_62_0.png)
    




```python
# color can be also composed into dimension
# ... while image is downsampled
reduce(ims, "b (h 2) (w 2) c -> (c h) (b w)", "mean")
```




    
![png](einops-basics_files/einops-basics_63_0.png)
    




```python
# ... or not downsampled
reduce(ims, "b h w c -> (c h) (b w)", "mean")
```




    
![png](einops-basics_files/einops-basics_64_0.png)
    




```python
# disproportionate resize
reduce(ims, "b (h 4) (w 3) c -> (h) (b w)", "mean")
```




    
![png](einops-basics_files/einops-basics_65_0.png)
    




```python
# spilt each image in two halves, compute mean of the two
reduce(ims, "b (h1 h2) w c -> h2 (b w)", "mean", h1=2)
```




    
![png](einops-basics_files/einops-basics_66_0.png)
    




```python
# doot-doo
reduce(ims, "(b1 b2) (h1 h2) (w1 w2) c -> (h2 b1 h1) (w2 b2 w1)", "mean", h1=2, w1=2, b1=2)
```




    
![png](einops-basics_files/einops-basics_67_0.png)
    




```python
# split in small patches and transpose each patch
rearrange(ims, "b (h1 h2) (w1 w2) c -> (h1 w2) (b w1 h2) c", h2=8, w2=8)
```




    
![png](einops-basics_files/einops-basics_68_0.png)
    




```python
# split in larger patches and transpose each patch
rearrange(ims, "b (h1 h2) (w1 w2) c -> (h1 w2) (b w1 h2) c", h2=16, w2=16)
```




    
![png](einops-basics_files/einops-basics_69_0.png)
    




```python
# stop me someone!
rearrange(ims, "b (h1 h2 h3) (w1 w2 w3) c -> (h1 w2 h3) (b w1 h2 w3) c", h2=2, w2=2, w3=2, h3=2)
```




    
![png](einops-basics_files/einops-basics_70_0.png)
    




```python
rearrange(ims, "(b1 b2) (h1 h2) (w1 w2) c -> (h1 b1 h2) (w1 b2 w2) c", h1=3, w1=3, b2=3)
```




    
![png](einops-basics_files/einops-basics_71_0.png)
    




```python
# patterns can be arbitrarily complicated
reduce(ims, "(b1 b2) (h1 h2 h3) (w1 w2 w3) c -> (h1 w1 h3) (b1 w2 h2 w3 b2) c", "mean", h2=2, w1=2, w3=2, h3=2, b2=2)
```




    
![png](einops-basics_files/einops-basics_72_0.png)
    




```python
# subtract background in each image individually and normalize
# pay attention to () - this is composition of 0 axis, a dummy axis with 1 element.
im2 = reduce(ims, "b h w c -> b () () c", "max") - ims
im2 /= reduce(im2, "b h w c -> b () () c", "max")
rearrange(im2, "b h w c -> h (b w) c")
```




    
![png](einops-basics_files/einops-basics_73_0.png)
    




```python
# pixelate: first downscale by averaging, then upscale back using the same pattern
averaged = reduce(ims, "b (h h2) (w w2) c -> b h w c", "mean", h2=6, w2=8)
repeat(averaged, "b h w c -> (h h2) (b w w2) c", h2=6, w2=8)
```




    
![png](einops-basics_files/einops-basics_74_0.png)
    




```python
rearrange(ims, "b h w c -> w (b h) c")
```




    
![png](einops-basics_files/einops-basics_75_0.png)
    




```python
# let's bring color dimension as part of horizontal axis
# at the same time horizontal axis is downsampled by 2x
reduce(ims, "b (h h2) (w w2) c -> (h w2) (b w c)", "mean", h2=3, w2=3)
```




    
![png](einops-basics_files/einops-basics_76_0.png)
    



## Ok, numpy is fun, but how do I use einops with some other framework?

If that's what you've done with `ims` being numpy array:
```python
rearrange(ims, 'b h w c -> w (b h) c')
```
That's how you adapt the code for other frameworks:

```python
# pytorch:
rearrange(ims, 'b h w c -> w (b h) c')
# tensorflow:
rearrange(ims, 'b h w c -> w (b h) c')
# gluon:
rearrange(ims, 'b h w c -> w (b h) c')
# cupy:
rearrange(ims, 'b h w c -> w (b h) c')
# jax:
rearrange(ims, 'b h w c -> w (b h) c')
# paddle:
rearrange(ims, 'b h w c -> w (b h) c')

...well, you got the idea.
```

Einops allows backpropagation as if all operations were native to framework.
Operations do not change when moving to another framework - einops notation is universal

# Summary

- `rearrange` doesn't change number of elements and covers different numpy functions (like `transpose`, `reshape`, `stack`, `concatenate`,  `squeeze` and `expand_dims`)
- `reduce` combines same reordering syntax with reductions (`mean`, `min`, `max`, `sum`, `prod`, and any others)
- `repeat` additionally covers repeating and tiling
- composition and decomposition of axes are a corner stone, they can and should be used together



- [Second part of tutorial](https://github.com/arogozhnikov/einops/tree/main/docs) shows how einops works with other frameworks
- [Third part of tutorial](https://arogozhnikov.github.io/einops/pytorch-examples.html) shows how to improve your DL code with einops
