
### Synthetic Graph Datasets

#### 3D Surface Dataset
It is a collection of the following 3D functions: **torus**, **elliptic paraboloid**, **saddle**, **ellipsoid**, **elliptic hyperboloid**, **another**.
Where we use different geometric transformations(scale, translate, rotate, reflex, shear) to give variability to the samples.
It is possible to change the number of point by surface, the number or type of functions.

<table>
  <tr>
    <th><img src="imgs/surf_1.png" alt="non-trivial image" width="100%" align="center"></th>
    <th><img src="imgs/surf_2.png" alt="non-trivial image" width="100%" align="center"></th>
    <th><img src="imgs/surf_8.png" alt="non-trivial image" width="100%" align="center"></th>
  </tr>
  <tr>
    <td><img src="imgs/surf_6.png" alt="non-trivial image" width="100%" align="center"></td>
    <td><img src="imgs/surf_9.png" alt="non-trivial image" width="100%" align="center"></td>
    <td><img src="imgs/surf_4.png" alt="non-trivial image" width="100%" align="center"></td>
  </tr>
</table>


The code is show in `graph_surface_dataset.py`. For show the samples run `test_create_surface()` function.
For feed your network use `test_batch_gen()` which show an example for do it using generators by batch.

#### Community Dataset
This dataset presents the following samples: **2-communities** and **4-communities**.
It is possible to change the number of individuals by community.

<table>
  <tr>
    <th><img src="imgs/comm_20.png" alt="non-trivial image" width="100%" align="center"></th>
    <th><img src="imgs/comm_24.png" alt="non-trivial image" width="100%" align="center"></th>
    <th><img src="imgs/comm_28.png" alt="non-trivial image" width="100%" align="center"></th>
  </tr>
  <tr>
    <td><img src="imgs/comm_00.png" alt="non-trivial image" width="100%" align="center"></td>
    <td><img src="imgs/comm_04.png" alt="non-trivial image" width="100%" align="center"></td>
    <td><img src="imgs/comm_08.png" alt="non-trivial image" width="100%" align="center"></td>
  </tr>
</table>

The code is show in `graph_communty_dataset.py`. For show the samples run `test_create_community()` function.
For feed your network use `test_batch_gen()` which show an example for do it using generators by batch.

Note: both datasets created graphs with permutations, besides, the number of permutation by graphs is variable.



#### Geometric Figures

This dataset contain 3 types of geometric figures (square, rectangle and line). Where we use different geometric transformations(scale, translate, rotate).

It is possible to change the image size (H,W) and choose the type of geometric figure that contain each synthetic image.

<table>
  <tr>
    <th><img src="imgs/img_0.png" alt="non-trivial image" width="100%" align="center"></th>
    <th><img src="imgs/img_3.png" alt="non-trivial image" width="100%" align="center">
  </tr>
  <tr>
    <td><img src="imgs/img_4.png" alt="non-trivial image" width="100%" align="center"></td>
    <td><img src="imgs/img_5.png" alt="non-trivial image" width="100%" align="center">
  </td>
  </tr>
</table>