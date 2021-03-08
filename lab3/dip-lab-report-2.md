# The Report for Digital Image Processing Laboratory 2

This report is contributed by HUANG Guanchao, SID 11912309, from SME. The complete resources of this laboratory, including source code, figures and report in both `.md` and `.pdf` format can be retrieved at [my GitHub repo](https://github.com/kommunium/dip-lab)

[toc]

---

## Introduction

*Histogram* is a statistical property of a digital image, symbolizing the distributions of pixels of different intensity levels. Histogram is defined as $h(r_k) = n_k$, in which $r_k$ is the $k$th intensity value, $n_k$ is the number of pixels in the image with intensity $r_k$. A generalized concept is *normalized histogram*

One common example is, in digital cameras, system offers histogram view for photographers to obtain a general awareness of whether a photo is overexposed of underexposed, or both. Histogram offers a quantitative rubric for the contrast of an image, namely a well-enhanced image should have evenly distributed histogram in most of the cases, which also provides a methodology for image enhancing.

This lab session is mainly about the utilities of histogram in digital images enhancement. Histogram equalization, histogram matching and local histogram equalization are implemented with `Python` and the results of which are compared.

<!-- TODO further descriptions for various methods -->

This lab session also contains utilizing median smoothing to reduce salt-and-pepper noise, which presents better result than simple average smoothing.

---

## Histogram Equalization

A general way to enhance the contrast of a digital image is histogram equalization, which maps the intensity levels in the raw images to another set of grey scales to form evenly distributed histogram.

### Methodology

The intensity levels in an image may be viewed as random variables in the interval $[0, L-1]$. Let $p_r(r)$ and $p_s(s)$ denote the probability density function (PDF) of random variables $r$ and $s$. Hence, we have the relation

$$
p_s(s) = p_r(r)\left|\frac{dr}{ds}\right|.
$$

Based on which, a transformation function of particular importance in image processing can be formed as

$$
s = T(r) = (L - 1)
$$

![Comparison for Q3_1_1.tif](Q3_1_1.tif_comparison.png)

![Comparison for Q3_1_1.tif](Q3_1_1.tif_histogram.png)

![Comparison for Q3_1_2.tif](Q3_1_2.tif_comparison.png)

![Comparison for Q3_1_2.tif](Q3_1_2.tif_histogram.png)

---

## Histogram Matching

---

## Local Histogram Equalization

---
