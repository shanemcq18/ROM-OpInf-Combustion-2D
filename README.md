[![License](https://img.shields.io/github/license/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/Willcox-Research-Group/ROM-OpInf-Combustion-2D)
[![Latest commit](https://img.shields.io/github/last-commit/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/commits/master)
[![Documentation](https://img.shields.io/badge/Documentation-WIKI-important)](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/wiki)
<!-- [![Issues](https://img.shields.io/github/issues/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/issues) -->

# Reduced-order Modeling via Operator Inference for 2D Combustion

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1TbIfBQW-YYXVydBFC0McJaSFjkvaeLvA" width="700">
</p>

This repository is an extensive example of **Operator Inference**, a data-driven procedure for reduced-order modeling, applied to a two-dimensional single-injector combustion problem.
It is the source code for the following publications (see [**References**](#references) below).
- [**`jrsnz2021`**](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/tree/jrsnz2021) has the source code of the paper [_Data-driven reduced-order models via regularised operator inference for a single-injector combustion process_](https://www.tandfonline.com/doi/full/10.1080/03036758.2020.1863237) by McQuarrie, Huang, and Willcox.
- [**`aiaape2021`**](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/tree/aiaape2021) has the source code for the upcoming paper _Performance comparison of data-driven reduced models for a single-injector combustion process_ by Jain, Kramer, and McQuarrie.
- The code can also replicate the results of the paper [_Learning physics-based reduced-order models for a single-injector combustion process_](https://arc.aiaa.org/doi/10.2514/1.J058943) by Swischuk, Kramer, Huang, and Willcox.

[**See the Wiki for details on the problem statement, instructions for using this repository, and visual results.**](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/wiki)

---

**Contributors**:
[Shane A. McQuarrie](https://github.com/shanemcq18),
[Renee Swischuk](https://github.com/swischuk),
[Parikshit Jain](https://github.com/PARIKSHITJAIN2102),
[Boris Kramer](http://kramer.ucsd.edu/),
[Karen Willcox](https://kiwi.oden.utexas.edu/)

## References

- [McQuarrie, S. A.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [**Data-driven reduced-order models via regularised operator inference for a single-injector combustion process**](https://www.tandfonline.com/doi/full/10.1080/03036758.2020.1863237).
_Journal of the Royal Society of New Zealand_, Vol. 51:2, pp. 194-211, 2021.
([Download](https://kiwi.oden.utexas.edu/papers/nonlinear-non-intrusive-model-reduction-combustion-McQuarrie-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{MHW2021regOpInfCombustion,
author    = {Shane A. McQuarrie and Cheng Huang and Karen E. Willcox},
title     = {Data-driven reduced-order models via regularised Operator Inference for a single-injector combustion process},
journal   = {Journal of the Royal Society of New Zealand},
volume    = {51},
number    = {2},
pages     = {194--211},
year      = {2021},
publisher = {Taylor & Francis},
}}</pre></details>

- [Jain, P.](https://www.linkedin.com/in/parikshit-jain-6b870961/), [Kramer, B.](http://kramer.ucsd.edu/), and [McQuarrie, S. A.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), **Performance comparison of data-driven reduced models for a single-injector combustion process**. _AIAA Propulsion and Energy Forum and Exposition_, 2021, to appear.

- [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [**Learning physics-based reduced-order models for a single-injector combustion process**](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, Vol. 58:6, pp. 2658-2672, 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13.
([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2020romCombustion,
    title     = {Learning physics-based reduced-order models for a single-injector combustion process},
    author    = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal   = {AIAA Journal},
    volume    = {58},
    number    = {6},
    pages     = {2658--2672},
    year      = {2020},
    publisher = {American Institute of Aeronautics and Astronautics}
}</pre></details>

- [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ) (2020). [**[Updated] 2D Benchmark Reacting Flow Dataset for Reduced Order Modeling Exploration \[Data set\]**](https://doi.org/10.7302/nj7w-j319). University of Michigan - Deep Blue. https://doi.org/10.7302/nj7w-j319.

- [**ROM Operator Inference Python 3 package**](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) ([pypi](https://pypi.org/project/rom-operator-inference/))
