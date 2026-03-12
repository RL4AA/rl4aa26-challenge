[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/18965253.svg)](https://zenodo.org/badge/latestdoi/18965253)


# Reinforcement Learning Challenge at the RL4AA'26 Workshop
This repository contains the material for challenge of the [RL4AA'26](https://indico.ph.liv.ac.uk/event/2025/) workshop.

Homepage for RL4AA Collaboration: [https://rl4aa.github.io/](https://rl4aa.github.io/)

## 🚨 Installation instructions 🚨
Please check [INSTALL.md](https://github.com/RL4AA/rl4aa26-challenge-private/blob/master/INSTALL.md) and make sure you have gone through the pre-requisites before the challenge starts.

## Disclaimer &#x2757;
This repository is an adapted and updated version of the RL4AA'25 challenge: [Zenodo](https://zenodo.org/doi/10.5281/zenodo.15120236). It builds on the previous work and changes the underlying accelerator simulation from the ARES-EA beam line at DESY, to the CLARA beam line at the Cockcroft Institute.

This repository contains advanced Python tutorials developed with care and dedication to foster learning and collaboration. The code and materials provided here are the result of significant effort, including state-of-the-art research and unpublished or pre-peer-reviewed work.

We share these resources in good faith, aiming to contribute to the community and advance knowledge in our field. If you use or build upon any part of this tutorial, whether in research, software, or educational materials, proper citation is required. Please cite the tutorial as indicated in the repository or its associated Zenodo entry.

While we encourage reuse and adaptation of our work, uncredited use or plagiarism is unacceptable. We actively monitor citations and expect users to engage in responsible scholarly practice. Failure to properly attribute this work may lead to formal actions.

By using this repository, you acknowledge and respect the effort behind it. We appreciate your support in maintaining academic integrity and fostering an open, collaborative environment.

Happy coding, and thank you for citing responsibly! 😊

## Citing the materials
This tutorial is uploaded to [Zenodo](https://doi.org/10.5281/zenodo.18965253).
Please use the following DOI when citing this code:

```bibtex
@software{santamaria_garcia_2026_18965253,
  author       = {Santamaria Garcia, Andrea and
                  Wulff, Joel and
                  Pollard, Amelia and
                  Kaiser, Jan and
                  Xu, Chenran},
  title        = {RL4AA'26 CLARA Challenge},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.18965253},
  url          = {https://doi.org/10.5281/zenodo.18965253},
  swhid        = {swh:1:dir:b0810d3ca33141c6327c9dfde9ca6dc9edc9b19e
                   ;origin=https://doi.org/10.5281/zenodo.18965252;vi
                   sit=swh:1:snp:92526b6caed578d5e100b96dc9cdbae61a05
                   dce4;anchor=swh:1:rel:1aae54d3317dc6d0829d1a2d686a
                   fe7ae4163d03;path=RL4AA-rl4aa26-challenge-2deb378
                  },
}
```

## Folder Structure
<p> This repository contains all the necessary code and configurations for running experiments using reinforcement learning (RL) for the CLARA transverse tuning task. Below is an overview of the directories to help you navigate the code contents:</p>

- `src` Contains the source code for the RL environment and accompanying wrappers/utility functions
  - `src/environments/CLARA` contains the gymnasium environment for the CLARA transverse tuning task
  - `src/wrappers` contains custom wrappers for the CLARA environment
- `data` contains the data from evaluating your agents

## Further Resources

For more examples and details on the RL4AA'25 challenge (the basis of this years formulation) see the papers on applying RL to the ARES-EA facility at DESY:

- [Reinforcement learning-trained optimisers and Bayesian optimisation for online particle accelerator tuning](https://www.nature.com/articles/s41598-024-66263-y)
  - Code repository: <https://github.com/desy-ml/rl-vs-bo>
- [Learning-based Optimisation of Particle Accelerators Under Partial Observability Without Real-World Training](https://proceedings.mlr.press/v162/kaiser22a.html)
