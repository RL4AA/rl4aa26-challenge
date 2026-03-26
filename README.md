[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/1172762585.svg)](https://doi.org/10.5281/zenodo.18965252)


# Reinforcement Learning Challenge at the RL4AA'26 Workshop
This repository contains the material for challenge of the [RL4AA'26](https://indico.ph.liv.ac.uk/event/2025/) workshop.

The workshop is going to be held from the 30th of March to the 1st of April 2026 in Liverpool. The **final submission for competitors in the challenge will be on Wednesday the 1st of April, 14:15.** The teams with first and second place submissions will be asked to briefly present their results and methods to the workshop (template slides available on [indico](https://indico.ph.liv.ac.uk/event/2025/sessions/1451/#20260330)), before being awarded their prize. Only attendees of the workshop are eligible to compete.

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
This tutorial is uploaded to [Zenodo](https://zenodo.org/records/18982468).
Please use the following DOI when citing this code:

```bibtex
@software{santamaria_garcia_2026_18982468,
  author       = {Santamaria Garcia, Andrea and
                  Wulff, Joel and
                  Pollard, Amelia and
                  Kaiser, Jan and
                  Xu, Chenran},
  title        = {RL4AA'26 CLARA Challenge},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.18982468},
  url          = {https://doi.org/10.5281/zenodo.18982468},
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
