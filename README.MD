# RIDB

A lightweight Python repository for storing and managing a custom refractive index database. 

## Features
- Cus†om refractive index database for deposited and measured with ellipsometry thin films
- Contains methods to search materials, get refractive index, get epsilon, and etc. 
- Contains Lorentz model parameters for FDTD

## Requirements
- Python
- numpy
- yaml

## Installation
Clone the repository and install dependencies:

git clone <REPO_URL>
pip install .

## Usage
    import numpy as np
    import yaml
    import matplotlib.pyplot as plt

    from ridb import Material, RIDB

    ridb = RIDB() #init the DB
    available_mat = ridb.find_materials('formula:TiO2 source:E-beam evaporation', topk=2)
    TiO2 = ridb.get_material(available_mat[0])

    wvl = np.linspace(400, 800, 41) #create array of wvl
    n_TiO2 = TiO2.n(wvl, units='nm) # interpolate complex refractive index

    plt.figure()
    plt.plot(wvl, n_TiO2.real)
    plt.show()


## Contact
Repository owner: Sergei Rodionov (adjust as needed).

For issues and feature requests, use GitHub Issues on the repository page.

---


