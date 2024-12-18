# Installing

You will need the [Anaconda Python distribution](https://www.anaconda.com/products/individual). On many systems that is already installed: try running `conda --version`.
If that fails, you may need to load an anaconda module first: try `module load anaconda` or `module load anaconda3`. If that still does not give you a working `conda` command,
you may want to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Now make sure your conda environment is initialized:

```
conda init bash
```

This needs to be done just once, as it modifies your `.bashrc` that is sourced every time you login.
After this, restart your shell by logging out and back in.

*If `conda init bash` asks you for your password (it tries to do "sudo"), it is likely because the permissions on your `~/.bashrc` file are incorrect.*
In that case, contact your system administrator to have these permissions corrected.
You may in the meantime be able to continue these instructions by executing `eval "$(conda shell.bash hook)"` - but this then needs to be done every time your login.

Now obtain the repository with setups and scripts:

```
git clone --recursive git@github.com:inogs/seamless-notebooks.git
cd seamless-notebooks
conda env create -f environment.yml
conda activate seamless-bb
bash ./my_install
```

# Staying up to date

To update this repository *including its submodules (FABM, ERSEM, PISCES, etc.)*, make sure you are in the `seamless-notebooks` directory and execute:

```
conda activate seamless-bb
git pull --recurse-submodules
git submodule update --init --recursive
conda env update -f environment.yml
bash ./my_install
```

The last two commands update the conda (Python) environment and GOTM-FABM, respectively.
Depending on the changes in the repository, they may not be needed, but there is no harm in running them just in case - it just takes a little longer.

# Running a 0-D simulation of FABM-BFM

Go to the setup directory

```
cd setups/0D-ogs
```

Link a configuration file 
```
ln -s ../../extern/ogs/fabm_monospectral_2xDetritus.yaml fabm.yaml
```

Run the simulation
```
python model.py
```

# Running a 1-D simulation of GOTM-FABM-BFM

Go to the setup directory

```
cd setups/setup_BFM1D_BOUSSOLE
```

A configuration file for the BFM of name `fabm.yaml` is needed, several are present in the setup directory and in the code directory `seamless-notebooks/extern/ogs`, e.g. a configuration file with 60 phytoplankton functional types

```
ln -s fabm_EXP21_60PFTs_final.yaml fabm.yaml
```

A configuration file for the turbolence model GOTM of name 'gotm.yaml', several are present in the setup directory, e.g. a configuration with multispectral irradiance and migration of plankton

```
ln -s gotm_multispectral_migration_5y_media_final.yaml gotm.yaml
```

Additional irradiance file are needed, two possibility are given: considering 4 diverse PFT for "light functions": `bcs_4PFTs` or 9 diverse PFT `bcs_9PFTs`

```
ln -s bcs_9PFTs bcs
```

During compilation a launcher of the model was created in  `seamless-notebooks/bin`, we link it in the setup directory

```
ln -s ../../bin/gotm gotm.xx
```

Launch the simulation

```
./gotm.xx --ignore_unknown_config
```

The model output are saved in `result.nc`

# Running a parallel sensitivity analysis

First, initialize your Python environment with:

```
conda activate seamless-bb
```

This needs to be done anytime you want to use parsac; you could add it to your `~/.bashrc`.

An example for a simple sensitivity analysis with GOTM-ERSEM for a Northern North Sea station is provided. To use this, first go to the `seamless-notebooks/parsac` directory.

To sample across parameter space, use:

```
parsac sensitivity sample northsea_sa.xml northsea_sa.pickle saltelli 64
```

To now run GOTM-ERSEM for all sampled parameter sets in parallel, use:

```
sbatch run.sbatch
```

NB this assumes you are on an HPC system that uses the SLURM job scheduler.

The above command submits a parallel job. It will report the identifier of the new job, e.g., "Submitted batch job 435384". Here, 435384 is the new job identifier (referred to as `<JOBID> ` below). You can monitor the job's progress with `squeue -u $USER`. After your job starts (status `ST` is `R`), you can monitor detailed progress with `tail -f <JOBID>.out`.

After the job completes, analyze your results to obtain sensitivity metrics with:

```
parsac sensitivity analyze northsea_sa.pickle sobol
```

To customize this for your own application:

* Most parsac settings are in an xml file. For the ERSEM North Sea example: `northsea_sa.xml`. To customize this for your own application, first create a copy of this file under a new name.
* The path to the GOTM-FABM setup is specified in the xml file with `<setup path="../setups/northsea"/>`. Change this to the directory with the setup you would like to analyze. Before you begin using parsac, verify that GOTM actually works for that setup: enter the setup directory, execute `gotm`, and verify that the simulation completes successfully. Note that parsac will ignore NetCDF files (*.nc) and subdirectories in your setup directory. If these are needed for you simulation, e.g. because gotm should read restart.nc or forcing files in a subdirectory, you can make parsac include them by adding attributes `exclude_files=""` and/or `exclude_dirs=""`: for instance `<setup path="../setups/northsea" exclude_files="" exclude_dirs=""/>`.
* The parameters to perturb are listed under the `<parameters>` section in the xml file. Each entry includes the name where the parameter is set (`fabm.yaml` for biogeochemistry, `gotm.yaml` for hydrodynamics), the full path of the parameter in the file (e.g., `instances/P1/parameters/phim`), and the range of values the parameter can take, specified by its minimum and maximum.
* The target metrics for which sensitivity to parameters is to be assessed are listed under `<targets>`. Note that these metrics must be *scalar* values that computed in some way from the time- and depth-explicit GOTM result, e.g., by slicing at a particular time and depth, by averaging, taking the minimum or maximum, etc. You can add any numnber of targets. For each, you need to specify the path to the NetCDF file where the necessary variables will be found, e.g., `path="result.nc"`. This *must* match the path of one of the NetCDF files written by GOTM-FABM as part of its output (`output` section in `gotm.yaml`).
* The parameter sampling method is *not* set in the xml file, but on the command line as part of the `sample` step. For instance, in the above example `parsac sensitivity sample northsea_sa.xml northsea_sa.pickle saltelli 64` specifies the Saltelli sampling method. To get an overview of available methods, use `parsac sensitivity sample -h`. The methods corresponds to those provided by [SALib](https://salib.readthedocs.io/en/latest/index.html). Most will have additional settings that you can see with `parsac sensitivity sample <XMLFILE> <PICKLEFILE> <METHOD> -h`, for instance, `parsac sensitivity sample northsea_sa.xml northsea_sa.pickle saltelli -h`. For more information about each setting, see [the SALib documentation](https://salib.readthedocs.io/en/latest/api.html).
* Likewise, the analysis method is specified on the command line as part of the `analyze` step. To see available methods, use `parsac sensitivity analyze -h`. As for the `sample` step, an analysis step may have additional arguments that you can see with `parsac sensitivity analyze <PICKLEFILE> <METHOD> -h`. *Note:* in most cases, a particular analysis method needs to be combined with a specific sampling method, e.g., [sampling with saltelli, analysis with sobol](https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis), or [sample with morris, analyze with morris](https://salib.readthedocs.io/en/latest/api.html#method-of-morris). See [the SALib documentation](https://salib.readthedocs.io/en/latest/api.html).

# Running on CINECA

The Anaconda Python distribution is already installed and can be loaded with:

```
module load anaconda3
```

The job scheduler is SLURM, as assumed in the above example.

# Try this online

You can use the code and Jupyter Notebooks in this repository with Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BoldingBruggeman/seamless-notebooks/HEAD?urlpath=lab%2Ftree%2Fsetups)
