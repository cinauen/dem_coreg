## DEM coregistration
Coregistering of raster DEMs (3D)

### install required modules with
Note:
$CONDA_ENVS is path to folder with conda environment<br>
Can be defined e.g. with (bash)
```
export CONDA_ENVS=/PATH_TO_ENVS/ENV_FOLDER
```

Create environment
```
conda env create --prefix $CONDA_ENVS/py_coreg_v1 -f environment.yml
```

activate env
```
conda activate $CONDA_ENVS/py_coreg_v1
```

update xdem
```
mamba update -c conda-forge xdem
```