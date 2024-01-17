conda create --name DynGL python=3.8
conda activate DynGL

# conda update -n base conda
# conda install -n base conda-libmamba-solver
# conda config --set solver libmamba

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch-geometric
pip install scipy==1.9.3
pip install dataclasses_json
pip install numba==0.56.4
pip install overrides==7.3.1
pip install matplotlib
pip install ogb==1.3.5
pip install torchsummary