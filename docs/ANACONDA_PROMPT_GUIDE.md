# Anaconda Prompt guide

From Anaconda Prompt:

```bat
cd C:\Users\DELL\Desktop\Papers\PINN_collection_paper\noharm-piil
conda create -n noharm_piil python=3.11 numpy pandas matplotlib scipy pytest -y
conda activate noharm_piil
pip install -e .
```

Run the smallest 3D Navier--Stokes test:

```bat
python scripts\no_harm_piil_ns3d_industrial.py --mode smoke --out runs\ns3d_smoke
```

Run the quick serious test:

```bat
python scripts\no_harm_piil_ns3d_industrial.py --mode quick --out runs\ns3d_quick
```

Check outputs:

```bat
dir runs\ns3d_quick\tables
dir runs\ns3d_quick\figures
```
