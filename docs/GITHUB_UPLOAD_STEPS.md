# GitHub upload steps

From Anaconda Prompt or Git Bash, inside the repository folder:

```bat
git init
git add .
git commit -m "Initial no-harm PIIL industrial validation release"
git branch -M main
git remote add origin https://github.com/karjxenval/noharm-piil.git
git push -u origin main
```

Before pushing, run:

```bat
python -m pytest
python -m py_compile scripts\no_harm_piil_validation.py scripts\no_harm_piil_sufficiency_map.py scripts\no_harm_piil_ns3d_industrial.py
```
