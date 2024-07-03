Create a virtual environment :
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies in the virtual environment :

`pip install -r requirements.txt`

Run the script:
```bash
python obstacle.py 
```

The constants.py file contains the geometry parameters for the obstacle ellipsoid. The actual orientation is chosen to have an obstacle that concretize into a real compact obstacle, easy to be 3D-printed.