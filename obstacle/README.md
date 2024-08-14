## Setup and Usage Instructions

### 1. Create a Virtual Environment

To create a virtual environment, run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Script

Execute the script using the command:

```bash
python obstacle.py
```

### Configuration

The `constants.py` file contains the geometry parameters for the obstacle ellipsoid. The orientation is designed to produce a compact obstacle, making it suitable for 3D printing.
