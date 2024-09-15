# Parameter identification

This folder contains both methodologies presented in the master thesis. The so-called 'basic experimental methodology' is in folder methodology 1 and the 'optimization-based' methodology is in folder methodology 2.

## Virtual environments

### For Python scripts, create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

The code is written in Python 3.10, other versions of Python may not work.

Install the required dependencies in the virtual environment :

```bash
pip install -r requirements.txt
```

### For Julia scripts

Open your Julia REPL and activate the environment

```bash
]
activate .
```

Then execute scripts with:

```julia
include("path_to_your_script.jl")
```
