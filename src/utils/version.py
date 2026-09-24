
"""Log environment versions for reproducibility."""

import json
import platform
from datetime import datetime


def get_versions() -> dict:
    """Collect version information for all key dependencies."""
    versions = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python': platform.python_version(),
    }
    
    # Qiskit ecosystem
    try:
        import qiskit
        versions['qiskit'] = qiskit.__version__
    except ImportError:
        versions['qiskit'] = 'NOT INSTALLED'
    
    try:
        import qiskit_aer
        versions['qiskit_aer'] = qiskit_aer.__version__
    except ImportError:
        versions['qiskit_aer'] = 'NOT INSTALLED'
    
    try:
        import qiskit_machine_learning
        versions['qiskit_machine_learning'] = qiskit_machine_learning.__version__
    except ImportError:
        versions['qiskit_machine_learning'] = 'NOT INSTALLED'
    
    # Classical ML
    try:
        import sklearn
        versions['scikit_learn'] = sklearn.__version__
    except ImportError:
        versions['scikit_learn'] = 'NOT INSTALLED'
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        versions['numpy'] = 'NOT INSTALLED'
    
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except ImportError:
        versions['scipy'] = 'NOT INSTALLED'
    
    try:
        import xgboost
        versions['xgboost'] = xgboost.__version__
    except ImportError:
        versions['xgboost'] = 'NOT INSTALLED'
    
    try:
        import torch
        versions['pytorch'] = torch.__version__
    except ImportError:
        versions['pytorch'] = 'NOT INSTALLED'
    
    # Visualization
    try:
        import matplotlib
        versions['matplotlib'] = matplotlib.__version__
    except ImportError:
        versions['matplotlib'] = 'NOT INSTALLED'
    
    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except ImportError:
        versions['pandas'] = 'NOT INSTALLED'
    
    return versions


def save_versions(path: str = 'results/versions.json') -> None:
    """Save version information to a JSON file."""
    versions = get_versions()
    with open(path, 'w') as f:
        json.dump(versions, f, indent=2)
    return versions
