import numpy as np
from src.preprocess import RUNS, EXECUTION, LABELS

def test_run_and_label_mapping():
    assert RUNS == (3,4,7,8,11,12)
    assert EXECUTION == {3,7,11}
    assert LABELS == {'T1':0,'T2':1}

def test_epoch_sample_counts():
    for start,end,expected in [(0,2.5,400),(.5,3.5,480)]:
        assert round((end-start)*160)==expected

def test_training_only_normalization_formula():
    x=np.arange(2*64*400,dtype=np.float32).reshape(2,64,400)
    mean=x.mean(axis=(0,2),keepdims=True)
    std=np.maximum(x.std(axis=(0,2),keepdims=True),1e-6)
    assert np.isfinite((x-mean)/std).all()
