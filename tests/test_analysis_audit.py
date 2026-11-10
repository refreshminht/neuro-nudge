import numpy as np
import pandas as pd
from src.audit_preprocessing import EXPECTED_RUNS


def test_run_partition_mapping():
    assert EXPECTED_RUNS == {3:'train',7:'val',11:'execution_test',4:'imagery_test',8:'imagery_test',12:'imagery_test'}


def test_ptp_flag_definition():
    x=np.zeros((2,64,480),dtype=np.float32)
    x[0,0,0]=250e-6
    x[1,:,::2]=10e-6
    ptp=np.ptp(x,axis=-1)*1e6
    flags=((ptp<0.5)|(ptp>200)).any(axis=1)
    assert np.array_equal(flags,[True,False])
