#!/usr/bin/env python3
import h5py

p_tmp_hdf_fn = r"C:\py\lwi_jpm_ras_adcirc_extract_ds_segments_645_psc\Amite_2022.p98.tmp.hdf"
# /twi/work/projects/p00667_louisiana_rtf/ras/sample_data/core_test/Amite_2022_1
with h5py.File(p_tmp_hdf_fn,  "a") as f:
    try:
        del f['Results']
        print( f"\nResults removed from: {p_tmp_hdf_fn}")
    except: 
        print( f"\nNo Results to remove from: {p_tmp_hdf_fn}")