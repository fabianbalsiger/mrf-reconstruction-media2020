ID_DATA = 'fingerprints'  # the measured fingerprints

ID_MAP_FF = 'FF'  #: fat fraction map
ID_MAP_T1H2O = 'T1H2O'  # T1 H2O (water) map
ID_MAP_T1FAT = 'T1FAT'  # T1 fat map
ID_MAP_DF = 'Df'  # off-resonance frequency map
ID_MAP_B1 = 'B1'  # B1 transmit field efficacy map

ID_MASK_FG = 'mask_fg'  #: The foreground tissue mask (foreground=1, background=0)
ID_MASK_T1H2O = 'mask_t1h2o'  #: The tissue mask to evaluate the T1H2O map (FF < 0.7, background=0)
ID_MASK_ROI = 'mask_roi'  #: The muscle mask with region of interests (ROIs) of the major muscles
ID_MASK_ROI_T1H2O = 'mask_roi_t1h2o'  #: The muscle mask with region of interests (ROIs) of the major muscles. Slice-wise masked for FF > threshold

REGION_LEG = 'LEG'
REGION_THIGH = 'THIGH'

KEY_NORM = 'norm'
