import os
import pandas as pd
import os.path as osp
import argparse
import numpy as np


 
if __name__ == '__main__': 
    os.chdir("/results")
    repetitions = 10
    
    for mod in ["gat","sage","gcn"]
        for el in ["val","test"]:
            ri = 0
            df = pd.read_csv("logw_"+el+"_"+mod+"_"+str(ri)+"_photo.csv")
            for i in range(1,repetitions):
                try:
                    df_t = pd.read_csv("logw_"+el+"_"+mod+"_"+str(ri)+"_photo.csv")
                    df = df + df_t
                except:
                    continue
            df = df/(i+1)
            df["perc"]= df["perc"]+2
            print(df.columns)
            
            n1 = el+"_r"
            n2 = el+"_br"
            n3 = el+"_e"
            n4 = el+"_be"
            reg = df[n1] - df[n2]
            ex = df[n3] - df[n4]
            df["diff_r"] = reg
            df["diff_e"] = ex
            df["sum_diff"] = ex+reg
            df.to_csv("../res_sage_"+el+"_photo.csv",index=False)
  
