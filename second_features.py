import os
import glob
import pandas as pd
import numpy as np
import re
import scipy.stats
from scipy.spatial.distance import pdist
from tqdm import tqdm
from pathlib import Path

def sampen_fast(L, m=2, r=0.2):
    """Fast Sample Entropy calculation function"""
    N = len(L)
    r_val = r * np.std(L)
    def _phi(m_val):
        x_emb = np.array([L[i:i+m_val] for i in range(N-m_val+1)])
        if len(x_emb) < 2: return 0.0
        dist = pdist(x_emb, 'chebyshev')
        matches = np.sum(dist <= r_val)
        return matches / float(len(dist)) if len(dist) > 0 else 0.0
    
    p_m = _phi(m)
    p_m1 = _phi(m+1)
    if p_m == 0 or p_m1 == 0:
        return 0.0
    return -np.log(p_m1 / p_m)

def hurst_exp(ts):
    """Hurst Exponent calculation function"""
    lags = range(2, 20)
    tau = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        tau.append(np.sqrt(np.mean(diffs**2)))
    if len(tau) < 2: return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def extract_features(row_val):
    """Extracts 14 features from a single raw signal string"""
    s = str(row_val).replace('[', '').replace(']', '').replace('\n', '')
    s = re.sub(r'([+-]?\d+\.?\d*(?:e[+-]\d+)?)[ \t]*([+-])[ \t]*(\d+\.?\d*(?:e[+-]\d+)?j)', r'\1\2\3', s)
    
    try:
        x = np.array([complex(v) for v in s.split()])
    except Exception:
        return pd.Series([np.nan]*14)
        
    I = np.real(x)
    Q = np.imag(x)
    amp = np.abs(x)
    phase = np.angle(x)
    
    skewness_I = scipy.stats.skew(I)
    skewness_Q = scipy.stats.skew(Q)
    kurtosis_I = scipy.stats.kurtosis(I)
    kurtosis_Q = scipy.stats.kurtosis(Q)
    dc_offset_I = np.mean(I)
    dc_offset_Q = np.mean(Q)
    
    var_Q = np.var(Q)
    iq_gain_imbalance = np.sqrt(np.var(I) / var_Q) if var_Q > 0 else 0
    
    std_I, std_Q = np.std(I), np.std(Q)
    iq_phase_imbalance = np.corrcoef(I, Q)[0, 1] if (std_I > 0 and std_Q > 0) else 0
        
    papr = 10 * np.log10(np.max(amp**2) / np.mean(amp**2))
    rms_power = np.sqrt(np.mean(amp**2))
    amplitude_std = np.std(amp)
    cfo_drift_rate = np.mean(np.diff(np.unwrap(phase)))
    
    sample_entropy = sampen_fast(amp)
    hurst_exponent = hurst_exp(amp)
    
    return pd.Series([
        skewness_I, skewness_Q, kurtosis_I, kurtosis_Q, dc_offset_I, dc_offset_Q,
        iq_gain_imbalance, iq_phase_imbalance, papr, rms_power, amplitude_std,
        cfo_drift_rate, sample_entropy, hurst_exponent
    ])

if __name__ == "__main__":
    # DYNAMIC DIRECTORY SETUP
    script_dir = Path(__file__).resolve().parent
    directory = script_dir / "raw_dataset"
    
    # Find all files ending with _pre.csv in the directory
    all_files = glob.glob(os.path.join(directory, "*_pre.csv"))
    
    if not all_files:
        print(f"ERROR: No files found in directory '{directory}'.")
    else:
        print(f"Total {len(all_files)} files found. Starting processing...\n")
        
        # Feature names
        cols = [
            'skewness_I', 'skewness_Q', 'kurtosis_I', 'kurtosis_Q', 'dc_offset_I', 'dc_offset_Q',
            'iq_gain_imbalance', 'iq_phase_imbalance', 'papr', 'rms_power', 'amplitude_std',
            'cfo_drift_rate', 'sample_entropy', 'hurst_exponent'
        ]
        
        all_results = []
        
        # Process files one by one
        for file in tqdm(all_files, desc="Processing files"):
            df = pd.read_csv(file)
            
            if 'preamble' in df.columns:
                tqdm.pandas(desc=f"{os.path.basename(file)}", leave=False)
                features_df = df['preamble'].progress_apply(extract_features)
                features_df.columns = cols
                
                base_cols = []
                if 'Device Number' in df.columns: base_cols.append('Device Number')
                if 'MAC_address' in df.columns: base_cols.append('MAC_address')
                
                result_df = pd.concat([df[base_cols], features_df], axis=1)
                result_df['Source_File'] = os.path.basename(file)
                all_results.append(result_df)
            else:
                print(f"\nWarning: 'preamble' column not found in {os.path.basename(file)}, skipping.")

        # Combine all results into a single DataFrame
        if all_results:
            final_dataset = pd.concat(all_results, ignore_index=True)
            
            # SAVE TO ROOT DIRECTORY
            output_file = script_dir / "RFFI_all_second_extracted_features.csv"
            final_dataset.to_csv(output_file, index=False)
            print(f"\nProcessing successfully completed! All data combined.")
            print(f"File location: {output_file}")
            print(f"Total Row Count: {len(final_dataset)}")
        else:
            print("\nNo data to process.")