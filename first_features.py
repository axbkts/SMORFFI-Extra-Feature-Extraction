import os
import math
import ast
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------- Utility Functions -----------------------------
def distance(a: float, b: float, c: float, d: float) -> float:
    return math.hypot(a - c, d - b)

def power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))

def frac_dimension(sample_real: np.ndarray, sample_imag: np.ndarray, tau: int) -> float:
    N = len(sample_real)
    if N < 2 or tau < 2:
        return float("nan")

    m1_end = N - (N % tau) - 2
    if m1_end < 0:
        return float("nan")
    summary_1 = 0.0
    for i in range(0, m1_end + 1):
        summary_1 += distance(sample_real[i], sample_imag[i],
                              sample_real[i + 1], sample_imag[i + 1])

    m_tau_end = (N - (N % tau)) // tau - 2
    if m_tau_end < 0:
        return float("nan")
    summary_tau = 0.0
    for i in range(0, m_tau_end + 1):
        summary_tau += distance(sample_real[i * tau], sample_imag[i * tau],
                                sample_real[(i + 1) * tau], sample_imag[(i + 1) * tau])

    if summary_1 <= 0 or summary_tau <= 0:
        return float("nan")

    dim = 0.5 * (3.0 - ((math.log10(summary_tau) - math.log10(summary_1)) / math.log10(tau)))
    return dim

def parse_complex_list(cell) -> np.ndarray:
    if pd.isna(cell):
        return np.array([], dtype=complex)

    s = str(cell)
    s = re.sub(r'\s+', '', s)
    s = s.replace('i', 'j')

    if not (s.startswith('[') and s.endswith(']')):
        s = f'[{s}]'

    patt = re.compile(
        r'[+-]?'
        r'(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)'
        r'[+-]'
        r'(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)j',
        re.IGNORECASE
    )
    parts = patt.findall(s)
    if parts:
        try:
            return np.array([complex(p) for p in parts], dtype=complex)
        except Exception:
            pass  

    try:
        lst = ast.literal_eval(s)
        return np.array(lst, dtype=complex)
    except Exception:
        return np.array([], dtype=complex)  

# ---------------------- Main Processing -------------------------------
def main():
    N_target = 1000       
    tau = 3               

    # Get script directory dynamically
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(os.getcwd()).resolve()

    # DYNAMIC DIRECTORY: Target 'raw_dataset' folder next to the script
    src_dir = script_dir / 'raw_dataset' 
    # OUTPUT: Root directory where the script is located
    dst_dir = script_dir 

    # Verify 'raw_dataset' exists
    if not src_dir.exists():
        print(f"[ERROR] 'raw_dataset' folder not found. Please check the directory.")
        return

    all_files = sorted([p for p in src_dir.iterdir() if p.is_file() and p.name.endswith('.csv') and not p.name.startswith('.')])
    error_files = []
    ok_files = []
    
    all_processed_dfs = []

    print(f"Total {len(all_files)} CSV files found. Processing started...")

    for src_path in all_files:
        try:
            df = pd.read_csv(src_path)
        except Exception as e:
            print(f"[READ-ERROR] {src_path.name}: {e}")
            error_files.append(src_path.name)
            continue

        columnName = 'preamble'
        if columnName not in df.columns:
            print(f'[ERROR] "preamble" column not found: {src_path.name}')
            error_files.append(src_path.name)
            continue
       
        H = len(df)
        
        packet_status_col = [None] * H 
        
        frac_dim_1_col  = [None] * H
        frac_dim_2_col  = [None] * H
        iqi_1_col       = [None] * H
        iqi_2_col       = [None] * H

        mag_err_1_mean_col = [None] * H
        mag_err_2_mean_col = [None] * H
        phs_err_1_mean_col = [None] * H
        phs_err_2_mean_col = [None] * H

        mag_err_1_var_col  = [None] * H
        mag_err_2_var_col  = [None] * H
        phs_err_1_var_col  = [None] * H
        phs_err_2_var_col  = [None] * H

        short_freq_col = [None] * H
        long_freq_col  = [None] * H
        CFO_col        = [None] * H

        L_long = np.array([
            0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,
            0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0
        ], dtype=float)

        samples_raw = df[columnName].values
        N = min(len(samples_raw), N_target)

        for idx in range(N):
            complexData = parse_complex_list(samples_raw[idx])
            
            # --- CORRUPTED PACKET CHECK ---
            if complexData.size < 288:
                packet_status_col[idx] = f"Bozuk (Boyut: {complexData.size})"
                continue
            
            packet_status_col[idx] = "Geçerli"

            S_short = complexData[0:128]     
            S_long  = complexData[160:288]   

            a_ST_s_sum = 0+0j
            for i in range(0, 112):  
                a_ST_s_sum += np.conj(S_short[i]) * S_short[i + 16]
            a_ST_s = (1.0/16.0) * np.angle(a_ST_s_sum)

            m = np.arange(len(S_long))
            corrected_1 = S_long * np.exp(-1j * m * a_ST_s)

            a_ST_L_sum = 0+0j
            for i in range(0, 64):
                a_ST_L_sum += np.conj(corrected_1[i]) * corrected_1[i + 64]
            a_ST_L = (1.0/64.0) * np.angle(a_ST_L_sum)

            corrected_2 = corrected_1 * np.exp(-1j * m * a_ST_L)

            long1 = corrected_2[0:64]
            long2 = corrected_2[64:128]

            k1 = np.fft.fftshift(np.fft.fft(long1))
            k2 = np.fft.fftshift(np.fft.fft(long2))
            
            EPS = 1e-12
            cNaN = np.nan + 1j * np.nan

            H_est = np.divide(
                0.5 * (k1 + k2),
                L_long,
                out=np.full_like(k1 + k2, cNaN, dtype=np.complex128),
                where=np.isfinite(L_long) & (np.abs(L_long) > EPS)
            )

            X1 = np.fft.fftshift(np.fft.fft(long1))
            X2 = np.fft.fftshift(np.fft.fft(long2))
            
            Y1 = np.divide(
                X1, H_est,
                out=np.full_like(X1, cNaN, dtype=np.complex128),
                where=np.isfinite(H_est) & (np.abs(H_est) > EPS)
            )

            Y2 = np.divide(
                X2, H_est,
                out=np.full_like(X2, cNaN, dtype=np.complex128),
                where=np.isfinite(H_est) & (np.abs(H_est) > EPS)
            )

            mask1 = np.isnan(Y1.real) | np.isnan(Y1.imag)
            mask2 = np.isnan(Y2.real) | np.isnan(Y2.imag)
            Y1[mask1] = 0
            Y2[mask2] = 0
            Y1 = np.where(Y1 == None, 0, Y1)
            Y2 = np.where(Y2 == None, 0, Y2)
            
            Y1 = Y1 - L_long
            Y2 = Y2 - L_long
            
            mag_err_1 = np.abs(L_long) - np.abs(Y1)
            mag_err_2 = np.abs(L_long) - np.abs(Y2)
            phs_err_1 = np.angle(L_long) - np.angle(Y1)
            phs_err_2 = np.angle(L_long) - np.angle(Y2)

            mag_err_1_mean_col[idx] = float(np.mean(mag_err_1))
            mag_err_2_mean_col[idx] = float(np.mean(mag_err_2))
            phs_err_1_mean_col[idx] = float(np.mean(phs_err_1))
            phs_err_2_mean_col[idx] = float(np.mean(phs_err_2))

            mag_err_1_var_col[idx]  = float(np.var(mag_err_1))
            mag_err_2_var_col[idx]  = float(np.var(mag_err_2))
            phs_err_1_var_col[idx]  = float(np.var(phs_err_1))
            phs_err_2_var_col[idx]  = float(np.var(phs_err_2))

            iq_real_1, iq_imag_1 = np.real(Y1), np.imag(Y1)
            iq_real_2, iq_imag_2 = np.real(Y2), np.imag(Y2)

            frac_dim_1_col[idx] = frac_dimension(iq_real_1, iq_imag_1, tau)
            frac_dim_2_col[idx] = frac_dimension(iq_real_2, iq_imag_2, tau)

            iqi_1_col[idx] = power(iq_real_1) / max(power(iq_imag_1), 1e-12)
            iqi_2_col[idx] = power(iq_real_2) / max(power(iq_imag_2), 1e-12)

            short_freq_col[idx] = a_ST_s
            long_freq_col[idx]  = a_ST_L
            CFO_col[idx]        = (a_ST_s + a_ST_L) * 20 * 1_000_000 / (2 * math.pi)

        for old_col in ('short_freq', 'long_freq', 'frequencyOffset', 'frequencyoffset', 'freq_bench'):
            if old_col in df.columns:
                df.drop(columns=[old_col], inplace=True)

        df['Packet_Status']        = packet_status_col
        df['CFO']                  = CFO_col
        df['short_freq']           = short_freq_col
        df['long_freq']            = long_freq_col
        df['frac_dimension_1']     = frac_dim_1_col
        df['frac_dimension_2']     = frac_dim_2_col
        df['iqi_1']                = iqi_1_col
        df['iqi_2']                = iqi_2_col
        df['mag_error_mean_1']     = mag_err_1_mean_col
        df['mag_error_var_1']      = mag_err_1_var_col
        df['mag_error_mean_2']     = mag_err_2_mean_col
        df['mag_error_var_2']      = mag_err_2_var_col
        df['phase_error_mean_1']   = phs_err_1_mean_col
        df['phase_error_var_1']    = phs_err_1_var_col
        df['phase_error_mean_2']   = phs_err_2_mean_col
        df['phase_error_var_2']    = phs_err_2_var_col

        df_processed = df.iloc[:N].copy()
        
        # --- REMOVE PREAMBLE COLUMN ---
        if 'preamble' in df_processed.columns:
            df_processed.drop(columns=['preamble'], inplace=True)
            
        all_processed_dfs.append(df_processed)
        
        ok_files.append(src_path.name)
        print(f"Processed: {src_path.name}")

    print(f"\n[INFO] Processing of individual files completed. Successful: {len(ok_files)}, Failed: {len(error_files)}")
    
    if all_processed_dfs:
        print("\nMerging all data into a single file...")
        final_merged_df = pd.concat(all_processed_dfs, ignore_index=True)
        
        merged_file_path = dst_dir / '123_devices_merged_first_features_clean.csv'
        final_merged_df.to_csv(merged_file_path, index=False, encoding='utf-8')
        print(f"[SUCCESS] Process complete! File saved:\n👉 {merged_file_path}")
    else:
        print("[ERROR] No data found to merge.")

if __name__ == "__main__":
    main()