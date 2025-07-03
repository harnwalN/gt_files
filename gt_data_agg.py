import cv2, os, glob, math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

class GenotypeAnalyzer:
    def __init__(self, folder_path):
        self.folder = folder_path
        self.male_out = os.path.join(folder_path, '_Output_Males')
        self.female_out = os.path.join(folder_path, '_Output_Females')
        os.makedirs(self.male_out, exist_ok=True)
        os.makedirs(self.female_out, exist_ok=True)

    def load_csv_from_subfolders(self):
        keys = ['genotype', 'position', 'lp', 'mp', 'hp']
        filenames = ['genotype_metadata.csv', 'position_Total_df.csv',
                     'percentage_LP_df.csv', 'percentage_MP_df.csv', 'percentage_HP_df.csv']
        data = {k: [] for k in keys}
        for subfolder in os.listdir(self.folder):
            subfolder_path = os.path.join(self.folder, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for key, file in zip(keys, filenames):
                file_path = os.path.join(subfolder_path, file)
                if os.path.exists(file_path):
                    data[key].append(pd.read_csv(file_path))
        return data['genotype'], data['position'], data['lp'], data['mp'], data['hp']

    def aggregate_genotype_measurement_data(self, geno_list, pos_list, lp_list, mp_list, hp_list):
        def extract_measurement(df_list, gender, rep_label, vial_identifier):
            ts_dict = {}
            for df in df_list:
                subset = df[df['Vial'] == vial_identifier]
                if not subset.empty:
                    return pd.Series(subset["MEAN"].values, index=subset["Seconds"])
            print(f"Warning: {vial_identifier} not found in measurement data. Skipping replicate {rep_label}.")
            return None

        data = {'position': {"M": {}, "F": {}}, 'lp': {"M": {}, "F": {}},
                'mp': {"M": {}, "F": {}}, 'hp': {"M": {}, "F": {}}}
        counters = {"M": {}, "F": {}}
        for geno_df, pos_df, lp_df, mp_df, hp_df in zip(geno_list, pos_list, lp_list, mp_list, hp_list):
            for _, row in geno_df.iterrows():
                gender = row['Gender']
                genotype = row['Genotype']
                vial_id = f"Vial_{row['Vial_Num']}"
                counters[gender][genotype] = counters[gender].get(genotype, 0) + 1
                rep_label = f"{genotype}_rep{counters[gender][genotype]}"
                for key, df in zip(['position', 'lp', 'mp', 'hp'], [pos_df, lp_df, mp_df, hp_df]):
                    subset = df[df['Vial'] == vial_id]
                    if not subset.empty:
                        ts = pd.Series(subset["MEAN"].values, index=subset["Seconds"])
                        data[key][gender][rep_label] = ts
                    else:
                        print(f"Warning: {vial_id} not found in {key} df. Skipping replicate {rep_label}.")

        def build_df(ts_dict):
            df = pd.DataFrame(ts_dict).T
            df = df.reindex(sorted(df.columns, key=lambda x: float(x)), axis=1)
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Unnamed: 0"}, inplace=True)
            df.columns = ["Unnamed: 0"] + [str(col) for col in df.columns if col != "Unnamed: 0"]
            return df

        def compute_velocity(pos_df):
            time_strs = [col for col in pos_df.columns if col != "Unnamed: 0"]
            time_points = np.array([float(t) for t in time_strs])
            rows = []
            for _, row in pos_df.iterrows():
                rep_label = row["Unnamed: 0"]
                pos = row[time_strs].astype(float).values
                vel = np.gradient(pos, time_points)
                rows.append([rep_label] + vel.tolist())
            return pd.DataFrame(rows, columns=pos_df.columns)

        male_pos, female_pos = build_df(data['position']['M']), build_df(data['position']['F'])
        male_lp, female_lp = build_df(data['lp']['M']), build_df(data['lp']['F'])
        male_mp, female_mp = build_df(data['mp']['M']), build_df(data['mp']['F'])
        male_hp, female_hp = build_df(data['hp']['M']), build_df(data['hp']['F'])
        male_vel, female_vel = compute_velocity(male_pos), compute_velocity(female_pos)

        return (male_pos, female_pos, male_vel, female_vel,
                male_lp, female_lp, male_mp, female_mp, male_hp, female_hp)

    def create_plot_aggregated_df(self, df, gender, output_folder, measurement):
        time_cols = df.columns[1:]
        grouped = {}
        for _, row in df.iterrows():
            genotype = row["Unnamed: 0"].rsplit("_rep", 1)[0]
            grouped.setdefault(genotype, []).append(row[time_cols].astype(float))

        agg_df = pd.DataFrame(index=[float(t) for t in time_cols])
        for genotype, reps in grouped.items():
            reps_df = pd.DataFrame(reps)
            mean, sem, count = reps_df.mean(), reps_df.std(ddof=1) / np.sqrt(len(reps)), reps_df.count()
            agg_df[f"{genotype}_mean"] = mean.values
            agg_df[f"{genotype}_sem"] = sem.values
            agg_df[f"{genotype}_N"] = count.values

        agg_df.index.name = "Time (seconds)"
        agg_df.to_csv(os.path.join(output_folder, f"{gender.lower()}_agg_{measurement}_data.csv"), index=False)

        plt.figure(figsize=(10, 6))
        for col in [c for c in agg_df.columns if c.endswith("_mean")]:
            genotype = col[:-5]
            plt.errorbar(agg_df.index, agg_df[col], yerr=np.nan_to_num(agg_df[f"{genotype}_sem"]),
                         capsize=3, label=f"{genotype} (N={int(agg_df[f'{genotype}_N'].iloc[0])})",
                         marker='o', linestyle='-')

        plt.axhline(0, color='black', linestyle='--') if measurement == "velocity" else None
        plt.xlabel("Time (seconds)")
        ylabel = {"velocity": "Velocity (cm/sec)", "position": "Position (cm)"}.get(measurement, f"{measurement.capitalize()} %")
        plt.ylabel(ylabel)
        plt.title(f"{gender.capitalize()} Genotype Aggregated {measurement.capitalize()} Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{gender.lower()}_{measurement}_plot.png"))
        plt.close()

    def run(self):
        geno_list, pos_list, lp_list, mp_list, hp_list = self.load_csv_from_subfolders()
        results = self.aggregate_genotype_measurement_data(geno_list, pos_list, lp_list, mp_list, hp_list)
    
        genders = ['male', 'female']
        measurements = ['position', 'velocity', 'low-performer', 'middle-performer', 'high-performer']
        output_folders = [self.male_out, self.female_out]
    
        data_mapping = {
            'position': results[0:2],
            'velocity': results[2:4],
            'low-performer': results[4:6],
            'middle-performer': results[6:8],
            'high-performer': results[8:10]
        }
    
        filename_suffix = {
            'position': 'pos_data.csv',
            'velocity': 'vel_data.csv',
            'low-performer': 'lp_perc_data.csv',
            'middle-performer': 'mp_perc_data.csv',
            'high-performer': 'hp_perc_data.csv'
        }
    
        for i, gender in enumerate(genders):
            for measurement in measurements:
                df = data_mapping[measurement][i]
                out_dir = output_folders[i]
                df.to_csv(os.path.join(out_dir, f"{gender}_stats_{filename_suffix[measurement]}"), index=False)
                self.create_plot_aggregated_df(df, gender, out_dir, measurement)