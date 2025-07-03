import cv2, os, glob, math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

class FinalizedGeotaxis:
    def __init__(self, experiment, spec_vid, fps=60, top_thresh=0.50, bottom_thresh=0.55, adder_val=150, remove_px = 125):
        self.experiment = experiment
        self.spec_vid = spec_vid
        self.fps = fps
        self.vid_path = f"./{experiment}/{spec_vid}/"
        self.top_thresh=top_thresh
        self.bottom_thresh=bottom_thresh
        self.thresh_step=5
        self.adder_val=adder_val
        self.remove_px = remove_px
        self.frame_data_stored = None
        self.geno_df = None

    def extract_frames(self, vid_path):
        frames=[]
        cap = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:, :, 2].astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def connectComp(self, threshold):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            threshold, 8, cv2.CV_32S
        )
        output = np.zeros(threshold.shape, dtype="uint8")
        centroid_points = []
        areas = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 2 < area < 4000:
                areas.append(area)
                componentMask = (labels == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)
                cX, cY = centroids[i]
                centroid_points.append((cX, cY))
        return output, centroid_points, areas

    def aggregate_data_across_trims(self, all_pos_dfs, all_perc_dfs, geno_df):
        vial_order = [f'Vial_{vial_num}' for vial_num in geno_df['Vial_Num'].values]
        aggregated_pos = {}
        aggregated_perc = {}

        # def process_and_order_data(combined_df, agg_column, col_names):
        #     grouped = combined_df.groupby(['Frame', 'Vial'], as_index=False)
        #     mean_sem = grouped[agg_column].agg(['mean', 'sem']).reset_index()
        #     mean_sem.columns = col_names
        #     mean_sem['Vial'] = pd.Categorical(mean_sem['Vial'], categories=vial_order, ordered=True)
        #     mean_sem = mean_sem.sort_values(['Frame', 'Vial'])
        #     mean_sem['Seconds'] = mean_sem['Frame'] / 60
        #     return mean_sem.reset_index(drop=True)
            
        def process_and_order_data(combined_df, agg_column, col_names):
            # group on Frame & Vial, then compute mean+SEM of our column
            grouped = combined_df.groupby(['Frame', 'Vial'], as_index=False)
            mean_sem = grouped[agg_column].agg(['mean', 'sem']).reset_index()
            # only rename the two aggregated columns
            mean_sem = mean_sem.rename(columns={
                'mean': 'MEAN',
                'sem':  'SEM'
            })
         
            mean_sem['Vial'] = pd.Categorical(
                mean_sem['Vial'],
                categories=vial_order,
                ordered=True
            )
            mean_sem = mean_sem.sort_values(['Frame', 'Vial'])
            mean_sem['Seconds'] = mean_sem['Frame'] / 60
            return mean_sem.reset_index(drop=True)

        pos_dfs = [
            all_pos_dfs[trim]['Total'].assign(Trim=trim) for trim in all_pos_dfs.keys()
        ]
        aggregated_pos['Total'] = process_and_order_data(
            pd.concat(pos_dfs, ignore_index=True),
            'Y_Position', ['Frame', 'Vial', 'MEAN', 'SEM']
        )

        for category in ['LP', 'MP', 'HP']:
            perc_dfs = [
                all_perc_dfs[trim][category].assign(Trim=trim) for trim in all_perc_dfs.keys()
            ]
            aggregated_perc[category] = process_and_order_data(
                pd.concat(perc_dfs, ignore_index=True),
                'Percent', ['Frame', 'Vial', 'MEAN', 'SEM']
            )
        return aggregated_pos, aggregated_perc

    def _load_vial_csvs(self, vial_file_pattern):
        vial_csv_paths = sorted(glob.glob(vial_file_pattern))
        return vial_csv_paths

    def _calc_vial_boundaries(self, vial_df):
        vial_numbers = vial_df['vials'].tolist()
        boundaries = {}
        for _, row in vial_df.iterrows():
            vial_num = int(row['vials'])
            boundaries[vial_num] = {
                'x1': row['x1'],
                'x2': row['x2'],
                'y1': row['y1'],
                'y2': row['y2'],
                'hp_boundary': row['y1'] + ((row['y2'] - row['y1']) / 3.0),
                'mp_boundary': row['y1'] + (2 * ((row['y2'] - row['y1']) / 3.0))
            }
        return boundaries, vial_numbers

    def _initialize_frame_dicts(self, frame_data, vial_key):
        frame_data[vial_key] = {
            'HP': {'x_pos': [], 'y_pos': [], 'count': 0},
            'MP': {'x_pos': [], 'y_pos': [], 'count': 0},
            'LP': {'x_pos': [], 'y_pos': [], 'count': 0},
            'Total': {'x_pos': [], 'y_pos': [], 'count': 0}
        }

    def _compute_zone_points(self, frame_number, points, hp_boundary, mp_boundary):
        hp = [(x, y) for x, y in points if y < hp_boundary]
        mp = [(x, y) for x, y in points if hp_boundary <= y < mp_boundary]
        lp = [(x, y) for x, y in points if y >= mp_boundary]
        return hp, mp, lp
    
    def saving_plot_n_data(self, aggregated_perc, aggregated_pos, frame_step):
        for catg, values in aggregated_perc.items():
            values.to_csv(os.path.join(self.vid_path, f"percentage_{catg}_df.csv"), index=False)
            plt.figure(figsize=(12, 6))
            ax = sns.lineplot(data=values, x='Seconds', y='MEAN', hue='Vial', marker='o', errorbar=None)
        
            handles, labels = ax.get_legend_handles_labels()
            vial_colors = {label: handle.get_color() for handle, label in zip(handles, labels)}
        
            for vial, vial_data in values.groupby('Vial'):
                color = vial_colors.get(vial, 'black')
                ax.fill_between(
                    vial_data['Seconds'],
                    vial_data['MEAN'] - vial_data['SEM'],
                    vial_data['MEAN'] + vial_data['SEM'],
                    alpha=0.2,
                    color=color
                )
        
            plt.title(f"Y Position ({catg}) by Vial Across Selected Frames (Step = {frame_step})")
            plt.xlabel("Seconds")
            plt.ylabel("Y Position")
            plt.legend(title="Vial")
            plt.grid(axis='y', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(self.vid_path, f"percentage_{catg}_plot.png"), bbox_inches="tight")
            plt.close()
    
        for catg, values in aggregated_pos.items():
            values.to_csv(os.path.join(self.vid_path, f"position_{catg}_df.csv"), index=False)
            plt.figure(figsize=(12, 6))
            ax = sns.lineplot(data=values, x='Seconds', y='MEAN', hue='Vial', marker='o', errorbar=None)
        
            handles, labels = ax.get_legend_handles_labels()
            vial_colors = {label: handle.get_color() for handle, label in zip(handles, labels)}
        
            for vial, vial_data in values.groupby('Vial'):
                color = vial_colors.get(vial, 'black')
                ax.fill_between(
                    vial_data['Seconds'],
                    vial_data['MEAN'] - vial_data['SEM'],
                    vial_data['MEAN'] + vial_data['SEM'],
                    alpha=0.2,
                    color=color
                )
        
            plt.title(f"Y Position ({catg}) by Vial Across Selected Frames (Step = {frame_step})")
            plt.xlabel("Seconds")
            plt.ylabel("Y Position")
            plt.legend(title="Vial")
            plt.grid(axis='y', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(self.vid_path, f"position_{catg}_plot.png"), bbox_inches="tight")
            plt.close()
# IMPORTANT ---------
    def run(self):
        if all(os.path.exists(os.path.join(self.vid_path, f)) for f in ['percentage_LP_df.csv','percentage_MP_df.csv','percentage_HP_df.csv','position_Total_df.csv','percentage_LP_plot.png','percentage_MP_plot.png','percentage_HP_plot.png','position_Total_plot.png']): return
        vid_pattern = os.path.join(self.vid_path, f"TRIM_*_{self.spec_vid}.mp4")
        matching_files = sorted(glob.glob(vid_pattern))
        vial_pattern = os.path.join(self.vid_path, f"trim_*_{self.spec_vid}_vials_pos.csv")
        vial_csv_paths = sorted(glob.glob(vial_pattern))

        all_backgrounds = [
            np.max(self.extract_frames(vid_path), axis=0)
            for vid_path in tqdm(matching_files, desc="Computing backgrounds")
        ]

        self.geno_df = pd.read_csv(os.path.join(self.vid_path, "genotype_metadata.csv"))
        print(f"geno_df: \n{self.geno_df}\n")

        self.frame_data_stored = {}
        self.frame_percent_stored = {}

        for trim_idx, (vid_path, csv_path) in tqdm(enumerate(zip(matching_files, vial_csv_paths), start=1)):
            trim_key = f"trim_{trim_idx}"
            self.frame_data_stored[trim_key] = {}
            self.frame_percent_stored[trim_key] = {}
    
            vial_df = pd.read_csv(csv_path)
            max_y1 = np.max(vial_df['y1'])
            max_y1_val = max_y1+self.adder_val
            vial_boundaries, vial_numbers = self._calc_vial_boundaries(vial_df)

            cap = cv2.VideoCapture(vid_path)
            for frame_number in range(780):
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_data_stored[trim_key][frame_number] = {}
                self.frame_percent_stored[trim_key][frame_number] = {}

                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                frame_red = frame[:, :, 2].astype(np.float32) / 255.0
                diff = frame_red - all_backgrounds[trim_idx - 1]
                norm = (diff - diff.min()) / (diff.max() - diff.min())

                height, width = norm.shape
                thresh = np.zeros_like(norm, dtype=np.uint8)
                thresh[:height//2] = (norm[:height//2] < self.top_thresh).astype(np.uint8) * 255
                thresh[height//2:] = (norm[height//2:] < self.bottom_thresh).astype(np.uint8) * 255 

                # thresh_val = 0.40 if frame_number <= 60 else 0.65
                # thresh = (norm < thresh_val).astype(np.uint8) * 255
                _, centroids, _ = self.connectComp(thresh)

                # 6. Assign each detected centroid to its correct vial key
                vial_centroids = {vial_num: [] for vial_num in vial_numbers}
                if centroids:
                    centroids_np = np.array(centroids)
                    used = np.zeros(len(centroids_np), dtype=bool)
        
                    for vial_num in vial_numbers:
                        vb = vial_boundaries[vial_num]
                        mask = (~used) & \
                               (centroids_np[:, 0] >= vb['x1']) & (centroids_np[:, 0] <= vb['x2']) & \
                               (centroids_np[:, 1] >= vb['y1']) & (centroids_np[:, 1] <= vb['y2'])
                        indices = np.where(mask)[0]
                        if indices.size > 0:
                            vial_centroids[vial_num] = [tuple(pt) for pt in centroids_np[indices]]
                            used[indices] = True
                else:
                    print(f"no centroid at frame {frame_number} for trim {trim_idx}")

                for vial_num, points in vial_centroids.items():
                    vial_key = f'Vial_{vial_num}'
                    if not points:
                        continue

                    self.frame_data_stored[trim_key][frame_number][vial_key] = {
                        zone: {'x_pos': [], 'y_pos': [], 'count': 0}
                        for zone in ['HP', 'MP', 'LP', 'Total']
                    }
                    self.frame_percent_stored[trim_key][frame_number][vial_key] = {
                        'Percent': {z: '0%' for z in ['HP', 'MP', 'LP', 'Total']},
                        'Count':   {z:  0   for z in ['HP', 'MP', 'LP', 'Total']}
                    }
    
                    hp_boundary = vial_boundaries[vial_num]['hp_boundary']
                    mp_boundary = vial_boundaries[vial_num]['mp_boundary']
    
                    hp_x, hp_y = [], []
                    mp_x, mp_y = [], []
                    lp_x, lp_y = [], []
    
                    for x, y in points: 
                        if y < max_y1_val and frame_number <= 120:
                            continue
                        if frame_number >= 10:
                            if y < hp_boundary:
                                hp_x.append(x)
                                hp_y.append(y)
                            elif y < mp_boundary:
                                mp_x.append(x)
                                mp_y.append(y)
                            else:
                                lp_x.append(x)
                                lp_y.append(y)
                        else:
                            if y >= mp_boundary:
                                lp_x.append(x)
                                lp_y.append(y)
    
                    self.frame_data_stored[trim_key][frame_number][vial_key]['HP'] = {
                        'x_pos': hp_x, 'y_pos': hp_y, 'count': len(hp_x)
                    }
                    self.frame_data_stored[trim_key][frame_number][vial_key]['MP'] = {
                        'x_pos': mp_x, 'y_pos': mp_y, 'count': len(mp_x)
                    }
                    self.frame_data_stored[trim_key][frame_number][vial_key]['LP'] = {
                        'x_pos': lp_x, 'y_pos': lp_y, 'count': len(lp_x)
                    }
                    self.frame_data_stored[trim_key][frame_number][vial_key]['Total'] = {
                        'x_pos': hp_x + mp_x + lp_x,
                        'y_pos': hp_y + mp_y + lp_y,
                        'count': len(hp_x) + len(mp_x) + len(lp_x)
                    }
    
                    geno_row = self.geno_df[self.geno_df['Vial_Num'] == vial_num]
                    if not geno_row.empty:
                        n = geno_row['N'].values[0]
                    else:
                        n = 1
    
                    total_detected = len(hp_x) + len(mp_x) + len(lp_x)
                    if total_detected > 0:
                        counts = {
                            'HP': len(hp_x),
                            'MP': len(mp_x),
                            'LP': len(lp_x),
                            'Total': total_detected
                        }
                        percent = {
                            'HP': f"{round((len(hp_x)/total_detected)*100, 2)}%",
                            'MP': f"{round((len(mp_x)/total_detected)*100, 2)}%",
                            'LP': f"{round((len(lp_x)/total_detected)*100, 2)}%",
                            'Total': f"{round((total_detected/total_detected)*100, 1)}%"
                        }
                        self.frame_percent_stored[trim_key][frame_number][vial_key]['Percent'] = percent
                        self.frame_percent_stored[trim_key][frame_number][vial_key]['Count'] = counts
                    
            cap.release()

        # --- Save RAW Position Data --- #
        for trim, frames in tqdm(self.frame_data_stored.items(), desc="Saving raw positional data"):
            for cat in ['LP','MP','HP','Total']:
                recs = []
                for f, frame_data in frames.items():
                    for vial, data in frame_data.items():
                        y_list = data[cat]['y_pos']
                        if not y_list:
                            continue
                        y_mean = np.mean(y_list)
                        recs.append({'Frame': f, 'Vial': vial, 'Raw_Y': y_mean})
                df = pd.DataFrame(recs)
                df['Y_Position'] = df.groupby('Vial')['Raw_Y'].transform(lambda y0: (17 * (y0.iloc[0] - y0)).clip(lower=0) / 720)
                df['Seconds'] = df['Frame'] / self.fps
                df = df[['Frame', 'Vial', 'Y_Position', 'Seconds']]
                raw_path = os.path.join(self.vid_path, f"raw_position_{cat}_df.csv")
                df.to_csv(raw_path, index=False)


        self.output_data()

    def output_data(self, frame_step=30):
        all_perc_dfs = {}
        for trim, frames in tqdm(self.frame_percent_stored.items(), desc="Aggregating percentage data"):
            min_f, max_f = min(frames), max(frames)
            all_perc_dfs[trim] = {}
            for cat in ['LP','MP','HP']:
                recs = []
                for f in range(min_f, max_f+1, frame_step):
                    for vial, data in frames[f].items():
                        pct = float(data['Percent'][cat].strip('%'))
                        recs.append({'Frame': f, 'Vial': vial, 'Percent': pct})
                df = pd.DataFrame(recs)
                df['Seconds'] = df['Frame'] / self.fps
                all_perc_dfs[trim][cat] = df

        all_pos_dfs = {}
        for trim, frames in tqdm(self.frame_data_stored.items(), desc="Aggregating positional data"):
            min_f, max_f = min(frames), max(frames)
            all_pos_dfs[trim] = {}
            for cat in ['LP','MP','HP','Total']:
                recs = []
                first_y = {}
                for f in range(min_f, max_f+1, frame_step):
                    for vial, data in frames[f].items():
                        y_list = data[cat]['y_pos']
                        if not y_list:
                            continue
                        y_mean = np.mean(y_list)
                        if vial not in first_y:
                            first_y[vial] = y_mean
                        recs.append({'Frame': f, 'Vial': vial, 'Raw_Y': y_mean})
                for r in recs:
                    base = first_y[r['Vial']]
                    adj  = max(0, base - r['Raw_Y'])
                    r['Y_Position'] = (17 * adj) / 720
                    r['Seconds']    = r['Frame'] / self.fps
                df = pd.DataFrame(recs)[['Frame','Vial','Y_Position','Seconds']]
                all_pos_dfs[trim][cat] = df

        aggregated_pos, aggregated_perc = self.aggregate_data_across_trims(
            all_pos_dfs, all_perc_dfs, self.geno_df
        )
        self.saving_plot_n_data(aggregated_perc, aggregated_pos, frame_step)

# def reaggregate_from_csv(self, frame_step=30):
#     all_perc_dfs = {}
#     all_pos_dfs = {}
#
#     # Load genotype metadata
#     self.geno_df = pd.read_csv(os.path.join(self.vid_path, "genotype_metadata.csv"))
#     vial_order = [f'Vial_{vial_num}' for vial_num in self.geno_df['Vial_Num'].values]
#
#     # --- Load Percentage CSVs --- #
#     for cat in ['LP', 'MP', 'HP']:
#         csv_path = os.path.join(self.vid_path, f"percentage_{cat}_df.csv")
#         if not os.path.exists(csv_path):
#             print(f"Missing {csv_path}")
#             continue
#         df = pd.read_csv(csv_path)
#
#         # Ensure proper format
#         if 'Seconds' not in df.columns:
#             df['Seconds'] = df['Frame'] / self.fps
#         if 'Trim' not in df.columns:
#             df['Trim'] = 'trim_1'
#
#         all_perc_dfs.setdefault('trim_1', {})[cat] = df
#
#     # --- Load Positional CSVs --- #
#     for cat in ['LP', 'MP', 'HP', 'Total']:
#         csv_path = os.path.join(self.vid_path, f"position_{cat}_df.csv")
#         if not os.path.exists(csv_path):
#             print(f"Missing {csv_path}")
#             continue
#         df = pd.read_csv(csv_path)
#
#         # Fix missing columns
#         if 'Y_Position' not in df.columns and 'Raw_Y' in df.columns:
#             df['Y_Position'] = df['Raw_Y']
#         if 'Seconds' not in df.columns:
#             df['Seconds'] = df['Frame'] / self.fps
#         if 'Trim' not in df.columns:
#             df['Trim'] = 'trim_1'
#
#         # Ensure column order
#         df = df[['Frame', 'Vial', 'Y_Position', 'Seconds', 'Trim']]
#
#         all_pos_dfs.setdefault('trim_1', {})[cat] = df
#
#     # --- Aggregate and Plot --- #
#     aggregated_pos, aggregated_perc = self.aggregate_data_across_trims(
#         all_pos_dfs, all_perc_dfs, self.geno_df
#     )
#     self.saving_plot_n_data(aggregated_perc, aggregated_pos, frame_step)
