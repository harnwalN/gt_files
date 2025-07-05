import os, warnings, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm, kruskal
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import re


class StatisticalAnalysis:
    def __init__(self, experiment_path, gender):
        self.experiment = experiment_path
        if self.experiment == "": return
        self.data_type_list = ['position', 'velocity', 'low performer', 'middle performer', 'high performer']
        self.base_path = experiment_path
        self.gender = gender
        self.gender_folder = '_Output_Males' if self.gender == 'male' else '_Output_Females'
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        # We will set self.stats_folder_name later in input_comps(), once we know gender and control genotype.

        # Run iteration with position to get control and comparison genotypes
        self.data_type = 'position'
        self.load_data()
        self.prepare_data()

    def run_analysis(self, start_time, end_time):
        self.messages = []
        self.end_time = float(end_time)
        self.start_time = float(start_time)
        self.stats_folder_name = f"{self.gender}_stats_con_['{self.control_genotype}']_ctr_{sorted(self.comparison_genotypes)}_time_cut_{self.start_time}s-{self.end_time}s"
        try:
            print(f"\nProcessing {self.gender} data from {self.start_time}s to {self.end_time}s...\n")
            self.messages.append(f"\nProcessing {self.gender} data from {self.start_time}s to {self.end_time}s...\n")
            for data_type in self.data_type_list:
                self.data_type = data_type
                self.load_data()
                self.prepare_data()
                if self.data_type == 'position':
                    self.skip_stats = True
                else: self.skip_stats = False # Default to False for non-'position' data

                self.select_genotypes()
                if not self.skip_stats:
                    self.perform_mixedlm_analysis()
                else:
                    self.combined_results = {}
                self.plot_results()
                if not self.skip_stats:
                    self.kruskal_analysis()

        except Exception as e:
            print(f"Error processing {self.gender}: {e}")
            self.messages.append(f"Error processing {self.gender}: {e}")

        return self.messages

    def load_data(self):
        file_map = {
            'position': f'{self.gender}_stats_pos_data.csv',
            'velocity': f'{self.gender}_stats_vel_data.csv',
            'low performer': f'{self.gender}_stats_lp_perc_data.csv',
            'middle performer': f'{self.gender}_stats_mp_perc_data.csv',
            'high performer': f'{self.gender}_stats_hp_perc_data.csv'
        }
        file_path = f'{self.base_path}/{self.gender_folder}/{file_map[self.data_type]}'
        self.df = pd.read_csv(file_path)
        # drop any weird unnamed columns
        self.df = self.df.loc[:, ~self.df.columns.str.startswith('Unnamed: 0.1')]
        # print(f"load_data -> self.df: \n{self.df}\n")

    def prepare_data(self):
        start_range = 1
        end_range = len(self.df.columns)
        newdf = self.df.iloc[:, [0] + list(range(start_range, end_range))].copy()
        # print(f"prepare_data -> newdf: \n{newdf}\n")

        if 'Unnamed: 0' in newdf.columns:
            newdf[['Genotype', 'Replicate']] = newdf['Unnamed: 0'].str.split('_rep', expand=True)
        else:
            raise KeyError("'Unnamed: 0' column is missing in newdf.")

        self.df_long = newdf.melt(
            id_vars=['Genotype', 'Replicate'],
            value_vars=newdf.columns[1:],
            var_name='Time',
            value_name='Position'
        )
        self.df_long['Time'] = self.df_long['Time'].astype(float)
        self.df_long['Subject'] = self.df_long['Genotype'] + '_' + self.df_long['Replicate']
        # print(f"prepare_data -> self.df_long: \n{self.df_long}\n")

    def input_comps(self, control_genotype, comparison_genotypes):
        self.control_genotype = control_genotype.strip().strip("'").strip('"')

        # If no input is given, only control is used
        if "None" in comparison_genotypes and len(comparison_genotypes) == 0:
            print(
                f"No comparison genotypes provided for {self.gender} {self.data_type}. Skipping statistical analysis.")
            self.messages.append(
                f"No comparison genotypes provided for {self.gender} {self.data_type}. Skipping statistical analysis.")
            self.comparison_genotypes = []
            self.skip_stats = True
        else:
            if "None" in comparison_genotypes: comparison_genotypes.remove("None")
            self.comparison_genotypes = comparison_genotypes
            self.skip_stats = False

        # Clean up any commas in genotype names
        all_genos = [self.control_genotype] + self.comparison_genotypes
        clean_map = {geno: geno.replace(',', '_') for geno in all_genos}
        self.df_long['Genotype'] = self.df_long['Genotype'].replace(clean_map)
        self.control_genotype = clean_map[self.control_genotype]
        self.comparison_genotypes = [clean_map[g] for g in self.comparison_genotypes]

    def select_genotypes(self):
        self.comb_genos = [self.control_genotype] + self.comparison_genotypes
        self.test_df = self.df_long[self.df_long['Genotype'].isin(self.comb_genos)].copy()
        self.original_df = self.test_df.copy()
        self.test_df = self.test_df.dropna(subset=["Position"]).reset_index(drop=True)
        self.test_df['Time_cat'] = self.test_df['Time'].astype(str)
        self.test_df = self.test_df[self.test_df['Time'] <= self.end_time].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df['Time'] >= self.start_time].reset_index(drop=True)
        self.replicate_counts = self.test_df.groupby('Genotype')['Replicate'].nunique().to_dict()

    def significance_stars(self, p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "NS"

    def perform_mixedlm_analysis(self):
        model_rs = smf.mixedlm(
            f"Position ~ C(Genotype, Treatment(reference='{self.control_genotype}')) * Time_cat",
            self.test_df,
            groups=self.test_df["Subject"],
            re_formula="~Time"
        )
        self.result = model_rs.fit()

        self.pvalues_df = pd.DataFrame({
            "Term": self.result.pvalues.index,
            "P-Value": self.result.pvalues.values,
            "Significance": [self.significance_stars(p) for p in self.result.pvalues.values]
        })
        self.pvalues_df["P-Value"] = self.pvalues_df["P-Value"].apply(lambda x: f"{x:.6f}")

        def combine_pvalues_hmp(pvals):
            m = len(pvals)
            eps = np.finfo(float).tiny
            p_val = np.clip(pvals, eps, 1.0)
            hmp = m / np.sum(1.0 / p_val)
            return min(hmp, 1.0)

        self.combined_results = {}
        self.geno_terms = {}
        for geno in self.comparison_genotypes:
            main_term = f"C(Genotype, Treatment(reference='{self.control_genotype}'))[T.{geno}]"
            interaction_prefix = f"C(Genotype, Treatment(reference='{self.control_genotype}'))[T.{geno}]:Time_cat"
            interaction_terms = [term for term in self.result.pvalues.index if term.startswith(interaction_prefix)]
            terms = [main_term] + interaction_terms
            pvals = self.result.pvalues[terms].values
            self.geno_terms[geno] = (pvals)

        for geno, (pvals) in self.geno_terms.items():
            HMP_p = combine_pvalues_hmp(pvals)
            HMP_sig_p = self.significance_stars(HMP_p)
            print(f"{self.control_genotype} vs {geno} for {self.data_type}: {HMP_p}, {HMP_sig_p}")
            self.messages.append(f"{self.control_genotype} vs {geno} for {self.data_type}: {HMP_p}, {HMP_sig_p}")
            self.combined_results[geno] = {"Harm_Mean": HMP_p, "Harm_Mean_Sig": HMP_sig_p}

    def plot_results(self):
        using_list = [self.control_genotype] + self.comparison_genotypes
        legend_labels = {
            self.control_genotype: f"{self.control_genotype} (N={self.replicate_counts[self.control_genotype]})"
        }
        for comparison, pval_type in self.combined_results.items():
            star_sig_value = pval_type['Harm_Mean_Sig']
            p_value = "{:.6f}".format(pval_type['Harm_Mean'])
            legend_labels[comparison] = (
                f"{comparison} (N={self.replicate_counts[comparison]}): (p: {p_value}) {star_sig_value}"
            )

        plt.figure(figsize=(8, 6))
        df_filtered = self.original_df[self.original_df['Genotype'].isin(using_list)]
        color_palette = sns.color_palette(n_colors=len(using_list))
        sns.lineplot(
            data=df_filtered,
            x="Time",
            y="Position",
            hue="Genotype",
            marker='o',
            err_style='band',
            errorbar='se',
            palette=color_palette
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [legend_labels[label] if label in legend_labels else label for label in labels]
        plt.legend(handles, new_labels, title="Genotype", loc="lower left", bbox_to_anchor=(0, -0.2), fontsize=7)

        if self.data_type == 'velocity':
            plt.axhline(y=0, color='red', linestyle='--')
            plt.ylabel('Climbing Velocity (cm/sec)')
        elif self.data_type == 'position':
            plt.ylabel('Climbing Position (cm)')
        else:
            plt.ylabel(f'Climbing {self.data_type.capitalize()} %')

        plt.axvline(x=self.start_time, color='black', linestyle='--')
        plt.axvline(x=self.end_time, color='black', linestyle='--')
        plt.title(
            f"{self.data_type.capitalize()} Trends Over Time With Control {self.control_genotype} for {self.gender.capitalize()}")
        plt.grid(axis='y')

        # ←— CHANGED: use the unique stats_folder_name instead of just control_genotype
        output_folder = os.path.join(self.base_path, self.gender_folder, self.stats_folder_name)  # <<< MODIFIED
        os.makedirs(output_folder, exist_ok=True)

        save_path = os.path.join(output_folder, f'{self.gender}_stats_plot_{self.data_type}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def kruskal_analysis(self):
        time_col = 'Time'
        genotype_col = 'Genotype'
        value_col = 'Position'

        times = sorted(self.test_df[time_col].unique())
        result = pd.DataFrame(index=times, columns=self.comparison_genotypes, dtype=object)
        df_ctrl = self.test_df[self.test_df[genotype_col] == self.control_genotype]

        for t in times:
            vals_ctrl = df_ctrl.loc[df_ctrl[time_col] == t, value_col].dropna().values
            for comp in self.comparison_genotypes:
                df_comp = self.test_df[self.test_df[genotype_col] == comp]
                vals_comp = df_comp.loc[df_comp[time_col] == t, value_col].dropna().values

                if len(vals_ctrl) < 2 or len(vals_comp) < 2:
                    p = np.nan
                else:
                    try:
                        _, p = kruskal(vals_ctrl, vals_comp)
                    except ValueError:
                        p = 1.0

                if np.isnan(p):
                    entry = np.nan
                else:
                    entry = [f"{p:.4f}", self.significance_stars(p)]

                result.at[t, comp] = entry

        result.index.name = time_col

        # ←— CHANGED: once again, write CSV into the same unique folder
        output_folder = os.path.join(self.base_path, self.gender_folder, self.stats_folder_name)  # <<< MODIFIED
        os.makedirs(output_folder, exist_ok=True)

        save_path_csv = os.path.join(output_folder,f'{self.gender}_stats_kruskal_{self.data_type}_control_{self.control_genotype}.csv')
        result.to_csv(save_path_csv)