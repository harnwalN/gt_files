import os
import tkinter as tk
from tkinter import *
from tkinter import ttk, scrolledtext, filedialog

import pandas as pd

from gt_data_agg import GenotypeAnalyzer
from gt_process import FinalizedGeotaxis
from vial_network import Vial_Network
from video_converter import VideoConverter
from video_processor import VideoProcessor


class MenuFrame(ttk.Frame):
    def __init__(self, parent, controller, vid_clips=4):
        super().__init__(parent)
        self.controller = controller
        self.vid_clips = vid_clips

        self.grid_columnconfigure([0, 1, 2, 3], weight=1, uniform="a")
        self.grid_rowconfigure([*range(13)], weight=1, uniform="a")

        self.log_area = scrolledtext.ScrolledText(self, state='disabled', bg="#d5b1a9")
        self.log_area.grid(row=0, column=1, rowspan=5, columnspan=2, sticky="nsew")

        self.progress = ttk.Progressbar(self, mode='determinate', maximum=1)
        self.progress.grid(row=5, column=1, columnspan=2, sticky="new")

        self.vid_folder_label = ttk.Label(self, text=f"Current Video Folder: {self.controller.experiment_dir}", font=("Segoe UI", 16))
        self.vid_folder_label.grid(row=6, column=1, columnspan=2)
        self.vid_folder_button = ttk.Button(self, text="Select Folder", command=self.select_dir)
        self.vid_folder_button.grid(row=7, column=1, columnspan=2)

        ttk.Label(self, text="Video Name:", font=("Segoe UI", 16)).grid(row=8, column=1, columnspan=2)
        self.experiment_var = StringVar(value=[i for i in sorted(os.listdir(self.controller.experiment_dir)) if i[0] != "."][0])
        experiment_options = [e for e in sorted(os.listdir(self.controller.experiment_dir)) if e[0] != "."]
        self.experiment_menu = OptionMenu(self, self.experiment_var, *experiment_options)
        self.experiment_menu.grid(row=9, column=1, columnspan=2, sticky="n")

        ttk.Label(self, text="Using Lamp?", font=("Segoe UI", 16)).grid(row=10, column=1, columnspan=2)
        self.lamp_var = StringVar(value="Yes")
        OptionMenu(self, self.lamp_var, "Yes", "No").grid(row=11, column=1, columnspan=2, sticky="n")

        self.go_button = ttk.Button(self, text="Go", command=self.go)
        self.go_button.grid(row=12, column=1, rowspan=1, columnspan=2)

    def select_dir(self):
        self.controller.experiment_dir = filedialog.askdirectory(initialdir=os.getcwd(), title="Choose Video Folder")
        if self.experiment_menu:
            self.experiment_menu.destroy()
        self.vid_folder_label.config(text=self.controller.experiment_dir)

        experiment_options = [e for e in sorted(os.listdir(self.controller.experiment_dir)) if e[0] != "."]
        self.experiment_var.set(experiment_options[0] if experiment_options else "")
        self.experiment_menu = OptionMenu(self, self.experiment_var, *experiment_options)
        self.experiment_menu.grid(row=9, column=1, columnspan=2, sticky="n")

    def log_message(self, message=""):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.update_idletasks()
        print(message)

    def go(self):
        self.go_button.config(state="disabled")
        self.progress.start(10)
        self.update_idletasks()

        self.log_message("Starting Analysis:")
        self.log_message(f"    Experiment Name: {self.experiment_var.get()}")
        self.log_message(f"    Using Lamp: {self.lamp_var.get()}")
        self.log_message()

        self.controller.experiment_name = self.experiment_var.get()
        self.controller.experiment_path = os.path.join(self.controller.experiment_dir, self.controller.experiment_name)
        self.controller.set_using_lamp(self.lamp_var.get().strip().capitalize())
        self.controller.set_stats_interval([0, 9])
        self.controller.entered = True

        self.spec_video_folders = [d for d in sorted(os.listdir(self.controller.experiment_path))
                                   if os.path.isdir(os.path.join(self.controller.experiment_path, d))
                                   and d not in ('.ipynb_checkpoints', '_Output_Males', '_Output_Females')]

        self.at_least_one_video = False
        self.folder_index = 0
        self.after(100, self.process_next_folder)

    def process_next_folder(self):
        if self.folder_index >= len(self.spec_video_folders):
            if not self.at_least_one_video:
                self.log_message("No valid videos found")
                return
            self.log_message("Analyzing videos...")
            self.after(100, self.analyze_videos)
            return

        spec_video = self.spec_video_folders[self.folder_index]
        self.folder_index += 1

        main_folder_path = os.path.join(self.controller.experiment_path, spec_video)
        safe = self.dissect_video(spec_video, main_folder_path)
        self.at_least_one_video = self.at_least_one_video or safe
        if not safe:
            self.log_message(f"\n    Skipping {spec_video}")

        self.progress["value"] = self.folder_index / (len(self.spec_video_folders) + 1)
        self.after(100, self.process_next_folder)

    def analyze_videos(self):
        analyzer = GenotypeAnalyzer(self.controller.experiment_path)
        analyzer.run()
        self.progress["value"] = 1
        self.log_message("Completed analysis.\n")
        self.progress.stop()
        self.controller.show_frame("StatsFrame")
        self.go_button.config(state="normal")

    def dissect_video(self, spec_video, main_folder_path):
        if not self.check_fatal_files_main(spec_video, main_folder_path):
            return False

        self.log_message(f"    Processing video folder: {spec_video}")

        self.after(100, lambda: self.check_required_files_main(spec_video, main_folder_path, lambda:
            self.check_required_trim_mp4_files(spec_video, main_folder_path, lambda:
                self.check_required_trim_output(spec_video, main_folder_path, lambda:
                    self.finish_video_processing(spec_video)
                )
            )
        ))
        return True

    def check_fatal_files_main(self, video_folder, main_folder_path):
        required = [f"{video_folder}.h264", "genotype_metadata.csv"]
        for f in required:
            if not os.path.exists(os.path.join(main_folder_path, f)) and not os.path.exists(os.path.join(main_folder_path, f"{video_folder}.mp4")):
                self.log_message(f"    FATAL: missing main file {f}.\n")
                return False
        return True

    def check_required_files_main(self, video_folder, main_folder_path, on_done):
        if not os.path.exists(os.path.join(main_folder_path, f"{video_folder}.mp4")):
            self.log_message(f"    Creating {video_folder}.mp4")
            self.after(100, lambda: self.create_required_files_main(video_folder, main_folder_path, False, on_done))
        else:
            self.after(100, lambda: self.create_required_files_main(video_folder, main_folder_path, True, on_done))

    def create_required_files_main(self, video_folder, main_folder_path, skip, on_done):
        if not skip:
            converter = VideoConverter(f"{main_folder_path}/{video_folder}.h264", f"{main_folder_path}/{video_folder}.mp4")
            converter.convert()
        self.after(100, on_done)

    def check_required_trim_mp4_files(self, video_folder, main_folder_path, on_done):
        trims = [f"TRIM_{i}_{video_folder}.mp4" for i in range(1, self.vid_clips + 1)]
        for f in trims:
            if not os.path.exists(os.path.join(main_folder_path, f)):
                self.log_message(f"    Creating Video TRIMS for {video_folder}.mp4:")
                self.after(100, lambda: self.create_required_trim_mp4_files(video_folder, main_folder_path, False, on_done))
                return
        self.after(100, lambda: self.create_required_trim_mp4_files(video_folder, main_folder_path, True, on_done))

    def create_required_trim_mp4_files(self, video_folder, main_folder_path, skip, on_done):
        if not skip:
            processor = VideoProcessor(f"{main_folder_path}/{video_folder}.mp4")
            processor.process_video()
            processor.video_filt()
            for i in range(1, self.vid_clips + 1):
                start = processor.frame_ranges_df.iloc[i - 1]['start_frame']
                end = processor.frame_ranges_df.iloc[i - 1]['end_frame']
                processor.crop_video(start, end, i)
        self.after(100, on_done)

    def check_required_trim_output(self, video_folder, main_folder_path, on_done):
        trims = [f"trim_{i}_{video_folder}_vials_pos.csv" for i in range(1, self.vid_clips + 1)]
        outputs = ["percentage_HP_df.csv", "percentage_LP_df.csv", "percentage_MP_df.csv", "position_Total_df.csv"]
        pngs = ["percentage_HP_plot.png", "percentage_LP_plot.png", "percentage_MP_plot.png", "position_Total_plot.png"]
        for f in trims + outputs + pngs:
            if not os.path.exists(os.path.join(main_folder_path, f)):
                self.log_message(f"    Creating trimmed analysis output for {video_folder}.mp4:")
                self.after(100, lambda: self.create_required_trim_output(video_folder, False, on_done))
                return
        self.after(100, lambda: self.create_required_trim_output(video_folder, True, on_done))

    def create_required_trim_output(self, video_folder, skip, on_done):
        if not skip:
            geno_path = f"{self.controller.experiment_path}/{video_folder}/genotype_metadata.csv"
            vials_to_drop, vial_nums = self.geno_meta(geno_path)
            self.log_message(f"        VIALS USED: \n{vial_nums}\n VIALS USED LENGTH: {len(vial_nums)}\n")
            for i in range(1, self.vid_clips + 1):
                vid_path = f"{self.controller.experiment_path}/{video_folder}/TRIM_{i}_{video_folder}.mp4"
                self.log_message(f"        Start Vial Network for {video_folder} TRIM {i}:")
                vn = Vial_Network(self.controller.experiment_path, video_folder, i, vid_path, vials_to_drop, self.controller.using_lamp)
                vn.predict_and_display()
                fin_geo = FinalizedGeotaxis(self.controller.experiment_path, os.path.basename(video_folder), 60, 0.50, 0.55, 150, 125)
                fin_geo.run()
        self.after(100, on_done)

    def finish_video_processing(self, video_folder):
        self.controller.fin_geo[video_folder] = FinalizedGeotaxis(
            experiment=self.controller.experiment_path,
            spec_vid=os.path.basename(video_folder),
            fps=60, top_thresh=0.50,
            bottom_thresh=0.55,
            adder_val=150, remove_px=125
        )
        self.log_message(f"    All required files found for {video_folder}.")

    def geno_meta(self, genotype_csv_input, n=12):
        geno_df = pd.read_csv(genotype_csv_input)
        vial_num_list = geno_df["Vial_Num"].tolist()
        irrel_vials = list(set(range(1, n + 1)) - set(vial_num_list))
        return irrel_vials, vial_num_list
