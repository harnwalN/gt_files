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

        # self.add_grid_overlay(12, 4)

        # Set style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Segoe UI", 14), padding=5)
        self.style.configure("TLabel", font=("Segoe UI", 20), background="#d9d9d9")

        # Set up grid
        self.grid_columnconfigure([0, 1, 2, 3], weight=1, uniform="a")
        self.grid_rowconfigure([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], weight=1, uniform="a")

        # Log display
        self.log_area = scrolledtext.ScrolledText(self, state='disabled', bg="#d5b1a9")
        self.log_area.grid(row=0, column=1, rowspan=5, columnspan=2, sticky="nsew")

        # Configure video folder
        self.vid_folder_label = ttk.Label(self, text=f"Current Video Folder: {self.controller.experiment_dir}")
        self.vid_folder_label.grid(row=5, column=1, columnspan=2)
        self.vid_folder_button = ttk.Button(self, text="Select Folder", command=self.select_dir)
        self.vid_folder_button.grid(row=6, column=1, columnspan=2)

        # Experiment name
        ttk.Label(self, text="Video Name:").grid(row=7, column=1, columnspan=2)
        self.experiment_var = StringVar(value=[i for i in sorted(os.listdir(self.controller.experiment_dir)) if i[0] != "."][0])
        experiment_options = [e for e in sorted(os.listdir(self.controller.experiment_dir)) if e[0] !="."]
        self.experiment_menu = OptionMenu(self, self.experiment_var, *experiment_options)
        self.experiment_menu.grid(row=8, column=1, columnspan=2, sticky="n")

        # Lamp option
        ttk.Label(self, text="Using Lamp?").grid(row=9, column=1, columnspan=2)
        self.lamp_var = StringVar(value="Yes")
        OptionMenu(self, self.lamp_var, "Yes", "No").grid(row=10, column=1, columnspan=2, sticky="n")

        # Enter
        self.go_button = ttk.Button(self, text="Go", command=self.go)
        self.go_button.grid(row=11, column=1, rowspan=1, columnspan=2)

    def select_dir(self):
        self.controller.experiment_dir = filedialog.askdirectory(initialdir=os.getcwd(), title="Choose Video Folder")

        if self.experiment_menu:
            self.experiment_menu.destroy()

        self.vid_folder_label.config(text=self.controller.experiment_dir)

        experiment_options = [e for e in sorted(os.listdir(self.controller.experiment_dir)) if e[0] != "."]
        if experiment_options:
            self.experiment_var.set(experiment_options[0])  # Set to first valid option
        else:
            self.experiment_var.set("")

        # Create new menu
        self.experiment_menu = OptionMenu(self, self.experiment_var, *experiment_options)
        self.experiment_menu.grid(row=8, column=1, columnspan=2, sticky="n")

    def log_message(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message)
        self.log_area.insert(tk.END, "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.update_idletasks()
        print(message)

    def go(self):
        self.log_message("Starting Analysis:")
        self.log_message(f"    Experiment Name: {self.experiment_var.get()}")
        self.log_message(f"    Using Lamp: {self.lamp_var.get()}")
        self.log_message("")

        # Set experiment variables
        self.controller.experiment_name = self.experiment_var.get()
        self.controller.experiment_path = os.path.join(self.controller.experiment_dir, self.controller.experiment_name)
        self.controller.set_using_lamp(self.lamp_var.get().strip().capitalize())
        self.controller.set_stats_interval([0, 9]) # Set by default, can change in StatsFrame
        self.controller.entered = True;

        # Run starting analysis
        spec_video_folders = [d for d in sorted(os.listdir(self.controller.experiment_path))
                              if os.path.isdir(os.path.join(self.controller.experiment_path, d))
                              and d not in ('.ipynb_checkpoints', '_Output_Males', '_Output_Females')]

        at_least_one_video = False
        for i, spec_video in enumerate(spec_video_folders, start=1):
            main_folder_path = f"{self.controller.experiment_path}/{spec_video}"
            safe = self.dissect_video(spec_video, main_folder_path)
            at_least_one_video = at_least_one_video or safe
            if not safe: self.log_message("\n    Skipping " + spec_video)

        if not at_least_one_video:
            self.log_message("No valid videos found")
            return

        # ________________DATA_SAVED_AND_PLOTTED________________#
        analyzer = GenotypeAnalyzer(self.controller.experiment_path)
        analyzer.run()

        self.controller.show_frame("StatsFrame")

    def dissect_video(self, spec_video, main_folder_path):
        safe = self.check_fatal_files_main(spec_video, main_folder_path)
        if not safe:
            print(spec_video, "not working")
            return False

        self.log_message(f"    Processing video folder: {spec_video}")
        self.create_required_files_main(spec_video, main_folder_path)
        self.create_required_trim_mp4_files(spec_video, main_folder_path)
        self.create_required_trim_output(spec_video, main_folder_path)

        self.controller.fin_geo[spec_video] = FinalizedGeotaxis(experiment=self.controller.experiment_path,
                                                 spec_vid=os.path.basename(spec_video),
                                                 fps=60, top_thresh=0.50,
                                                 bottom_thresh=0.55,
                                                 adder_val=150, remove_px=125)

        self.log_message(f"    All required files found for {spec_video}.")

        return True

    def check_fatal_files_main(self, video_folder, main_folder_path):
        fatal_files_main = [
            f"{video_folder}.h264",
            "genotype_metadata.csv"
        ]
        for file in fatal_files_main:
            if not os.path.exists(os.path.join(main_folder_path, file)) and not os.path.exists(os.path.join(main_folder_path, f"{video_folder}.mp4")):
                self.log_message(f"    FATAL: missing main file {file}.\n")
                print("FATAL Path", os.path.join(main_folder_path, file))
                return False
        return True

    def create_required_files_main(self, video_folder, main_folder_path):
        required_files_main = [
            f"{video_folder}.mp4",
        ]
        for file in required_files_main:
            if not os.path.exists(os.path.join(main_folder_path, file)):
                self.log_message(f"    {video_folder}.h264 has not been converted to .mp4")
                self.log_message(f"    Creating required main file: {file}")
                # ______________H264_TO_MP4_________________
                input_file = f"{self.controller.experiment_path}/{video_folder}/{video_folder}.h264"
                output_file = f"{self.controller.experiment_path}/{video_folder}/{video_folder}.mp4"

                converter = VideoConverter(input_file, output_file)
                converter.convert()
                del input_file, output_file
                self.log_message("")

    def create_required_trim_mp4_files(self, video_folder, main_folder_path):
        required_trim_mp4_files = [
            f"TRIM_1_{video_folder}.mp4",
            f"TRIM_2_{video_folder}.mp4",
            f"TRIM_3_{video_folder}.mp4",
            f"TRIM_4_{video_folder}.mp4"
        ]
        for file in required_trim_mp4_files:
            if not os.path.exists(os.path.join(main_folder_path, file)):
                self.log_message(f"    {file} has not been created")

                # _____________VIDEO_SNIPS__________________
                video_path = f"{self.controller.experiment_path}/{video_folder}/{video_folder}.mp4"
                self.log_message(f"    Creating Video TRIMS for {video_folder}:")
                processor = VideoProcessor(video_path)
                processor.process_video()
                processor.video_filt()

                trims_ttl = self.vid_clips
                for trim_cnt in range(1, trims_ttl + 1):
                    start_frame = processor.frame_ranges_df.iloc[trim_cnt - 1]['start_frame']
                    end_frame = processor.frame_ranges_df.iloc[trim_cnt - 1]['end_frame']
                    processor.crop_video(start_frame, end_frame, trim_cnt)

    def create_required_trim_output(self, video_folder, main_folder_path):
        required_trim_csv_files = [
            f"trim_1_{video_folder}_vials_pos.csv",
            f"trim_2_{video_folder}_vials_pos.csv",
            f"trim_3_{video_folder}_vials_pos.csv",
            f"trim_4_{video_folder}_vials_pos.csv"
        ]
        required_csv_outputs = [
            "percentage_HP_df.csv",
            "percentage_LP_df.csv",
            "percentage_MP_df.csv",
            "position_Total_df.csv"
        ]
        required_png_outputs = [
            "percentage_HP_plot.png",
            "percentage_LP_plot.png",
            "percentage_MP_plot.png",
            "position_Total_plot.png"
        ]
        for file in required_trim_csv_files + required_csv_outputs + required_png_outputs:
            if not os.path.exists(os.path.join(main_folder_path, file)):
                self.log_message(f"    Missing trimmed video output: {file}")

                # ______________VIAL_NETWORK___________________
                genotype_csv_pth = f"{self.controller.experiment_path}/{video_folder}/genotype_metadata.csv"
                vials_to_drop, vial_num_list = self.geno_meta(genotype_csv_pth)

                temp = list(range(1, self.vid_clips + 1))

                video_inputs = [f"{self.controller.experiment_path}/{video_folder}/TRIM_{i}_{video_folder}.mp4"
                                for i
                                in temp]

                vial_pos_lists = []
                self.log_message(f"VIALS USED: \n{vial_num_list}\n VIALS USED LENGTH: {len(vial_num_list)}\n")

                for idx, vid in enumerate(video_inputs, start=1):
                    self.log_message(f"Start Vial Network for {video_folder} TRIM {idx}:")
                    vial_network = Vial_Network(self.controller.experiment_path, video_folder, idx, vid, vials_to_drop,
                                                self.controller.using_lamp)
                    vial_network.predict_and_display()
                    # vial_network.save_model("gt_newVial_nn.pth")

                    vials_input = f"{self.controller.experiment_path}/{video_folder}/trim_{idx}_{video_folder}_vials_pos.csv"
                    vial_pos_lists.append(vials_input)

                    # ________________GEOTAXIS_MAIN________________#
                    print(f"\n\nRUNNING Video: '{os.path.basename(video_folder)}':\n")
                    fin_geo = FinalizedGeotaxis(experiment=self.controller.experiment_path,
                                                 spec_vid=os.path.basename(video_folder),
                                                 fps=60, top_thresh=0.50,
                                                 bottom_thresh=0.55,
                                                 adder_val=150, remove_px=125)
                    fin_geo.run()

    def geno_meta(self, genotype_csv_input, n=12):
        geno_df = pd.read_csv(genotype_csv_input)
        vial_num_list = geno_df["Vial_Num"].tolist()
        irrel_vials = list(set(range(1, n + 1)) - set(geno_df["Vial_Num"].tolist()))
        return irrel_vials, vial_num_list

    def add_grid_overlay(parent, rows, cols):
        """Adds visible labels in each grid cell to visualize layout."""
        for r in range(rows):
            for c in range(cols):
                label = tk.Label(
                    parent,
                    text=f"{r},{c}",
                    bg="#e0e0e0",
                    fg="black",
                    font=("Arial", 8),
                    borderwidth=1,
                    relief="solid"
                )
                label.grid(row=r, column=c, sticky="nsew")