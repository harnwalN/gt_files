import os
import tkinter as tk
import zipfile
from tkinter import *
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from tqdm import tqdm
import numpy as np

from MenuFrame import MenuFrame
from statistical_analysis import StatisticalAnalysis


class StatsFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        # self.add_grid_overlay(10, 8)

        # Set style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Segoe UI", 12))
        self.style.configure("TLabel", font=("Segoe UI", 12), background="#d9d9d9")

        # Set up grid
        self.grid_columnconfigure([1, 2], weight=4, uniform="a")
        self.grid_columnconfigure([3, 4], weight=3, uniform="a")
        self.grid_columnconfigure([0, 5], weight=2, uniform="a")
        self.grid_columnconfigure([6], weight=30, uniform="a")
        self.grid_columnconfigure([7], weight=1, uniform="a")

        self.grid_rowconfigure([0, 1, 2, 3, 4, 5, 6, 7], weight=1, uniform="a")
        self.grid_rowconfigure([8, 9], weight=0, uniform="a")

        # Back button
        self.back_button = ttk.Button(self, text="<", command=self.return_to_menu)
        self.back_button.grid(row=0, column=0, sticky="w")

        # Experiment title
        ttk.Label(self, text=f"{self.controller.experiment_name}", font=("Segoe UI", 22)).grid(row=0, column=1, columnspan=4)

        # Gender option menu
        ttk.Label(self, text="Gender", font=("Segoe UI", 14)).grid(row=1, column=1, columnspan=2)

        self.gender_var = tk.StringVar(value="male")
        self.gender_menu = tk.OptionMenu(self, self.gender_var, "male", "female", command=self.switch_gender)
        self.gender_menu.config(anchor="center")
        self.gender_menu.grid(row=1, column=3, columnspan=2)

        # Assume male at the start
        self.gender = "male"

        # Frame step
        # ttk.Label(self, text="Frame Steps").grid(row=2, column=1, columnspan=2)
        # self.frame_steps_entry = ttk.Entry(self, width=3, justify="center")
        # self.frame_steps_entry.grid(row=2, column=3, columnspan=2)
        # self.frame_steps_entry.focus_set()
        # self.frame_steps_entry.insert(0, "30")

        # Stats time cut
        ttk.Label(self, text="Stats Time Cut", font=("Segoe UI", 14)).grid(row=3, column=1, columnspan=2)

        self.time_cut_start_entry = ttk.Entry(self, width=3, justify="center")
        self.time_cut_start_entry.grid(row=3, column=3)
        self.time_cut_start_entry.insert(0, "0")

        ttk.Label(self, text="-", font=("Segoe UI", 14)).grid(row=3, column=3, columnspan=2)

        self.time_cut_end_entry = ttk.Entry(self, width=3, justify="center")
        self.time_cut_end_entry.grid(row=3, column=4)
        self.time_cut_end_entry.insert(0, "9")


        if self.controller.entered:
            self.sa = {
                "male": StatisticalAnalysis(self.controller.experiment_path, "male"),
                "female": StatisticalAnalysis(self.controller.experiment_path, "female")
            }
            self.genotype_options = {
                "male": self.sa["male"].df_long['Genotype'].unique(),
                "female": self.sa["female"].df_long['Genotype'].unique()
            }
        else:
            self.genotype_options = {
                "male": np.array([""], dtype=object),
                "female": np.array([""], dtype=object)
            }
        # Control genotype
        ttk.Label(self, text="Control Genotype", font=("Segoe UI", 14)).grid(row=4, column=1, columnspan=2)

        self.control_genotype_var = tk.StringVar(value="--Chose--")
        self.cont_menu = tk.OptionMenu(self, self.control_genotype_var, *self.genotype_options[self.gender])
        self.cont_menu.config(anchor="center")
        self.cont_menu.grid(row=4, column=3, columnspan=2)

        # Comparison genotype
        ttk.Label(self, text="Comparison Genotype", font=("Segoe UI", 14)).grid(row=5, column=1, rowspan=2, columnspan=2)

        self.comparison_list = tk.Listbox(self, selectmode=tk.MULTIPLE)
        for item in self.genotype_options[self.gender].tolist():
            self.comparison_list.insert(tk.END, item)
        self.comparison_list.grid(row=5, column=3, rowspan=2, columnspan=2)

        # Run Button
        self.run_button = ttk.Button(self, text="Run Stats", command=self.run_analysis)
        self.run_button.grid(row=7, column=1, columnspan=2)

        # ZIP Button
        self.zip_button = ttk.Button(self, text="ZIP Results", command=self.zip_folder)
        self.zip_button.grid(row=7, column=3, columnspan=2)

        # Display plots
        self.i1 = 0
        self.img1_canvas = tk.Canvas(self, bd=0, highlightthickness=0, relief="ridge")
        self.img1_canvas.grid(row=0, column=6, rowspan=7, columnspan=1, sticky="nsew")

        # Next plot button
        self.next_button1 = ttk.Button(self, text=">", command=self.change_image1)
        self.next_button2 = ttk.Button(self, text="<", command=self.change_image2)

        # Log display
        self.log_area = scrolledtext.ScrolledText(self, state='disabled', bg="#d5b1a9", height=10)
        self.log_area.grid(row=8, column=6, rowspan=2, sticky="ew")

    def switch_gender(self, gender):
        self.gender = gender
        self.change_image1(no_inc=True)


    def change_image1(self, no_inc=False):
        plots = [
            f"{self.gender}_stats_plot_position.png",
            f"{self.gender}_stats_plot_velocity.png",
            f"{self.gender}_stats_plot_high performer.png",
            f"{self.gender}_stats_plot_middle performer.png",
            f"{self.gender}_stats_plot_low performer.png",
        ]
        if not no_inc: self.i1 += 1
        self.i1 %= len(plots)

        self.img1 = Image.open(
            f"{self.controller.experiment_path}/_Output_{self.gender.capitalize()}s/{self.sa[self.gender].stats_folder_name}/{plots[self.i1]}")

        # Display plot
        event = tk.Event()
        event.width = self.img1_canvas.winfo_width()
        event.height = self.img1_canvas.winfo_height()
        self.stretch_image(event)

    def change_image2(self):
        plots = [
            f"{self.gender}_stats_plot_position.png",
            f"{self.gender}_stats_plot_velocity.png",
            f"{self.gender}_stats_plot_high performer.png",
            f"{self.gender}_stats_plot_middle performer.png",
            f"{self.gender}_stats_plot_low performer.png",
        ]
        self.i1 -= 1
        self.i1 %= len(plots)

        self.img1 = Image.open(
            f"{self.controller.experiment_path}/_Output_{self.gender.capitalize()}s/{self.sa[self.gender].stats_folder_name}/{plots[self.i1]}")

        # Display plot
        event = tk.Event()
        event.width = self.img1_canvas.winfo_width()
        event.height = self.img1_canvas.winfo_height()
        self.stretch_image(event)

    def stretch_image(self, event):
        resized_image = self.img1.resize((event.width, event.height))
        self.resized_tk = ImageTk.PhotoImage(resized_image)

        self.img1_canvas.delete("all")
        self.img1_canvas.create_image(0, 0, image=self.resized_tk, anchor="nw")

    def run_analysis(self):
        self.control_genotype = self.control_genotype_var.get()
        self.comparison_genotypes = [str(self.comparison_list.get(i)) for i in self.comparison_list.curselection()]
        self.log_message("    Running Stats")
        self.log_message(f"        Frame Steps: 30")
        self.log_message(f"        Stats Time Cut: {self.time_cut_start_entry.get()} - {self.time_cut_end_entry.get()}")
        self.log_message(f"        Control GT: {self.control_genotype}")
        self.log_message(f"        Comparison GT(s): {self.comparison_genotypes}")

        worked = {"male": False, "female": False}
        for gender in ["male", "female"]:
            try:
                self.sa[gender].input_comps(self.control_genotype, self.comparison_genotypes)
                worked[gender] = True
            except Exception:
                self.log_message("Control/Comparison Genotype not found for " + gender)

        messages1, messages2 = [], []
        if worked["male"]: messages1 = self.sa["male"].run_analysis(self.time_cut_start_entry.get(), self.time_cut_end_entry.get())
        if worked["female"]: messages2 = self.sa["female"].run_analysis(self.time_cut_start_entry.get(), self.time_cut_end_entry.get())

        for m in messages1 + messages2:
            self.log_message(m)
        self.change_image1(no_inc=True)
        self.img1_canvas.bind("<Configure>", self.stretch_image)
        self.next_button1.grid(row=7, column=6, ipadx=3, ipady=3, sticky="ne")
        self.next_button2.grid(row=7, column=6, ipadx=3, ipady=3, sticky="nw")

        # Display plot
        event = tk.Event()
        event.width = self.img1_canvas.winfo_width()
        event.height = self.img1_canvas.winfo_height()
        self.stretch_image(event)

        self.log_message()

    def return_to_menu(self):
        self.controller.frames["MenuFrame"].log_message("    Exiting Stats Frame")
        self.controller.show_frame("MenuFrame")

    def zip_folder(self):
        input_folder = f'{self.controller.experiment_path}/'
        output_zip_path = f'{self.controller.experiment_path}_ZIPPED.zip'

        if not os.path.exists(input_folder):
            self.controller.frames["MenuFrame"].log_message(f"Input folder '{input_folder}' does not exist.")
            return

        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            total_files = sum(len(files) for _, _, files in os.walk(input_folder))
            with tqdm(total=total_files, unit='file', desc='Zipping') as pbar:
                for root, dirs, files in os.walk(input_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, input_folder))
                        pbar.update(1)

        self.controller.frames["MenuFrame"].log_message(f"Folder '{input_folder}' has been zipped to '{output_zip_path}'.")

    def log_message(self, message=""):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message)
        self.log_area.insert(tk.END, "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')
        self.controller.frames["MenuFrame"].log_message(message)
        self.update_idletasks()
        print(message)

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
