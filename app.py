import tkinter as tk
from tkinter import ttk
from tkinter import *

from MenuFrame import MenuFrame
from StatsFrame import StatsFrame
from statistical_analysis import *

import os

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configure style
        self.title("Geotaxis Analysis App")
        self.geometry("1000x700")
        self.resizable(True, True)
        self.configure(background="#000000")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#d9d9d9")

        # Variables used
        self.experiment_name = ""
        self.experiment_dir = os.getcwd()
        self.experiment_path = os.path.join(self.experiment_dir, self.experiment_name)
        self.using_lamp = "Yes"
        self.frame_steps = 30
        self.stats_interval = [0, 9]
        self.entered = False

        self.genotype_options = []
        self.control_genotype = ""
        self.comparison_genotype = ""

        self.fin_geo = {}
        self.sa = None

        # Set up frames
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True, anchor="center")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MenuFrame, StatsFrame):
            frame = F(self.container, self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Display menu
        self.show_frame("MenuFrame")



    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(self.experiment_dir, self.experiment_name)

    def set_using_lamp(self, using_lamp):
        self.using_lamp = using_lamp

    def set_frame_steps(self, frame_steps):
        self.frame_steps = frame_steps

    def set_stats_interval(self, stats_interval):
        self.stats_interval = stats_interval

    def show_frame(self, page_name):
        if page_name == "MenuFrame":
            self.frames["MenuFrame"].log_message("\n*************")
            self.frames["MenuFrame"].log_message("Entering Menu")
            self.frames["MenuFrame"].log_message("*************\n")
            self.frames["MenuFrame"].log_message("Settings reset to default\n")
            self.frame_steps = 30
            self.stats_interval = [0, 9]
        if page_name == "StatsFrame":
            frame = StatsFrame(self.container, self)
            self.frames["StatsFrame"] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames["MenuFrame"].log_message("\n**************")
            self.frames["MenuFrame"].log_message("Entering Stats")
            self.frames["MenuFrame"].log_message("**************\n")

        self.frames[page_name].tkraise()