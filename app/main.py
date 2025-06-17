import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import psutil
from app.functions import (
    Graf_1,
    ROI_drawer_manual,
    dicom_image,
    tew_correction,
    premenovy_zakon,
    compute_time_differences,
    riu_fit,
    riu_uptace_fce,
    align_images,
    posunuti_image,
)
from datetime import datetime
import numpy as np
from scipy.special import lambertw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
import platform


class aplikace:
    def __init__(self, init_gui=True):
        if init_gui:
            self.root = tk.Tk()
            self.root.title("Dosithyroid - version 1.0")

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            if platform.system() == "Windows":
                # Omezíme velikost na max 1800x1000 (nebo kolik potřebuješ)
                self.window_width = min(screen_width - 100, 1800)
                self.window_height = min(screen_height - 100, 1000)
            else:
                # Na macOS používáme dynamickou velikost
                self.window_width = screen_width - 100
                self.window_height = screen_height - 100

            x_position, y_position = (
                (screen_width - self.window_width) // 2,
                (screen_height - self.window_height) // 2 - 30,
            )
            self.root.geometry(
                f"{self.window_width}x{self.window_height}+{x_position}+{y_position}"
            )

            ### spodni lista pro RAM usage a treba jeste neco
            self.bottom_bar = tk.Frame(self.root, height=20, bg="gray")
            self.bottom_bar.pack(side="bottom", fill="x")

            self.ram_label = tk.Label(
                self.bottom_bar,
                text="RAM Used: 0.00 GB / 0.00 GB",
                font=("Arial", 12),
                bg="gray",
                fg="white",
            )
            self.ram_label.pack(side="right", padx=10, pady=5)

            self.update_ram_usage(self.root, self.ram_label)

            ### styl
            self.style = ttk.Style()
            self.style.theme_use("clam")

            self.style.configure(
                "Treeview",
                background="#F9F9F9",
                foreground="black",
                rowheight=28,
                fieldbackground="#F4F4F4",
                font=("Arial", 14),
            )
            self.style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

            self.entry_style = {"bd": 1, "relief": "solid"}
            self.button_style = {"bd": 5, "relief": "raised", "highlightthickness": 2}
            self.text_window_style = {"bd": 1, "relief": "solid"}

            ### notebook a zalozky
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True)

            self.tab1 = tk.Frame(self.notebook)
            self.notebook.add(self.tab1, text="Image loading")

            self.tab2 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab2, text="Graph creation")

            self.tab3 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab3, text="Dose computation / Protocol making")

            self.tab4 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab4, text="Technical parameters")

            ### ZALOZKA 1 - IMAGE LOADING

            # container, nadpisy a LOAD
            self.image_container = tk.Frame(
                self.tab1, bd=5, relief="solid", padx=10, pady=5
            )
            self.image_container.pack(padx=20, pady=10)

            self.image_frame = tk.Frame(self.image_container)
            self.image_frame.pack()

            self.image_size = min(
                round(self.window_height / 3) - 60, round(self.window_width / 5) - 60
            )
            self.blank_image = Image.new(
                "RGB", (self.image_size, self.image_size), "white"
            )
            self.blank_image_tk = ImageTk.PhotoImage(self.blank_image)

            self.dicom_images = {}
            self.img_labels_ant = {}
            self.img_labels_pos = {}

            self.date_labels = {}
            self.time_labels = {}
            self.duration_labels = {}

            self.img_titles1 = [
                "Acquisition 1 h",
                "Acquisition 4-6 h",
                "Acquisition 24 h",
                "Acquisition 48 h",
                "Acquisition 144 h",
            ]

            for i, title in enumerate(self.img_titles1):
                # nadpisy
                tk.Label(self.image_frame, text=title, font=("Arial", 16, "bold")).grid(
                    row=0, column=i, pady=5
                )

                # obrazky ANT a POS
                self.img_labels_ant[i] = tk.Label(
                    self.image_frame,
                    image=self.blank_image_tk,
                    width=self.image_size,
                    height=self.image_size,
                )
                self.img_labels_ant[i].grid(row=1, column=i, padx=10, pady=10)
                self.img_labels_ant[i].image = self.blank_image_tk

                self.img_labels_pos[i] = tk.Label(
                    self.image_frame,
                    image=self.blank_image_tk,
                    width=self.image_size,
                    height=self.image_size,
                )
                self.img_labels_pos[i].grid(row=2, column=i, padx=10, pady=10)
                self.img_labels_pos[i].image = self.blank_image_tk

                # LOAD buttony
                tk.Button(
                    self.image_frame,
                    text="Load",
                    font=("Arial", 13, "bold"),
                    **self.button_style,
                    command=lambda i=i: self.safe_call(self.load_image, i),
                ).grid(row=3, column=i, pady=5)

                # Datumy, casy a doba akvizice
                self.date_labels[i] = tk.Label(
                    self.image_frame, text="Date: N/A", font=("Arial", 13)
                )
                self.date_labels[i].grid(row=4, column=i, pady=5)

                self.time_labels[i] = tk.Label(
                    self.image_frame, text="Time: N/A", font=("Arial", 13)
                )
                self.time_labels[i].grid(row=5, column=i, pady=5)

                self.duration_labels[i] = tk.Label(
                    self.image_frame, text="Duration: N/A", font=("Arial", 13)
                )
                self.duration_labels[i].grid(row=6, column=i, pady=5)

            ### buttony
            self.button_frame_1 = tk.Frame(self.tab1)
            self.button_frame_1.pack(padx=10, pady=0)

            # Korekce MD button
            self.cor_MD_button = tk.Button(
                self.button_frame_1,
                text="DT Correction",
                font=("Arial", 13, "bold"),
                height=2,
                **self.button_style,
                command=lambda: self.safe_call(self.DT_correction),
            )
            self.cor_MD_button.grid(row=0, column=0, padx=10)

            # align ANT button
            self.align_ant_button = tk.Button(
                self.button_frame_1,
                text="Align ANT",
                font=("Arial", 13, "bold"),
                height=2,
                **self.button_style,
                command=lambda: self.safe_call(self.align_ANT),
            )
            self.align_ant_button.grid(row=0, column=2, padx=(80, 10))

            # segment ANT button
            self.segment_ant_button = tk.Button(
                self.button_frame_1,
                text="Segment ANT",
                font=("Arial", 13, "bold"),
                height=2,
                **self.button_style,
                command=lambda: self.safe_call(self.segment_ANT),
            )
            self.segment_ant_button.grid(row=0, column=3, padx=10)

            # align POS button
            self.align_pos_button = tk.Button(
                self.button_frame_1,
                text="Align POS",
                font=("Arial", 13, "bold"),
                height=2,
                **self.button_style,
                command=lambda: self.safe_call(self.align_POS),
            )
            self.align_pos_button.grid(row=0, column=4, padx=(80, 10))

            # segment POS button
            self.segment_pos_button = tk.Button(
                self.button_frame_1,
                text="Segment POS",
                font=("Arial", 13, "bold"),
                height=2,
                **self.button_style,
                command=lambda: self.safe_call(self.segment_POS),
            )
            self.segment_pos_button.grid(row=0, column=5, padx=10)

            ### ZALOZKA 2 - GRAPH CREATION

            self.tab2_frame = tk.Frame(self.tab2)
            self.tab2_frame.pack(fill="both", expand=True)

            self.tab2_frame_left = tk.Frame(self.tab2_frame)
            self.tab2_frame_left.pack(side="left", anchor="nw")

            # vyber korekce
            self.radio_button_frame_1 = tk.Frame(self.tab2_frame_left)
            self.radio_button_frame_1.pack(anchor="nw", padx=30, pady=(30, 15))

            self.korekce_options = ["ACSC", "SC", "AC", "No corr"]
            self.typ_korekce = tk.StringVar(
                value=self.korekce_options[0]
            )  # default = "ACSC"

            tk.Label(
                self.radio_button_frame_1,
                text="Choose type of evaluation:",
                font=("Arial", 18, "bold"),
            ).pack(anchor="nw")

            self.korekce_combobox = ttk.Combobox(
                self.radio_button_frame_1,
                textvariable=self.typ_korekce,
                values=self.korekce_options,
                font=("Arial", 16),
                state="readonly",  # zabranuje rucnimu psani, jen vyber z listu
            )
            self.korekce_combobox.pack(anchor="nw", pady=5)

            # vyber objemu
            self.radio_button_frame_2 = tk.Frame(self.tab2_frame_left)
            self.radio_button_frame_2.pack(anchor="nw", padx=30, pady=15)

            self.sz_options = ["Whole thyroid gland", "Right lobe", "Left lobe", "Node"]
            self.sz_selected_option = tk.StringVar(
                value=self.sz_options[0]
            )  # default = "Whole thyroid gland"

            tk.Label(
                self.radio_button_frame_2,
                text="What type of thyroid evaluation:",
                font=("Arial", 18, "bold"),
            ).pack(anchor="nw")

            self.sz_combobox = ttk.Combobox(
                self.radio_button_frame_2,
                textvariable=self.sz_selected_option,
                values=self.sz_options,
                font=("Arial", 16),
                state="readonly",  # zabranuje rucnimu psani, jen vyber z listu
            )
            self.sz_combobox.pack(anchor="nw", pady=(5, 30))

            ## zapsani hodnoty aktivity a datumu
            # hlidani charakteru
            self.val_float = (self.root.register(self.validate_input_activity), "%P")
            self.val_date = (self.root.register(self.validate_date_input), "%P")

            # vytvoreni framu
            self.date_activity_frame = tk.Frame(self.tab2_frame_left)
            self.date_activity_frame.pack(anchor="nw", padx=30, pady=15)

            # zapis hodnoty aktivity
            tk.Label(
                self.date_activity_frame,
                text="Activity value (MBq):",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw")
            self.entry_activity = tk.Entry(
                self.date_activity_frame,
                font=("Arial", 14),
                validate="key",
                validatecommand=self.val_float,
                **self.entry_style,
            )
            self.entry_activity.pack(anchor="w", pady=5)
            self.entry_activity.insert(0, "0")
            self.entry_activity.bind("<KeyRelease>", self.update_administered_activity)

            # zapis datumu aktivity
            tk.Label(
                self.date_activity_frame,
                text="Date of activity (dd.mm.yyyy hh:mm):",
                font=("Arial", 16, "bold"),
            ).pack(anchor="w", pady=(10, 0))
            self.entry_date_activity = tk.Entry(
                self.date_activity_frame,
                font=("Arial", 14),
                validate="key",
                validatecommand=self.val_date,
                **self.entry_style,
            )
            self.entry_date_activity.pack(anchor="w", pady=5)
            self.entry_date_activity.insert(0, "0")
            self.entry_date_activity.bind(
                "<KeyRelease>", self.update_administered_activity
            )

            # zapis datumu podani
            tk.Label(
                self.date_activity_frame,
                text="Administration (dd.mm.yyyy hh:mm):",
                font=("Arial", 16, "bold"),
            ).pack(anchor="w", pady=(10, 0))
            self.entry_date_pacient = tk.Entry(
                self.date_activity_frame,
                font=("Arial", 14),
                validate="key",
                validatecommand=self.val_date,
                **self.entry_style,
            )
            self.entry_date_pacient.pack(anchor="w", pady=5)
            self.entry_date_pacient.insert(0, "0")
            self.entry_date_pacient.bind(
                "<KeyRelease>", self.update_administered_activity
            )

            # vypocet realne podanne aktivity
            tk.Label(
                self.date_activity_frame,
                text="Administered activity (MBq):",
                font=("Arial", 16, "bold"),
            ).pack(anchor="w", pady=(10, 0))
            self.entry_act_computed_value = tk.Entry(
                self.date_activity_frame,
                font=("Arial", 14),
                state="readonly",
                **self.entry_style,
            )
            self.entry_act_computed_value.pack(anchor="w", pady=5)
            self.entry_act_computed_value.insert(0, "0")  # Start with 0

            # frame pro SPECT uptake a evaluate button
            self.spect_and_evaluate_frame = tk.Frame(self.date_activity_frame)
            self.spect_and_evaluate_frame.pack(padx=10, pady=(50, 10), anchor="w")

            # title a entry pro SPECT
            self.spect_title_and_entry_frame = tk.Frame(self.spect_and_evaluate_frame)
            self.spect_title_and_entry_frame.grid(row=0, column=0, padx=(0, 10), pady=5)
            tk.Label(
                self.spect_title_and_entry_frame,
                text="SPECT 24h uptake (%):",
                font=("Arial", 14, "bold"),
            ).pack(anchor="w")
            self.spect_entry_value = tk.Entry(
                self.spect_title_and_entry_frame,
                font=("Arial", 14),
                validate="key",
                validatecommand=self.val_float,
                **self.entry_style,
            )
            self.spect_entry_value.pack(anchor="w", pady=5)
            self.spect_entry_value.insert(0, "0")

            # frame pro buttony
            self.evaluation_buttons_frame = tk.Frame(self.spect_and_evaluate_frame)
            self.evaluation_buttons_frame.grid(row=0, column=1, padx=(10, 0), pady=5)
            # button, ktery udela graf a vypocty ulozi
            self.evaluation_button = tk.Button(
                self.evaluation_buttons_frame,
                text="Evaluate",
                font=("Arial", 18, "bold"),
                height=1,
                width=10,
                command=lambda: self.safe_call(self.graph_evalueation),
                **self.button_style,
            )
            self.evaluation_button.pack(anchor="w", pady=5)

            # button, ktery udela graf a vypocty ulozi
            self.add_spect_button = tk.Button(
                self.evaluation_buttons_frame,
                text="Add SPECT",
                font=("Arial", 16, "bold"),
                height=1,
                width=10,
                command=lambda: self.safe_call(self.add_spect),
                **self.button_style,
            )
            self.add_spect_button.pack(anchor="center", pady=5)

            # **Right-side: Graph**
            self.graph_frame = tk.Frame(
                self.tab2_frame,
                width=self.window_width * 0.7,
                height=self.window_height * 0.7,
            )
            self.graph_frame.pack(anchor="e", padx=10, pady=10, expand=True)

            ### ZALOZKA 3 - DOSE COMPUTATION

            self.tab3_frame = tk.Frame(self.tab3)
            self.tab3_frame.pack(fill="both", expand=True)

            self.dose_computation_frame = tk.Frame(self.tab3_frame)
            self.dose_computation_frame.pack(fill="both", anchor="n", pady=30)

            # frame pro tabulku hmotnosti organu
            self.volume_frame = tk.Frame(self.dose_computation_frame)
            self.volume_frame.pack(side="left", anchor="nw", padx=30)

            tk.Label(
                self.volume_frame,
                text="Volume of interested organ (ml)",
                font=("Arial", 18, "bold"),
            ).pack(anchor="w", pady=(10, 0))
            self.volume_of_organ = tk.Entry(
                self.volume_frame,
                font=("Arial", 14),
                width=15,
                validate="key",
                validatecommand=self.val_float,
                **self.entry_style,
            )
            self.volume_of_organ.pack(anchor="n", pady=10)
            self.volume_of_organ.insert(0, "0")

            # computation button
            self.computation_button = tk.Button(
                self.volume_frame,
                anchor="center",
                text="Compute ACTIVITY/DOSE",
                height=2,
                width=25,
                font=("Arial", 18, "bold"),
                command=lambda: self.safe_call(self.compute_activity_and_dose),
                **self.button_style,
            )
            self.computation_button.pack(pady=30)

            # frame pro TIAC, Eff_halflife, podil_f, E, davku a nasledne planovaci davku
            self.results_frame_dose = tk.Frame(self.dose_computation_frame)
            self.results_frame_dose.pack(side="right", anchor="n", padx=30)

            # prvni tabulka (jakoby terap sekce)
            self.results_columns_1 = (
                "TIAC (days)",
                "f_[fitting] (%)",
                "T_[eff] (days)",
                "E ((Gy*gram)/(MBq*day))",
                "Dose (Gy)",
            )
            self.results_tree_dose_1 = ttk.Treeview(
                self.results_frame_dose,
                columns=self.results_columns_1,
                show="headings",
                height=1,
            )

            for col in self.results_columns_1:
                self.results_tree_dose_1.heading(col, text=col, anchor="center")
                col_width = len(col) * 10
                self.results_tree_dose_1.column(
                    col, anchor="center", minwidth=col_width
                )

            self.results_tree_dose_1.pack(fill="x", pady=10)

            # druha tabulka
            self.results_columns_2 = ("Dose (Gy)", "A_[ter] (MBq)")
            self.results_tree_dose_2 = ttk.Treeview(
                self.results_frame_dose,
                columns=self.results_columns_2,
                show="headings",
                height=6,
            )

            for col in self.results_columns_2:
                self.results_tree_dose_2.heading(col, text=col, anchor="center")

            self.results_tree_dose_2.pack(pady=10)

            # Vytvoreni protokolu
            self.osobni_informace_frame = tk.Frame(self.tab3_frame)
            self.osobni_informace_frame.pack(
                side="left", padx=30, pady=30, fill="both", expand=True
            )  # Ensure padding between elements

            # Jmeno a prijmeni
            tk.Label(
                self.osobni_informace_frame,
                text="Name and Surname:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw", pady=(10, 0))
            self.jmeno_a_prijmeni = tk.Entry(
                self.osobni_informace_frame,
                font=("Arial", 12),
                width=40,
                **self.entry_style,
            )
            self.jmeno_a_prijmeni.pack(anchor="nw", pady=5)

            # Datum narozeni
            tk.Label(
                self.osobni_informace_frame,
                text="Date of Birth:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw", pady=(10, 0))
            self.datum_narozeni = tk.Entry(
                self.osobni_informace_frame,
                font=("Arial", 12),
                width=40,
                **self.entry_style,
            )
            self.datum_narozeni.pack(anchor="nw", pady=5)

            # Typ zjisteni objemu SZ
            tk.Label(
                self.osobni_informace_frame,
                text="Type of volume evaluation:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw", pady=(10, 0))

            self.type_of_volume_options = ["CT", "Ultrasound", "MRI"]
            self.typ_zjisteni_objemu = tk.StringVar(
                value=self.type_of_volume_options[0]
            )  # default = "CT"

            self.typ_zjisteni_combobox = ttk.Combobox(
                self.osobni_informace_frame,
                textvariable=self.typ_zjisteni_objemu,
                values=self.type_of_volume_options,
                font=("Arial", 14),
                state="readonly",
            )
            self.typ_zjisteni_combobox.pack(anchor="nw", pady=5)

            # Datum vyhodnoceni objemu
            tk.Label(
                self.osobni_informace_frame,
                text="Date of the volume evaluation and by who:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw", pady=(10, 0))
            self.datum_vyhodnoceni_objemu = tk.Entry(
                self.osobni_informace_frame,
                font=("Arial", 12),
                width=40,
                **self.entry_style,
            )
            self.datum_vyhodnoceni_objemu.pack(anchor="nw", pady=5)
            self.datum_vyhodnoceni_objemu.insert(0, "01.01.0101, MUDr. Name Surname")

            # Typ protokolu
            tk.Label(
                self.osobni_informace_frame,
                text="Type of protocol:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw", pady=(10, 0))

            self.type_of_protocol_option = ["Planning", "Therapy"]
            self.typ_protokolu = tk.StringVar(
                value=self.type_of_protocol_option[0]
            )  # default = "Planning"

            # Frame pro usporadani comboboxu a vstupu vedle sebe
            protocol_selection_frame = tk.Frame(self.osobni_informace_frame)
            protocol_selection_frame.pack(anchor="nw", pady=5, fill="x")

            self.typ_protokolu_combobox = ttk.Combobox(
                protocol_selection_frame,
                textvariable=self.typ_protokolu,
                values=self.type_of_protocol_option,
                font=("Arial", 14),
                state="readonly",
                width=15,
            )
            self.typ_protokolu_combobox.grid(row=0, column=0, sticky="nw")

            # Frame pro 3 vstupy vpravo od comboboxu
            self.protocol_params_frame = tk.Frame(protocol_selection_frame)
            self.protocol_params_frame.grid(row=0, column=1, padx=20, sticky="w")

            # Popisky a vstupy pro k_t, k_T, k_B s validaci
            tk.Label(
                self.protocol_params_frame, text="k_t (1/h):", font=("Arial", 12)
            ).grid(row=0, column=0, sticky="e", padx=(0, 5))
            tk.Label(
                self.protocol_params_frame, text="k_T (1/h):", font=("Arial", 12)
            ).grid(row=1, column=0, sticky="e", padx=(0, 5))
            tk.Label(
                self.protocol_params_frame, text="k_B (1/h):", font=("Arial", 12)
            ).grid(row=2, column=0, sticky="e", padx=(0, 5))

            self.k_t_entry = tk.Entry(
                self.protocol_params_frame,
                font=("Arial", 12),
                width=10,
                validate="key",
                validatecommand=self.val_float,
            )
            self.k_T_entry = tk.Entry(
                self.protocol_params_frame,
                font=("Arial", 12),
                width=10,
                validate="key",
                validatecommand=self.val_float,
            )
            self.k_B_entry = tk.Entry(
                self.protocol_params_frame,
                font=("Arial", 12),
                width=10,
                validate="key",
                validatecommand=self.val_float,
            )

            self.k_t_entry.grid(row=0, column=1, pady=2)
            self.k_T_entry.grid(row=1, column=1, pady=2)
            self.k_B_entry.grid(row=2, column=1, pady=2)

            # Funkce pro skrytt/zobrazeni parametru podle vyberu v comboboxu
            def update_protocol_params_visibility(event=None):
                if self.typ_protokolu.get() == "Therapy":
                    self.protocol_params_frame.grid()
                else:
                    self.protocol_params_frame.grid_remove()

            # Bind udalost zmeny vyberu
            self.typ_protokolu_combobox.bind(
                "<<ComboboxSelected>>", update_protocol_params_visibility
            )

            # Inicializuj viditelnost podle vychozi hodnoty
            update_protocol_params_visibility()

            # zaver a vzexportovani protokolu
            self.zaver_a_export_frame = tk.Frame(self.tab3_frame)
            self.zaver_a_export_frame.pack(
                side="left", padx=(0, 30), pady=30, fill="both", expand=True
            )

            tk.Label(
                self.zaver_a_export_frame,
                text="Conclusion:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="nw", pady=(10, 0))
            self.zaver_window = tk.Text(
                self.zaver_a_export_frame,
                width=100,
                height=8,
                font=("Arial", 12),
                **self.text_window_style,
            )
            self.zaver_window.pack(anchor="w", pady=(10, 0))
            self.zaver_window.insert(
                tk.END, "This is a sample text window.\nYou can edit this content.\n"
            )

            # frame pro SPECT uptake a evaluate button
            self.jazyk_and_export_button = tk.Frame(self.zaver_a_export_frame)
            self.jazyk_and_export_button.pack(padx=10, pady=40, anchor="w")

            # CZ nebo ENG
            self.jazyk_frame = tk.Frame(self.jazyk_and_export_button)
            self.jazyk_frame.grid(row=0, column=0, padx=10, pady=5)

            tk.Label(
                self.jazyk_frame,
                text="Language of protocol:",
                font=("Arial", 16, "bold"),
            ).pack(anchor="w")

            self.jazyk_options = ["CZ", "EN"]
            self.jazyk_selected_option = tk.StringVar(
                value=self.jazyk_options[0]
            )  # default = "CZ"

            self.jazyk_combobox = ttk.Combobox(
                self.jazyk_frame,
                textvariable=self.jazyk_selected_option,
                values=self.jazyk_options,
                font=("Arial", 14),
                state="readonly",
            )
            self.jazyk_combobox.pack(anchor="w", pady=5)

            # Protokol button
            self.protocol_button = tk.Button(
                self.jazyk_and_export_button,
                anchor="center",
                text="Export a PDF protocol",
                height=2,
                width=25,
                font=("Arial", 18, "bold"),
                command=lambda: self.safe_call(self.protocol_export),
                **self.button_style,
            )
            self.protocol_button.grid(row=0, column=1, padx=10)

            ### ZALOZKA 4 - TECHNICKE PARAMETRY

            self.tab4_frame = tk.Frame(self.tab4)
            self.tab4_frame.pack(fill="both", expand=True)

            # MRTVA DOBA
            self.md_frame = tk.Frame(self.tab4_frame)
            self.md_frame.pack(side="left", padx=30, pady=(30, 15), anchor="nw")

            # Frame to hold the table
            self.md_data_table_frame = tk.Frame(self.md_frame)

            tk.Label(
                self.md_frame,
                text="Values of Dead Time of the camera",
                font=("Arial", 18, "bold"),
            ).pack(anchor="nw")

            # Radio buttons for table modes
            self.md_parameters_value = tk.IntVar()

            # Optima 640 - jen pro cteni
            self.rb1_md = tk.Radiobutton(
                self.md_frame,
                text="FNKV - GE Optima NM/CT 640",
                font=("Arial", 14),
                variable=self.md_parameters_value,
                value=1,
                command=lambda: self.safe_call(self.update_table_md_params),
            )
            self.rb1_md.pack(anchor="w")

            # Discovery 870 - jen pro cteni
            self.rb2_md = tk.Radiobutton(
                self.md_frame,
                text="FNKV - GE Discovery NM/CT 870 DR",
                font=("Arial", 14),
                variable=self.md_parameters_value,
                value=2,
                command=lambda: self.safe_call(self.update_table_md_params),
            )
            self.rb2_md.pack(anchor="w")

            # vlastni hodnoty mrtve doby - zapis hodnot
            self.rb3_md = tk.Radiobutton(
                self.md_frame,
                text="Individual Dead Time parameters",
                font=("Arial", 14),
                variable=self.md_parameters_value,
                value=3,
                command=lambda: self.safe_call(self.update_table_md_params),
            )
            self.rb3_md.pack(anchor="w")

            self.md_data_table_frame.pack(anchor="nw", pady=15)

            # deafultne nastaveno pro Optimu 640
            self.md_parameters_value.set(1)
            self.update_table_md_params()

            # KALIBRACNI FAKTORY
            self.kal_frame = tk.Frame(self.tab4_frame)
            self.kal_frame.pack(side="top", padx=70, pady=(30, 15), anchor="nw")

            # Frame pro tabulku
            self.kal_data_table_frame = tk.Frame(self.kal_frame)

            tk.Label(
                self.kal_frame,
                text="Values of Calibration Factors of the camera",
                font=("Arial", 18, "bold"),
            ).pack(side="top", anchor="nw")

            # Radio buttony pro tabulku
            self.kal_parameters_value = tk.IntVar()

            # Optima 640 - jen pro cteni
            self.rb1_kal = tk.Radiobutton(
                self.kal_frame,
                text="FNKV - GE Optima NM/CT 640",
                font=("Arial", 14),
                variable=self.kal_parameters_value,
                value=1,
                command=lambda: self.safe_call(self.update_table_kal_params),
            )
            self.rb1_kal.pack(anchor="w")

            # Discovery 870 - jen pro cteni
            self.rb2_kal = tk.Radiobutton(
                self.kal_frame,
                text="FNKV - GE Discovery NM/CT 870 DR",
                font=("Arial", 14),
                variable=self.kal_parameters_value,
                value=2,
                command=lambda: self.safe_call(self.update_table_kal_params),
            )
            self.rb2_kal.pack(anchor="w")

            # vlastni hodnoty mrtve doby - zapis hodnot
            self.rb3_kal = tk.Radiobutton(
                self.kal_frame,
                text="Individual Dead Time parameters",
                font=("Arial", 14),
                variable=self.kal_parameters_value,
                value=3,
                command=lambda: self.safe_call(self.update_table_kal_params),
            )
            self.rb3_kal.pack(anchor="w")

            self.kal_data_table_frame.pack(pady=15)

            # deafultne nastaveno pro Optimu 640
            self.kal_parameters_value.set(1)
            self.update_table_kal_params()

            self.provedeni_korekce_MD = False
            self.evaluace_grafu = False

        else:
            self.root = None

    ### --------------------------------------------------------------
    ### podpurne FUNKCE

    ## funkce pro safe_call - vyhodi messagebox
    def safe_call(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            messagebox.showerror("Error", f"Error in {func.__name__}: {e}")
            return None

    ## funkce, ktera stale zobrazuje vyuziti RAM
    def update_ram_usage(self, root, ram_label):
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)  # prevod na GB
        ram_total = ram.total / (1024**3)  # prevod na GB

        new_text = f"RAM Used: {ram_used:.2f} GB / {ram_total:.2f} GB"
        if ram_label["text"] != new_text:
            ram_label.config(text=new_text)  # pdate jen pokud se lisi

        root.after(
            1000, self.update_ram_usage, root, ram_label
        )  # update kazdou sekundu

    ### --------------------------------------------------------------

    ### --------------------------------------------------------------
    ### FUNKCE 1. záložky

    # funkce pro tlacitko Load
    def load_image(self, index):
        # Otevre dialog pro vyber souboru DICOM (*.dcm)
        file_path = filedialog.askopenfilename(filetypes=[("DICOM Files", "*.dcm")])
        if file_path:
            try:
                # Ulozi cestu k adresari, kde je vybrany soubor
                self.folder_path = os.path.dirname(file_path)

                # Vytvori vystupni slozku "dosithyroid_output" v teto slozce, pokud jeste neexistuje
                self.output_folder = os.path.join(
                    self.folder_path, "dosithyroid_output"
                )
                os.makedirs(self.output_folder, exist_ok=True)

                # Vytvori instanci tridy dicom_image a nacte DICOM soubor do slovniku s klicem 'index'
                self.dicom_images[index] = dicom_image()
                self.dicom_images[index].load_dicom(file_path)

                # Prevede obraz 'ant_pw' na PIL obrazek pro zobrazeni
                ant_pw_image = self.dicom_images[index].convert_to_image("ant_pw")
                # Zmeni velikost obrazku na pozadovane rozmery
                ant_pw_image_resized = ant_pw_image.resize(
                    (self.image_size, self.image_size)
                )
                # Prevede PIL obrazek na Tkinter kompatibilni obrazek
                ant_pw_image_tk = ImageTk.PhotoImage(ant_pw_image_resized)

                # Aktualizuje label, aby zobrazil novy obrazek
                self.img_labels_ant[index].config(image=ant_pw_image_tk)
                # Udrzuje referenci na obrazek, aby nedoslo k jeho odstraneni
                self.img_labels_ant[index].image = ant_pw_image_tk

                # Stejne provede pro obraz 'pos_pw' (pokud je potreba zobrazit i ten)
                pos_pw_image = self.dicom_images[index].convert_to_image("pos_pw")
                pos_pw_image_resized = pos_pw_image.resize(
                    (self.image_size, self.image_size)
                )
                pos_pw_image_tk = ImageTk.PhotoImage(pos_pw_image_resized)

                self.img_labels_pos[index].config(image=pos_pw_image_tk)
                self.img_labels_pos[index].image = pos_pw_image_tk

                # Prevede cas akvizice z formatu "093108.00" na "09:31:08"
                acq_time = self.dicom_images[index].acq_time
                acq_time_formatted = f"{acq_time[:2]}:{acq_time[2:4]}:{acq_time[4:6]}"

                # Prevede datum akvizice z formatu "20250212" na "12.02.2025"
                acq_date = self.dicom_images[index].acq_date
                acq_date_formatted = datetime.strptime(acq_date, "%Y%m%d").strftime(
                    "%d.%m.%Y"
                )

                # Aktualizuje textove labely s datumem, casem a dobou trvani akvizice
                self.date_labels[index].config(text=f"Date: {acq_date_formatted}")
                self.time_labels[index].config(text=f"Time: {acq_time_formatted}")
                self.duration_labels[index].config(
                    text=f"Duration: {self.dicom_images[index].acq_dur:.2f} seconds"
                )

            except Exception as e:
                # Pokud nastane chyba pri nacitani, vypise ji a znovu vyhodi vyjimku
                print(
                    f"Error loading DICOM image for index {index} in by Load Button: {e}"
                )
                raise Exception(
                    f"Error loading DICOM image for index {index} in by Load Button: {e}"
                )

    def update_image_labels(
        self, index, img_labels_ant, img_labels_pos, image_size, type
    ):
        """
        Aktualizuje obrazkove labely s novymi zarovnanými obrazky.
        """
        try:
            if type == "ant":
                # Aktualizuje obraz 'ant_pw'
                ant_pw_image = self.dicom_images[index].convert_to_image("ant_pw")
                # Zmeni velikost obrazku na pozadovanou
                ant_pw_image_resized = ant_pw_image.resize((image_size, image_size))
                # Prevede PIL obrazek na Tkinter obrazek
                ant_pw_image_tk = ImageTk.PhotoImage(ant_pw_image_resized)
                # Nastavi obrazek do prislusneho labelu
                img_labels_ant[index].config(image=ant_pw_image_tk)
                # Udrzuje referenci na obrazek, aby se neztratil
                img_labels_ant[index].image = ant_pw_image_tk

            else:
                # Aktualizuje obraz 'pos_pw'
                pos_pw_image = self.dicom_images[index].convert_to_image("pos_pw")
                pos_pw_image_resized = pos_pw_image.resize((image_size, image_size))
                pos_pw_image_tk = ImageTk.PhotoImage(pos_pw_image_resized)
                img_labels_pos[index].config(image=pos_pw_image_tk)
                img_labels_pos[
                    index
                ].image = pos_pw_image_tk  # Udrzuje referenci na obrazek

        except Exception as e:
            # Pokud nastane chyba, vypise ji a znovu vyhodi vyjimku
            print(f"Error updating image labels: {e}")
            raise Exception(f"Error updating image labels: {e}")

    # funkce tlaticka DT correction
    def DT_correction(self):
        try:
            # Zkontroluje, zda uz byla korekce provedena
            if self.provedeni_korekce_MD:
                print("Correction has already been applied.")
                return  # Pokud byla korekce uz provedena, funkce se ukonci

            # Slovnik pro ulozeni namerenych a teoretickych cetnosti
            merena_cetnost = {}
            teoreticka_cetnost = {}
            kor_faktor = {}

            # Projde vsechny dicom obrazky a aplikuje korekci
            with open(
                os.path.join(self.output_folder, "DT_correction_params.txt"), "w"
            ) as dt_file:
                # Zapise informace o pouzite korekci do souboru
                dt_file.write(
                    "Correction applied by Lambert W function:\n"
                    "R_corr = -REAL(W(-R_m * tau)) / tau\n"
                    "Python: corrected_rate = -np.real(lambertw(-measured_rate * dead_time, k=0)) / dead_time\n"
                    "----> Correction factor = corrected_rate / measured_rate\n"
                    "-----------------------------------------------------------------------------------------\n\n"
                )

                # Pro kazdy index v slovniku dicom obrazku
                for index in self.dicom_images.keys():
                    # Pro kazdy klic v md_data (typy dat, napr. ant_pw, pos_pw)
                    for key in self.md_data.keys():
                        try:
                            # Vypocita namerenou cetnost jako soucet pixelu deleno dobou akvizice
                            merena_cetnost[key] = np.sum(
                                getattr(self.dicom_images[index], key)
                            ) / getattr(self.dicom_images[index], "acq_dur")
                            # Zapise namerenou cetnost do souboru
                            dt_file.write(
                                f"Measured rate for {key} for index {index}: {merena_cetnost[key]} cps\n"
                            )

                            # Vypocita teoretickou cetnost pomoci Lambert W funkce pro korekci mrtve
                            teoreticka_cetnost[key] = (
                                -np.real(
                                    lambertw(
                                        -merena_cetnost[key] * self.md_data[key], k=0
                                    )
                                )
                                / self.md_data[key]
                            )

                            # Vypocita korekcni faktor jako pomer teoreticke a namerene cetnosti
                            kor_faktor[key] = (
                                teoreticka_cetnost[key] / merena_cetnost[key]
                            )
                            # Zapise korekcni faktor do souboru
                            dt_file.write(
                                f"Correction factor for {key} for index {index}: {kor_faktor[key]}\n\n"
                            )

                            # Aplikuje korekcni faktor na data v dicom obrazku
                            setattr(
                                self.dicom_images[index],
                                key,
                                getattr(self.dicom_images[index], key)
                                * kor_faktor[key],
                            )

                            # Podle klice aktualizuje prislusny obrazkovy label (predni nebo zadni obrazek)
                            if key == "ant_pw":
                                self.safe_call(
                                    self.update_image_labels,
                                    index,
                                    self.img_labels_ant,
                                    self.img_labels_pos,
                                    self.image_size,
                                    "ant",
                                )
                            if key == "pos_pw":
                                self.safe_call(
                                    self.update_image_labels,
                                    index,
                                    self.img_labels_ant,
                                    self.img_labels_pos,
                                    self.image_size,
                                    "pos",
                                )

                        except Exception as e:
                            # Pokud nastane chyba u zpracovani daneho klice, vypise a znovu vyhodi vyjimku
                            print(f"Error processing {key} for index {index}: {e}")
                            raise Exception(
                                f"Error processing {key} for index {index}: {e}"
                            )
                    # Oddelovac mezi zaznamy v souboru
                    dt_file.write("----\n\n")

            # Po uspesnem provedeni korekce nastavi priznak, ze korekce byla provedena
            self.provedeni_korekce_MD = True
            print("Correction applied successfully.")

        except Exception as e:
            # Pokud nastane neocekavana chyba, vypise a znovu vyhodi vyjimku
            print(f"Unexpected error in korekce_MD: {e}")
            raise Exception(f"Unexpected error in korekce_MD: {e}")

    # funkce tlacitka align ANT
    def align_ANT(self):
        try:
            # Nastavi referencni obrazek jako ant_pw obrazek s indexem 2
            reference = self.dicom_images[2].ant_pw
            # Projde vsechny klice ve slovniku dicom obrazku
            for key in self.dicom_images.keys():
                # Zarovna aktualni ant_pw obrazek na referencni, vrati zarovnany obrazek a posuny v x a y
                self.dicom_images[key].ant_pw, x_shift, y_shift = align_images(
                    reference, self.dicom_images[key].ant_pw
                )
                # Posune dalsi obrazky ant_usw a ant_lsw o stejny posun, aby zustaly zarovnane
                self.dicom_images[key].ant_usw = posunuti_image(
                    self.dicom_images[key].ant_usw, x_shift, y_shift
                )
                self.dicom_images[key].ant_lsw = posunuti_image(
                    self.dicom_images[key].ant_lsw, x_shift, y_shift
                )

                # Aktualizuje obrazkove labely v GUI pro ant obrazky
                self.update_image_labels(
                    key,
                    self.img_labels_ant,
                    self.img_labels_pos,
                    self.image_size,
                    "ant",
                )

        except Exception as e:
            # Pri chybe vypise hlasku a vyhodi vyjimku
            print(f"Error aligning anterior images: {e}")
            raise Exception(f"Error aligning anterior images: {e}")

    # funkce tlacitka align POS
    def align_POS(self):
        try:
            # Nastavi referencni obrazek jako pos_pw obrazek s indexem 2
            reference = self.dicom_images[2].pos_pw
            # Projde vsechny klice ve slovniku dicom obrazku
            for key in self.dicom_images.keys():
                # Zarovna aktualni pos_pw obrazek na referencni, vrati zarovnany obrazek a posuny v x a y
                self.dicom_images[key].pos_pw, x_shift, y_shift = align_images(
                    reference, self.dicom_images[key].pos_pw
                )
                # Posune dalsi obrazky pos_usw a pos_lsw o stejny posun, aby zustaly zarovnane
                self.dicom_images[key].pos_usw = posunuti_image(
                    self.dicom_images[key].pos_usw, x_shift, y_shift
                )
                self.dicom_images[key].pos_lsw = posunuti_image(
                    self.dicom_images[key].pos_lsw, x_shift, y_shift
                )

                # Aktualizuje obrazkove labely v GUI pro pos obrazky
                self.update_image_labels(
                    key,
                    self.img_labels_ant,
                    self.img_labels_pos,
                    self.image_size,
                    "pos",
                )

        except Exception as e:
            # Pri chybe vypise hlasku a vyhodi vyjimku
            print(f"Error aligning posterior images: {e}")
            raise Exception(f"Error aligning posterior images: {e}")

    # funkce tlacitka segment ANT
    def segment_ANT(self):
        # Spusti manualni segmentaci na 24hodinovem anteriornim obrazku a aplikuje na vsechny anteriorni obrazky
        try:
            # Zkontroluje, zda je nacten obrazek s indexem 2
            if 2 in self.dicom_images:
                # Vytvori instanci ROI_drawer_manual pro ant_pw obrazky a zobrazi ji (spusti GUI segmentaci)
                roi_drawer = ROI_drawer_manual(
                    self.dicom_images, "ant_pw", self.img_labels_ant, self.image_size
                )
                roi_drawer.show()
            else:
                # Pokud obrazek s indexem 2 neni, vypise chybu a vyhodi vyjimku
                print("Error: No DICOM image loaded for the 24h timepoint.")
                raise Exception("No DICOM image loaded for the 24h timepoint.")
        except Exception as e:
            # Pri chybe vypise hlasku a vyhodi vyjimku
            print(f"Error starting manual segmentation for ANT: {e}")
            raise Exception(f"Error starting manual segmentation for ANT. {e}")

    # funkce tlacitka segment POS
    def segment_POS(self):
        # Spusti manualni segmentaci na 24hodinovem posteriornim obrazku a aplikuje na vsechny posteriorni obrazky
        try:
            # Zkontroluje, zda je nacten obrazek s indexem 2
            if 2 in self.dicom_images:
                # Vytvori instanci ROI_drawer_manual pro pos_pw obrazky a zobrazi ji (spusti GUI segmentaci)
                roi_drawer = ROI_drawer_manual(
                    self.dicom_images, "pos_pw", self.img_labels_pos, self.image_size
                )
                roi_drawer.show()
            else:
                # Pokud obrazek s indexem 2 neni, vypise chybu a vyhodi vyjimku
                print("Error: No DICOM image loaded for the 24h timepoint.")
                raise Exception("No DICOM image loaded for the 24h timepoint.")
        except Exception as e:
            # Pri chybe vypise hlasku a vyhodi vyjimku
            print(f"Error starting manual segmentation for POS: {e}")
            raise Exception(f"Error starting manual segmentation for POS. {e}")

    ### --------------------------------------------------------------

    ### --------------------------------------------------------------
    ### FUNKCE 2. zalozky

    # Validacni funkce pro vstup aktivity
    def validate_input_activity(self, P):
        # Povoli prazdny retezec (na zacatku) nebo platna cisla s maximalne jednou teckou
        if P == "" or P.count(".") <= 1 and all(c.isdigit() or c == "." for c in P):
            return True
        else:
            return False

    # Validacni funkce pro vstup data
    def validate_date_input(self, P):
        # Povoli prazdny retezec (na zacatku) nebo platna cisla s maximalne 2 teckami, 1 dvojteckou a 1 mezerou
        if P == "":
            return True  # Povoli, aby pole bylo prazdne

        dot_count = P.count(".")
        colon_count = P.count(":")
        space_count = P.count(" ")

        # Overi, ze je nejvyse 2 tecky, 1 mezera a 1 dvojtecka a ze vsechny znaky jsou cisla, tecky, dvojtecky nebo mezery
        if (
            dot_count <= 2
            and space_count <= 1
            and colon_count <= 1
            and all(c.isdigit() or c == "." or c == ":" or c == " " for c in P)
        ):
            return True
        else:
            return False

    # Update funkce pro vypocitani aktivity
    def update_administered_activity(self, event=None):
        try:
            # Vymazani aktualni hodnoty v poli pro vypocitany udaj
            self.entry_act_computed_value.config(state="normal")
            self.entry_act_computed_value.delete(0, tk.END)

            # Nacteni hodnot z trid vstupu
            activity_str = self.entry_activity.get()
            date_activity_str = self.entry_date_activity.get()
            date_patient_str = self.entry_date_pacient.get()

            # Kontrola, jestli jsou vsechny policka dostatecne vyplnena pro vypocet
            if (
                activity_str
                and activity_str != "0"
                and date_activity_str
                and date_activity_str != "0"
                and len(date_activity_str) >= 16
                and date_patient_str
                and date_patient_str != "0"
                and len(date_patient_str) >= 16
            ):
                try:
                    # Prevod aktivity na desetinne cislo
                    activity = float(activity_str)

                    # Prevod datumu ze stringu na datetime objekty
                    date_format = "%d.%m.%Y %H:%M"
                    reference_time = datetime.strptime(date_activity_str, date_format)
                    nynejsi_time = datetime.strptime(date_patient_str, date_format)

                    # Vypocet podle predane funkce s korektnim typem argumentu
                    result = self.safe_call(
                        premenovy_zakon, activity, reference_time, nynejsi_time
                    )

                    # Kontrola, zda vypocet neskoncil None a vlozeni vysledku do pole
                    if result is not None:
                        self.entry_act_computed_value.insert(0, f"{result:.2f}")
                    else:
                        self.entry_act_computed_value.insert(
                            0, "Calculation returned None"
                        )

                except ValueError as e:
                    # Chyba pri prevodu datumu
                    print(f"Date format error: {e}")
                    self.entry_act_computed_value.insert(0, "Invalid date format")

                except Exception as e:
                    # Obecna chyba pri vypoctu
                    print(f"Calculation error: {e}")
                    self.entry_act_computed_value.insert(0, "Calculation error")

            else:
                # Nektera pole nejsou jeste validne vyplnena, cekani na vstup
                self.entry_act_computed_value.insert(0, "Waiting for input...")

            # Zpetne nastaveni pole jako pouze pro cteni
            self.entry_act_computed_value.config(state="readonly")

        except Exception as e:
            # Obecne osetreni chyby, vymazani a zobrazeni chyby v poli
            print(f"Error updating administered activity: {e}")
            self.entry_act_computed_value.config(state="normal")
            self.entry_act_computed_value.delete(0, tk.END)
            self.entry_act_computed_value.insert(0, "Error")
            self.entry_act_computed_value.config(state="readonly")

    # funkce pro tlacitko evaluate
    def graph_evalueation(self):
        # Vymazani predchoziho grafu, pokud existuje
        if self.evaluace_grafu:
            for widget in self.graph_frame.winfo_children():
                widget.destroy()  # odstraneni predchozich widgetu grafu

        self.evaluace_grafu = True
        # Inicializace noveho grafu s nastavenymi parametry
        self.graph = Graf_1(
            fontsize=10,
            title="",
            xlabel="Čas (h)",
            ylabel="Uptake aktivity (%)",
            figsize=(10, 6),
            dpi=round(self.window_height * 0.15),
        )

        try:
            # Nacteni hodnoty podane aktivity z UI a konverze na float
            self.podana_aktivita = float(self.entry_act_computed_value.get())
        except Exception as e:
            raise Exception(f"Error calculating podana_aktivita: {e}")

        try:
            # Ziskani vybrane volby pro oblast vyhodnoceni
            option_sz = self.sz_selected_option.get()
            # Nastaveni titulku grafu podle vybrane volby
            if option_sz == "Whole thyroid gland":
                title = "Uptake aktivity v celé ŠŽ"
            elif option_sz == "Right lobe":
                title = "Uptake aktivity v pravém laloku ŠŽ"
            elif option_sz == "Left lobe":
                title = "Uptake aktivity v levém laloku ŠŽ"
            else:
                title = "Uptake aktivity v hyperfunkčním uzlu ŠŽ"

            self.graph.fig.set_title(title)

        except Exception as e:
            print(f"Error setting graph title: {e}")
            raise Exception(f"Error setting graph title: {e}")

        # Inicializace seznamu pro ulozeni cetnosti a hodnot
        cetnosti_ant_pw = []
        cetnosti_ant_usw = []
        cetnosti_ant_lsw = []

        cetnosti_pos_pw = []
        cetnosti_pos_usw = []
        cetnosti_pos_lsw = []

        hodnoty_ant = []
        hodnoty_pos = []
        datumy = []
        casy = []

        try:
            # Ziskani typu korekce z UI
            option_corr = self.typ_korekce.get()
            if option_corr == "ACSC":
                # Prochazeni vsech DICOM snimku a vypocet korekce s ACSC
                for index in self.dicom_images.keys():
                    doba_akv = self.dicom_images[index].acq_dur
                    ant_roi = self.dicom_images[index].ant_roi
                    pos_roi = self.dicom_images[index].pos_roi

                    datumy.append(self.dicom_images[index].acq_date)
                    casy.append(self.dicom_images[index].acq_time)

                    # Vypocet cetnosti s ROI a normalizace na dobu akvizice
                    ant_pw = self.dicom_images[index].ant_pw * ant_roi
                    cetnosti_ant_pw.append(ant_pw.sum() / doba_akv)
                    ant_usw = self.dicom_images[index].ant_usw * ant_roi
                    cetnosti_ant_usw.append(ant_usw.sum() / doba_akv)
                    ant_lsw = self.dicom_images[index].ant_lsw * ant_roi
                    cetnosti_ant_lsw.append(ant_lsw.sum() / doba_akv)

                    pos_pw = self.dicom_images[index].pos_pw * pos_roi
                    cetnosti_pos_pw.append(pos_pw.sum() / doba_akv)
                    pos_usw = self.dicom_images[index].pos_usw * pos_roi
                    cetnosti_pos_usw.append(pos_usw.sum() / doba_akv)
                    pos_lsw = self.dicom_images[index].pos_lsw * pos_roi
                    cetnosti_pos_lsw.append(pos_lsw.sum() / doba_akv)

                # Aplikace tew korekce na cetnosti
                hodnoty_ant = tew_correction(
                    np.array(cetnosti_ant_pw),
                    np.array(cetnosti_ant_usw),
                    np.array(cetnosti_ant_lsw),
                )[0]
                hodnoty_pos = tew_correction(
                    np.array(cetnosti_pos_pw),
                    np.array(cetnosti_pos_usw),
                    np.array(cetnosti_pos_lsw),
                )[0]

                # Vypocet celkove uptake hodnoty a normalizace na podanou aktivitu
                hodnoty = (
                    np.sqrt(np.array(hodnoty_ant) * np.array(hodnoty_pos))
                    / self.kal_data[option_corr]
                )
                uptake_array = hodnoty / self.podana_aktivita

            elif option_corr == "SC":
                # Vypocet s korekci SC - pouze anteriorni snimky
                for index in self.dicom_images.keys():
                    doba_akv = self.dicom_images[index].acq_dur
                    ant_roi = self.dicom_images[index].ant_roi

                    datumy.append(self.dicom_images[index].acq_date)
                    casy.append(self.dicom_images[index].acq_time)

                    ant_pw = self.dicom_images[index].ant_pw * ant_roi
                    cetnosti_ant_pw.append(ant_pw.sum() / doba_akv)
                    ant_usw = self.dicom_images[index].ant_usw * ant_roi
                    cetnosti_ant_usw.append(ant_usw.sum() / doba_akv)
                    ant_lsw = self.dicom_images[index].ant_lsw * ant_roi
                    cetnosti_ant_lsw.append(ant_lsw.sum() / doba_akv)

                hodnoty_ant = tew_correction(
                    np.array(cetnosti_ant_pw),
                    np.array(cetnosti_ant_usw),
                    np.array(cetnosti_ant_lsw),
                )[0]
                hodnoty = hodnoty_ant / self.kal_data[option_corr]
                uptake_array = hodnoty / self.podana_aktivita

            elif option_corr == "AC":
                # Korekce AC - jen anteriorni a posteriorni pw snimky
                for index in self.dicom_images.keys():
                    doba_akv = self.dicom_images[index].acq_dur
                    ant_roi = self.dicom_images[index].ant_roi
                    pos_roi = self.dicom_images[index].pos_roi

                    datumy.append(self.dicom_images[index].acq_date)
                    casy.append(self.dicom_images[index].acq_time)

                    ant_pw = self.dicom_images[index].ant_pw * ant_roi
                    cetnosti_ant_pw.append(ant_pw.sum() / doba_akv)

                    pos_pw = self.dicom_images[index].pos_pw * pos_roi
                    cetnosti_pos_pw.append(pos_pw.sum() / doba_akv)

                hodnoty = (
                    np.sqrt(np.array(cetnosti_ant_pw) * np.array(cetnosti_pos_pw))
                    / self.kal_data[option_corr]
                )
                uptake_array = hodnoty / self.podana_aktivita

            else:  # Bez korekce
                for index in self.dicom_images.keys():
                    doba_akv = self.dicom_images[index].acq_dur
                    ant_roi = self.dicom_images[index].ant_roi

                    datumy.append(self.dicom_images[index].acq_date)
                    casy.append(self.dicom_images[index].acq_time)

                    ant_pw = self.dicom_images[index].ant_pw * ant_roi
                    cetnosti_ant_pw.append(ant_pw.sum() / doba_akv)

                hodnoty = np.array(cetnosti_ant_pw) / self.kal_data[option_corr]
                uptake_array = hodnoty / self.podana_aktivita

        except Exception as e:
            print(f"Error processing DICOM images in calculation: {e}")
            raise Exception(f"Error processing DICOM images in calculation: {e}")

        # Prevod uptake hodnot do slovniku podle indexu snimku
        self.uptake = {}
        self.time_differencies = {}

        for idx, index in enumerate(self.dicom_images.keys()):
            self.uptake[index] = uptake_array[idx]
            # Vypocet casovych rozdilu mezi datumem administrace aktivity pacientovi a akvizicemi
            self.time_differencies[index] = compute_time_differences(
                self.entry_date_pacient.get(), [datumy[idx]], [casy[idx]]
            )[0]

        # Vypsani dulezitych informaci
        print(
            "Time differencies:",
            {i: round(float(t), 2) for i, t in self.time_differencies.items()},
        )
        print("Uptake:", self.uptake)

        self.times_for_graph = np.array(list(self.time_differencies.values()))
        self.uptake_for_graph = np.array(list(self.uptake.values()))
        # Fit parametru pro riu funkci
        self.riu_params, riu_params_err, riu_params_covar = riu_fit(
            [self.times_for_graph, self.uptake_for_graph], y_err=None
        )

        try:
            # Vytvoreni pole casu pro vykresleni fitu
            self.time_diff_linspace = np.linspace(
                0, self.times_for_graph[-1] + 150, 100
            )

            # Vykresleni fitu a namerenych dat do grafu
            self.graph.plot(
                self.time_diff_linspace,
                riu_uptace_fce(self.time_diff_linspace, *self.riu_params) * 100,
                "-",
                "Proklad dat",
                "orange",
                1,
                1,
            )
            self.graph.plot(
                self.times_for_graph,
                self.uptake_for_graph * 100,
                "o",
                "Naměřené hodnoty",
                "blue",
                3,
                6,
            )
            # Ulozeni grafu do souboru
            self.graph.Figure.savefig(
                os.path.join(self.output_folder, "Graph.png"), bbox_inches="tight"
            )

            # Vytvoreni canvasu pro Tkinter a zobrazeni grafu v GUI
            canvas = FigureCanvasTkAgg(self.graph.Figure, master=self.graph_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, anchor="center")
            plt.close()

        except Exception as e:
            print(f"Error displaying graph in GUI: {e}")
            raise Exception(f"Error displaying graph in GUI: {e}")

    def add_spect(self):
        # Vymazani existujiciho grafu, pokud uz byl vytvoren
        if self.evaluace_grafu:
            for widget in self.graph_frame.winfo_children():
                widget.destroy()  # odstraneni vsech widgetu grafu v ramci frame

        self.evaluace_grafu = True
        # Inicializace noveho grafu s danymi parametry
        self.graph = Graf_1(
            fontsize=10,
            title="",
            xlabel="Čas (h)",
            ylabel="Uptake aktivity (%)",
            figsize=(10, 6),
            dpi=round(self.window_height * 0.15),
        )

        try:
            # Ziskani vybrane moznosti oblasti z UI
            option_sz = self.sz_selected_option.get()
            # Nastaveni titulku grafu podle volby uzivatele
            if option_sz == "Whole thyroid gland":
                title = "Uptake aktivity v celé ŠŽ"
            elif option_sz == "Right lobe":
                title = "Uptake aktivity v pravém laloku ŠŽ"
            elif option_sz == "Left lobe":
                title = "Uptake aktivity v levém laloku ŠŽ"
            else:
                title = "Uptake aktivity v hyperfunkčním uzlu ŠŽ"

            self.graph.fig.set_title(title)

        except Exception as e:
            print(f"Error setting graph title: {e}")
            raise Exception(f"Error setting graph title: {e}")

        # Nacteni hodnoty uptake ze SPECT (prevedeno z procent na cislo)
        spect_uptake = 0.01 * float(self.spect_entry_value.get())
        # Vypocet pomeru mezi uptake SPECT a modelem riu v danem case (index 2 - 24h)
        self.pomer = spect_uptake / riu_uptace_fce(
            self.time_differencies[2], *self.riu_params
        )

        # Vykresleni prokladu dat (fit) bez korekce
        self.graph.plot(
            self.time_diff_linspace,
            riu_uptace_fce(self.time_diff_linspace, *self.riu_params) * 100,
            "-",
            "Proklad dat",
            "orange",
            1,
            1,
        )
        # Vykresleni prokladu dat s korekci pomerem SPECT uptake
        self.graph.plot(
            self.time_diff_linspace,
            riu_uptace_fce(self.time_diff_linspace, *self.riu_params)
            * 100
            * self.pomer,
            "--",
            "Proklad dat s korekcí na SPECT",
            "orange",
            1,
            1,
        )

        # Vykresleni namerenych hodnot jako modre body
        self.graph.plot(
            self.times_for_graph,
            self.uptake_for_graph * 100,
            "o",
            "Naměřené hodnoty",
            "blue",
            3,
            6,
        )
        # Vykresleni SPECT uptake jako cerveny bod v dane casove poloze
        self.graph.plot(
            self.time_differencies[2],
            spect_uptake * 100,
            "o",
            "SPECT uptake",
            "red",
            3,
            6,
        )

        # Ulozeni grafu do souboru
        self.graph.Figure.savefig(
            os.path.join(self.output_folder, "Graph.png"), bbox_inches="tight"
        )

        # Vytvoreni canvasu pro Tkinter a zobrazeni grafu v GUI
        canvas = FigureCanvasTkAgg(self.graph.Figure, master=self.graph_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, anchor="center")
        # Uzavreni plt, aby se neuvolnilo pamet
        plt.close()

    ### --------------------------------------------------------------

    ### --------------------------------------------------------------
    ### FUNKCE 3. zalozky

    # funkce pro tlacitko dose/activity
    def compute_activity_and_dose(self):
        # Vymazani predchozich zaznamu v tabulkach s vysledky
        for item in self.results_tree_dose_1.get_children():
            self.results_tree_dose_1.delete(item)

        for item in self.results_tree_dose_2.get_children():
            self.results_tree_dose_2.delete(item)

        # Pokud je hodnota ze SPECT rovna nule, nastav pomer na 1 (zadna korekce)
        if self.spect_entry_value.get() == "0":
            self.pomer = 1

        print(self.pomer)
        # Vypocet integralni hodnoty uptake z fitu riu upravene pomerem, prepocteno na dny
        self.integral_riu = round(
            self.pomer
            * self.riu_params[0]
            / (self.riu_params[1] * self.riu_params[2])
            / 24,
            3,
        )
        # Integral uptake vypocitany numericky (quad) a upraveny pomerem, prepocteno na dny
        self.integral_statik = round(
            self.pomer
            * quad(
                riu_uptace_fce,
                self.times_for_graph[0],
                self.times_for_graph[-1],
                args=tuple(self.riu_params),
            )[0]
            / 24,
            3,
        )
        # Podil F vypocteny jako procento rozdilu mezi integralem statikem a integralem riu
        self.podil_f = 100 - round(self.integral_statik / self.integral_riu * 100, 3)
        # Efektivni polovocas v dnech vypocteny z posledniho parametru riu (vylucovaci konstanta)
        self.eff_polocas = round(np.log(2) / self.riu_params[-1] / 24, 3)  # dny
        # Hmotnost organu (objem * hustota)
        self.organ_mass = float(self.volume_of_organ.get()) * 1.045
        # Velke E (konstanta pro prepočet davky), vypocet podle dane rovnice
        self.big_E = round((self.organ_mass**0.25 + 18) / 7.2, 3)  # (Gy*gram)/(MBq*day)

        # Vypocet absorbovane davky (aktivita * velka E * integral riu / hmotnost organu)
        self.absorbovana_davka = round(
            self.podana_aktivita * self.big_E * self.integral_riu / self.organ_mass, 3
        )
        # Vlozeni vysledku do prvni tabulky
        self.results_tree_dose_1.insert(
            "",
            "end",
            values=(
                self.integral_riu,
                self.podil_f,
                self.eff_polocas,
                self.big_E,
                self.absorbovana_davka,
            ),
        )

        # Preddefinovane hodnoty pozadovanych davky
        pozadovana_davka = [150, 200, 250, 300, 350, 400]

        # Pro kazdou pozadovanou davku vypocet pozadovane aktivity a vlozeni do druhe tabulky
        for dose in pozadovana_davka:
            pozadovana_A = round(
                (1 / self.big_E) * (self.organ_mass * dose) / self.integral_riu, 3
            )
            self.results_tree_dose_2.insert(
                "", "end", values=(dose, pozadovana_A)
            )  # Spravne vlozeni do tabulky

    def protocol_export(self):
        pass

    ### --------------------------------------------------------------

    ### --------------------------------------------------------------
    ### FUNKCE 4. zalozky

    # funkce pro update tabulek mrtve doby
    def update_table_md_params(self):
        # Kontrola, zda je ram tabulky md_data_table_frame vytvoren
        if not hasattr(self, "md_data_table_frame"):
            messagebox.showerror(
                "Error", "Table frame not initialized."
            )  # Pokud ne, vypise chybu a ukonci funkci
            return

        # Vymazani vsech widgetu v ramci tabulky pred aktualizaci
        for widget in self.md_data_table_frame.winfo_children():
            widget.destroy()

        # Podle hodnoty md_parameters_value vybere, jakou tabulku vytvorit
        if self.md_parameters_value.get() <= 2:
            # Pokud je hodnota mensi nebo rovna 2, vytvori readonly tabulku
            self.create_readonly_table_md_params()
        elif self.md_parameters_value.get() == 3:
            # Pokud je hodnota 3, vytvori tabulku s druhym sloupcem upravitelnym
            self.create_editable_table_md_params()

    # funkce pro vytvoreni readonly tabulky mrtvych dob
    def create_readonly_table_md_params(self):
        # Vytvori Treeview widget s 2 sloupci v ramci md_data_table_frame
        tree = ttk.Treeview(
            self.md_data_table_frame,
            columns=("Window type", "DT value"),
            show="headings",
            height=6,
        )

        # Nastavi nadpisy sloupcu
        tree.heading("Window type", text="Window type")
        tree.heading("DT value", text="DT value (s)")

        # Nastavi dummy data podle hodnoty md_parameters_value
        if self.md_parameters_value.get() == 1:
            # Data pro pripad, kdy md_parameters_value je 1
            self.md_data = {
                "ant_pw": 1.2278225074520933e-05,
                "ant_usw": 7.523878532633466e-05,
                "ant_lsw": 7.52862007849338e-05,
                "pos_pw": 1.2120528337418146e-05,
                "pos_usw": 5.7212768749137515e-05,
                "pos_lsw": 5.6539578680391605e-05,
            }
            print(self.md_data)
        else:
            # Data pro ostatni pripady (tu je stejna jako vyse)
            self.md_data = {
                "ant_pw": 1.2278225074520933e-05,
                "ant_usw": 7.523878532633466e-05,
                "ant_lsw": 7.52862007849338e-05,
                "pos_pw": 1.2120528337418146e-05,
                "pos_usw": 5.7212768749137515e-05,
                "pos_lsw": 5.6539578680391605e-05,
            }
            print(self.md_data)

        # Vlozi data do tabulky, kazdy radek odpovida jednomu klici a hodnote ve slovniku
        for window_type, value in self.md_data.items():
            tree.insert("", "end", values=(window_type, value))

        # Nastavi font pro vsechny radky tabulky
        style = ttk.Style()
        style.configure(
            "Treeview", font=("Arial", 12)
        )  # Nastavi font Arial velky 12 pro vsechny radky

        # Zobrazi tabulku v ramci widgetu
        tree.pack()

    # funkce pro vytvoreni editable tabulku mrtve doby
    def create_editable_table_md_params(self):
        # Vytvori tabulku, kde je druhy sloupec editovatelny (entry widgety)
        initial_data = {
            "ant_pw": 1,
            "ant_usw": 1,
            "ant_lsw": 1,
            "pos_pw": 1,
            "pos_usw": 1,
            "pos_lsw": 1,
        }

        # Seznamy pro ulozeni labelu a entry widgetu, aby s nimi slo pozdeji pracovat
        self.md_labels = []
        self.md_entries = []

        # Vytvori ramecek pro hlavicky sloupcu, aby byly vedle sebe
        label_frame = tk.Frame(self.md_data_table_frame)
        label_frame.pack(fill="x", pady=5)  # Zabal ramecek s hlavickami tabulky

        # Hlavicka prvniho sloupce
        label = tk.Label(
            label_frame, text="Window type", width=12, font=("Arial", 16, "bold")
        )
        label.pack(side="left")

        # Hlavicka druheho sloupce
        label = tk.Label(
            label_frame, text="DT value (s)", width=15, font=("Arial", 16, "bold")
        )
        label.pack(side="left")

        # Pro kazdy zaznam vytvori radek s label a entry widgetem
        for window_type, value in initial_data.items():
            frame = tk.Frame(self.md_data_table_frame)
            frame.pack(fill="x", pady=5)

            # Prvni sloupec - nazev parametru (readonly)
            label = tk.Label(frame, text=window_type, width=15, font=("Arial", 12))
            label.pack(side="left", padx=5)

            # Ulozi label do seznamu pro pozdejsi pouziti
            self.md_labels.append(label)

            # Druhy sloupec - editovatelna hodnota
            entry = tk.Entry(frame, width=20, font=("Arial", 12))
            entry.insert(0, str(value))  # Prednastavi puvodni hodnotu jako text
            entry.pack(side="left", padx=5)

            # Ulozi entry widget do seznamu pro pozdejsi pouziti
            self.md_entries.append(entry)

        # Vytvori tlacitko pro ulozeni zadanych hodnot, ktere zavola save_md_data pres safe_call (pro osetreni vyjimek)
        save_md_button = tk.Button(
            self.md_data_table_frame,
            text="Save",
            font=("Arial", 12, "bold"),
            command=lambda: self.safe_call(self.save_md_data),
            **self.button_style,
        )
        save_md_button.pack(pady=15)

    # funkce pro ulozeni dat mrtve doby
    def save_md_data(self):
        # Inicializuje md_data jako prazdny slovnik
        self.md_data = {}

        # Projde vsechny entry widgety a odpovidajici labely
        for i in range(len(self.md_entries)):
            # Ziska text (window type) z labelu na dane pozici
            window_type = self.md_labels[i].cget("text")
            try:
                # Snazi se prevest hodnotu z entry na float
                value = float(self.md_entries[i].get())
            except ValueError:
                # Pokud pretypovani selze (nezadana hodnota apod.), nastavi default 0.0
                value = 0.0

            # Prida key-value par do slovniku md_data
            self.md_data[window_type] = value

        # Vypise slovnik do konzole (pro kontrolu)
        print(self.md_data)

    # funkce pro update tabulek kalibracnich faktoru
    def update_table_kal_params(self):
        # Zkontroluje, jestli je inicializovany frame pro tabulku kal_data
        if not hasattr(self, "kal_data_table_frame"):
            # Pokud neni, zobrazi chybovou hlasku a skonci
            messagebox.showerror("Error", "Table frame not initialized.")
            return

        # Vymaze vsechny widgety ve frame pred aktualizaci
        for widget in self.kal_data_table_frame.winfo_children():
            widget.destroy()

        # Podle hodnoty vybrane v kal_parameters_value vybere, jakou tabulku vytvori
        if self.kal_parameters_value.get() <= 2:
            # Pokud je hodnota mensi nebo rovna 2, vytvori readonly tabulku
            self.create_readonly_table_kal_params()
        elif self.kal_parameters_value.get() == 3:
            # Pokud je hodnota 3, vytvori tabulku s editovatelnou druhou sloupcem
            self.create_editable_table_kal_params()

    # funkce pro vytvoreni readonly tabulek kal faktoru
    def create_readonly_table_kal_params(self):
        # Vytvori Treeview widget s 2 sloupci pro zobrazeni tabulky kalibracnich faktoru
        tree = ttk.Treeview(
            self.kal_data_table_frame,
            columns=("Type", "CF (cps/MBq)"),
            show="headings",
            height=4,
        )

        # Nastavi nadpisy sloupcu
        tree.heading("Type", text="Type")
        tree.heading("CF (cps/MBq)", text="CF (cps/MBq)")

        # Podle hodnoty kal_parameters_value nastavi dummy data do tabulky
        if self.kal_parameters_value.get() == 1:
            # Pokud je hodnota 1, priradi realne kalibracni faktory
            self.kal_data = {"ACSC": 7.77, "SC": 13.30, "AC": 18.3, "No corr": 19.7}
            print(self.kal_data)
        else:
            # Jinak nastavi vsechny hodnoty na 1 (napr. testovaci nebo default hodnoty)
            self.kal_data = {"ACSC": 7.77, "SC": 13.30, "AC": 18.3, "No corr": 19.7}
            print(self.kal_data)

        # Vlozi data do tabulky
        for correction_type, cf_value in self.kal_data.items():
            tree.insert("", "end", values=(correction_type, cf_value))

        # Nastavi font pro vsechny radky tabulky
        style = ttk.Style()
        style.configure("Treeview", font=("Arial", 12))

        # Zobrazi tabulku ve frame, zarovnano vlevo
        tree.pack(anchor="w")

    # funkce pro tabulku editable kalibracnich faktoru
    def create_editable_table_kal_params(self):
        # Inicializace slovniku s vychozimi hodnotami kalibracnich faktoru (namisto seznamu dvojic)
        initial_data = {"ACSC": 1, "SC": 1, "AC": 1, "No corr": 1}

        # Vytvoreni seznamu pro ulozeni widgetu label a entry pro pozdejsi pristup
        self.kal_labels = []
        self.kal_entries = []

        # Vytvoreni frame pro hlavicku tabulky (popisky sloupcu)
        label_frame = tk.Frame(self.kal_data_table_frame)
        label_frame.pack(fill="x", pady=5)  # Zarovnani a mezera

        # Popisek pro prvni sloupec (nazev korekce)
        label = tk.Label(
            label_frame, text="Correction", width=12, font=("Arial", 16, "bold")
        )
        label.pack(side="left")

        # Popisek pro druhy sloupec (hodnota kalibracniho faktoru)
        label = tk.Label(
            label_frame, text="CF (cps/MBq)", width=15, font=("Arial", 16, "bold")
        )
        label.pack(side="left")

        # Pro kazdou polozku ve slovniku vytvori radek s label a editovatelne pole (entry)
        for correction_type, cf_value in initial_data.items():
            frame = tk.Frame(self.kal_data_table_frame)
            frame.pack(fill="x", pady=5)

            # Prvni sloupec - pouze pro cteni (nazev parametru)
            label = tk.Label(frame, text=correction_type, width=15, font=("Arial", 12))
            label.pack(side="left", padx=5)
            self.kal_labels.append(label)

            # Druhy sloupec - editovatelne pole s vychozi hodnotou kalibracniho faktoru
            entry = tk.Entry(frame, width=20, font=("Arial", 12))
            entry.insert(0, str(cf_value))
            entry.pack(side="left", padx=5)
            self.kal_entries.append(entry)

        # Tlacitko pro ulozeni zadanych hodnot a volani funkce save_kal_data s osetrenim vyjimek
        save_kal_button = tk.Button(
            self.kal_data_table_frame,
            text="Save",
            font=("Arial", 12, "bold"),
            command=lambda: self.safe_call(self.save_kal_data),
            **self.button_style,
        )
        save_kal_button.pack(pady=15)

    # funkce pro ulozeni kalibracnich dat
    def save_kal_data(self):
        # Inicializace prazdneho slovniku pro ulozeni kalibracnich faktoru
        self.kal_data = {}

        # Prochazi vsechny entry widgety a odpovidajici popisky (labely)
        for i in range(len(self.kal_entries)):
            # Ziska typ korekce z popisku (napr. "AC", "SC", atd.)
            correction_type = self.kal_labels[i].cget("text")
            try:
                # Pokusi se prevést hodnotu zadane v entry na cislo (float)
                cf_value = float(self.kal_entries[i].get())
            except ValueError:
                # Pokud prevod selze (napr. uzivatel zadal text), nastavi hodnotu na 0.0
                cf_value = 0.0

            # Prida typ korekce a jeho hodnotu do slovniku kal_data
            self.kal_data[correction_type] = cf_value

        # Vytistení vysledneho slovniku do konzole pro kontrolu
        print(self.kal_data)

    ### --------------------------------------------------------------


if __name__ == "__main__":
    app = aplikace()
    app.root.mainloop()
