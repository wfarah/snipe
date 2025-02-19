import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import os,sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib

from snipe.utils import normalize, findNearest, calculate_snr, text_to_matrix

from sigpyproc.readers import FilReader

import argparse

# make plots prettier
exec(open(os.path.join(os.path.dirname(__file__),
                       "rcparams.py"), encoding="utf-8").read())

import matplotlib.gridspec as gridspec
import matplotlib as mpl
# remove the key binding that belong to matplotlib
mpl.rcParams['keymap.save'].remove('s')
mpl.rcParams['keymap.back'].remove('c')


class ControlFrame(tk.Frame):
    def __init__(self, parent, start_sample=None,
                 num_samples=None, dm=None, file_path=None):
        super().__init__(parent)
        self.root = parent

        # Variables
        self.file_path = tk.StringVar(value=file_path)
        self.read_mode = tk.StringVar(value="Samples")  # Default mode
        self.start_sample = tk.StringVar(value=start_sample)
        self.num_samples = tk.StringVar(value=num_samples)
        self.start_time = tk.StringVar()
        self.total_time = tk.StringVar(value=dm)
        self.dm = tk.StringVar()

        # File Selection Button
        tk.Button(self, text="Select File", font=NORMAL_FONT,
                  command=self.select_file).grid(row=0, column=0, columnspan=2, pady=5)
        tk.Label(self, textvariable=self.file_path, font=NOBOLD_FONT,
                 wraplength=300, anchor="w").grid(row=1, column=0, columnspan=2)

        # Dropdown Menu to Select Read Mode
        tk.Label(self, text="Read Mode:", font=NORMAL_FONT).grid(
                row=2, column=0)
        self.read_mode_menu = tk.OptionMenu(self, 
                                            self.read_mode, "Samples", "Time",
                                            command=self.toggle_input_fields)
        self.read_mode_menu.config(font=NORMAL_FONT)
        self.read_mode_menu.grid(row=2, column=1, pady=20)

        # Numeric Validators
        self.register_numeric_input()

        # Sample Input Fields
        self.start_sample_label = tk.Label(self, text="Start Sample:",
                                           font=NORMAL_FONT)
        self.start_sample_label.grid(row=3, column=0)
        self.start_sample_entry = tk.Entry(self,
                                           textvariable=self.start_sample,
                                           validate="key", font=NOBOLD_FONT,
                                           validatecommand=(self.vcmd, "%P"),
                                           width=10)
        self.start_sample_entry.grid(row=3, column=1)

        self.num_samples_label = tk.Label(self, text="Total samples:",
                                          font=NORMAL_FONT)
        self.num_samples_label.grid(row=4, column=0)
        self.num_samples_entry = tk.Entry(self, textvariable=self.num_samples,
                                          validate="key",
                                          font=NOBOLD_FONT,
                                          validatecommand=(self.vcmd, "%P"),
                                          width=10)
        self.num_samples_entry.grid(row=4, column=1)

        # Time Input Fields (Initially Disabled)
        self.start_time_label = tk.Label(self, text="Start Time (s):",
                                         font=NORMAL_FONT)
        self.start_time_label.grid(row=5, column=0)
        self.start_time_entry = tk.Entry(self, textvariable=self.start_time,
                                         validate="key", font=NOBOLD_FONT,
                                         validatecommand=(self.vcmd_float,
                                                          "%P"), 
                                         state="disabled", width=10)
        self.start_time_entry.grid(row=5, column=1)

        self.total_time_label = tk.Label(self, text="Total Time (s):",
                                         font=NORMAL_FONT)
        self.total_time_label.grid(row=6, column=0)
        self.total_time_entry = tk.Entry(self, textvariable=self.total_time,
                                         validate="key",font=NOBOLD_FONT,
                                         validatecommand=(self.vcmd_float, 
                                                          "%P"),
                                         state="disabled", width=10)
        self.total_time_entry.grid(row=6, column=1)

        # Dispersion Measure Input
        tk.Label(self, text="DM (pc/cc):",
                 font=NORMAL_FONT).grid(row=7, column=0)
        tk.Entry(self, textvariable=self.dm, validate="key", font=NOBOLD_FONT,
                 validatecommand=(self.vcmd_float, "%P"), width=10).grid(row=7, 
                                                               column=1)

        # Read File Button
        tk.Button(self, text="Read Block", command=self.read_block,
                  font=NORMAL_FONT).grid(
                row=8, column=0, columnspan=2, pady=10)

        # Fscrunch & Tscrunch Buttons
        self.fscrunch_button = tk.Button(self, text="Fscrunch", 
                                         font=NORMAL_FONT,
                                         command=self.fscrunch)
        self.fscrunch_button.grid(row=9, column=0, pady=5)

        self.tscrunch_button = tk.Button(self, text="Tscrunch", 
                                         font=NORMAL_FONT,
                                         command=self.tscrunch)
        self.tscrunch_button.grid(row=9, column=1, pady=5)

        # Calculate SNR Button
        self.snr_button = tk.Button(self, text="Calculate SNR", 
                                    font=NORMAL_FONT,
                                    command=self.root._calculate_snr)
        self.snr_button.grid(row=10, column=0, columnspan=2, pady=60)

        # Save .npz format
        self.help_button = tk.Button(self, text="Save npz",
                                     font=NORMAL_FONT,
                                     command=self.root.save_npz)
        self.help_button.grid(row=11, column=0, columnspan=2, pady=0)

        # Show help
        self.help_button = tk.Button(self, text="Help",
                                     font=NORMAL_FONT,
                                     command=self.root.show_help)
        self.help_button.grid(row=12, column=0, columnspan=2, pady=0)

    def select_file(self):
        """Opens a file dialog and updates the file_path variable."""
        file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[("Filterbank files", ".fil")])
        if file_path:
            self.file_path.set(file_path)
        self.root.filename = file_path

    def register_numeric_input(self):
        """Registers validation for numeric input fields."""
        self.vcmd = (self.register(self.validate_int), "%P")
        self.vcmd_float = (self.register(self.validate_float), "%P")

    def validate_int(self, new_value):
        """Ensures only integers are entered."""
        return new_value.isdigit() or new_value == ""

    def validate_float(self, new_value):
        """Ensures only floats are entered."""
        try:
            if new_value == "" or new_value == "-":
                return True
            float(new_value)
            return True
        except ValueError:
            return False

    def toggle_input_fields(self, mode):
        """Enables or disables input fields based on the selected mode."""
        if mode == "Samples":
            self.start_sample_entry.config(state="normal")
            self.num_samples_entry.config(state="normal")
            self.start_time_entry.config(state="disabled")
            self.total_time_entry.config(state="disabled")
        else:
            self.start_sample_entry.config(state="disabled")
            self.num_samples_entry.config(state="disabled")
            self.start_time_entry.config(state="normal")
            self.total_time_entry.config(state="normal")

    def read_block(self):
        """Handles reading the file/block after validation."""
        if not self.file_path.get():
            messagebox.showerror("Error", "No file selected!")
            return

        try:
            # Determine which mode is selected
            if self.read_mode.get() == "Samples":
                start_sample = int(self.start_sample.get())\
                        if self.start_sample.get() else None
                num_samples = int(self.num_samples.get())\
                        if self.num_samples.get() else None
                start_time = None
                total_time = None
            else:
                start_sample = None
                num_samples = None
                start_time = float(self.start_time.get())\
                        if self.start_time.get() else None
                total_time = float(self.total_time.get())\
                        if self.total_time.get() else None

            dm = float(self.dm.get()) if self.dm.get() else None
            self.root.dm = dm

        
        except ValueError:
            messagebox.showerror(
                    "Error", "Invalid input! Please enter valid numbers.")

        try:
            self.filfile = FilReader(self.file_path.get())
        except Exception as e:
            messagebox.showerror("Error", "Could not read sigproc file")
            raise

        # we got time instead of samples, convert to samples
        if self.read_mode.get() == "Time":
            start_sample = int(start_time / self.filfile.header.tsamp)
            num_samples  = int(total_time / self.filfile.header.tsamp)

        try:
            # read block and dedisperse
            block = self.filfile.read_block(start_sample, num_samples)
            block  = block.dedisperse(dm).data

            # flip axis in the frequency direction
            if self.filfile.header.foff < 0:
                block = np.flipud(block)
                # frequency axis
                self.root.chan_freqs = self.filfile.header.chan_freqs[::-1]
            else:
                self.root.chan_freqs = self.filfile.header.chan_freqs

            # frequency and time resolution
            self.root.foff = self.filfile.header.foff #MHz
            self.root.tsamp = self.filfile.header.tsamp * 1e3 #ms

            self.root.wfall = block
            self.root.orig_wfall = self.root.wfall.copy()

            # time-axis
            self.root.time_vals = np.linspace(0, 
                        (num_samples-1)*self.filfile.header.tsamp*1e3,
                        num_samples)
            self.root.time_vals_orig = self.root.time_vals.copy()
            self.root.chan_freqs_orig = self.root.chan_freqs.copy()

            self.root.fname = os.path.basename(self.file_path.get())
            self.root.source_name = self.filfile.header.source

            # draw
            self.root.update_plots(init=True)

        except Exception as e:
            messagebox.showerror("Error", "Couldn't read block")
            raise


        print(f"File: {self.file_path.get()}")
        print(f"Read Mode: {self.read_mode.get()}")
        print(f"Start Sample: {start_sample}, Number of Samp: {num_samples}")
        print(f"Start Time: {start_time} s, Total Time: {total_time} s")
        print(f"Dispersion Measure: {dm} pc/cc")

        self.file_loaded = True

    def fscrunch(self):
        """ Average waterfall in frequency """
        tmp = self.root.wfall
        tmp_chan_freqs = self.root.chan_freqs

        # deal with odd number of channels
        if tmp.shape[0] % 2 != 0:
            tmp = tmp[:-1, :]
            tmp_chan_freqs = tmp_chan_freqs[:-1]

        # reshape before summing
        tmp_reshape = tmp.reshape(tmp.shape[0]//2, 
                                  -1, tmp.shape[1])
        self.root.wfall = tmp_reshape.sum(axis=1)
        self.root.chan_freqs = tmp_chan_freqs.reshape(
                tmp_chan_freqs.shape[0]//2, -1).mean(axis=1)
        self.root.foff *= 2
        self.root.update_plots(init=True)


    def tscrunch(self):
        """ Average waterfall in time """
        tmp = self.root.wfall
        tmp_time_vals = self.root.time_vals
        # deal with odd number of samples
        if tmp.shape[1] % 2 != 0:
            tmp = tmp[:, :-1]
            tmp_time_vals = tmp_time_vals[:-1]

        tmp_reshape = tmp.reshape(tmp.shape[0], 
                                  tmp.shape[1]//2, -1)
        self.root.wfall = tmp_reshape.sum(axis=2)
        self.root.time_vals = tmp_time_vals.reshape(
                tmp_time_vals.shape[0]//2, -1).mean(axis=1)
        self.root.tsamp *= 2
        self.root.update_plots(init=True)

class SNIPEApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.title("SNIPE –– Signal-to-Noise Investigation of Pulsed Events")

        screen_width  = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set the window size to a percentage of the screen
        window_width = int(screen_width * 0.7)  # 70% of the screen width
        window_height = int(screen_height * 0.7)  # 70% of the screen height

        #self.geometry("10x10")
        self.geometry(f"{window_width}x{window_height}")
        self.resizable(True, True)

        # Normalize for different screens
        dpi_scaling = self.winfo_fpixels('1i') / 72

        base_normal, base_text = 16, 10
        normal_size = max(10, int(base_normal * dpi_scaling))
        text_size   = max(8, int(base_text * dpi_scaling))

        global NORMAL_FONT
        global NOBOLD_FONT
        global TEXT_FONT
        NORMAL_FONT = ("Helvetica", normal_size, "bold")
        NOBOLD_FONT = ("Helvetica", normal_size)
        TEXT_FONT   = ("Helvetica", text_size, "bold")

        print(self.winfo_fpixels('1i'))
        print(NORMAL_FONT, NOBOLD_FONT, TEXT_FONT)

        # Overwrite some RC params
        matplotlib.rc('font', size=text_size)
        matplotlib.rc('axes', titlesize=text_size)
        matplotlib.rc('xtick', labelsize=text_size)
        matplotlib.rc('ytick', labelsize=text_size)
        #matplotlib.rcParams['figure.dpi'] = dpi_scaling * 100

        # Override window close button (X)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize waterfall to something fun :)
        self.wfall = text_to_matrix("SNIPE", 40, 60).astype(np.float64)
        # Take a copy that we can fall back to when resetting
        self.orig_wfall = self.wfall.copy()

        # Initialize frequency and time axes
        # Some random values
        self.chan_freqs = np.linspace(1408, 1407.5+0.5*self.wfall.shape[0],
                                      self.wfall.shape[0]) 
        self.time_vals  = np.arange(0, 64e-3*self.wfall.shape[1], 64e-3)
        self.chan_freqs_orig = self.chan_freqs.copy()
        self.time_vals_orig = self.time_vals.copy()

        # Some random values for now
        self.foff       = self.chan_freqs[0] - self.chan_freqs[1]
        self.tsamp      = 64e-3

        self.dm = 0

        self.fname = "SNIPE.fil"
        self.source_name = "snipe"
        self.snr = 1
        self.width = 5


        self.fig = plt.figure(figsize=(10, 8))

        # Define the grid layout
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])

        # Waterfall, timeseries, bandpass, and text space
        self.ax_wfall = self.fig.add_subplot(gs[1, 0])
        self.ax_ts = self.fig.add_subplot(gs[0, 0], sharex=self.ax_wfall)
        self.ax_bp = self.fig.add_subplot(gs[1, 1], sharey=self.ax_wfall)
        self.ax_text = self.fig.add_subplot(gs[0, 1])

        # Add text box
        self.ax_text.set_xticks([])
        self.ax_text.set_yticks([])
        self.ax_text.set_frame_on(False)
        self.text = None


        self.fig.subplots_adjust(hspace=0.02, wspace=0.02)  # Reduce spacing

        # Embed Matplotlib Figure into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=4, 
                                         padx=2, pady=2, sticky="nsew")
        
        # Add Navigation Toolbar
        self.navigation = NavigationToolbar2Tk(self.canvas, self, 
                                            pack_toolbar=False)
        self.navigation.update()
        self.navigation.grid(row=4, column=0, columnspan=1, sticky="ew", pady=2)

        # Event connections
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # State variables for interaction when RFI zapping
        self.mode = None  # "signal", "noise", or None
        self.signal_lines = []
        self.noise_lines = []
        self.clicks = []

        # Draw initial plots
        self.update_plots(init = True)

        # Initialize left-hand-side control
        self.control_frame = ControlFrame(self)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tk.call('tk', 'scaling', 0.5)  # Adjust this factor if needed

    def on_close(self):
        """Properly closes the application."""
        self.destroy()  # Destroys the main window
        self.quit()  # Ensures event loop exits


    def update_plots(self, init=False):
        """
        Function to call whenever we want to update plots
        init = True when we don't want to preserve the current zoom state
        """

        self.signal_lines.clear()
        self.noise_lines.clear()
        if not init:
            # Preserve current zoom state (x/y limits)
            xlim_wfall = self.ax_wfall.get_xlim()
            ylim_wfall = self.ax_wfall.get_ylim()
            xlim_xsum = self.ax_ts.get_xlim()
            ylim_ysum = self.ax_bp.get_ylim()

        # Updates all plots based on the current wfall.
        self.ax_wfall.clear()
        self.ax_ts.clear()
        self.ax_bp.clear()

        # Main wfall plot
        extent = [self.time_vals[0] - self.tsamp/2., 
                  self.time_vals[-1] + self.tsamp/2.,
                  self.chan_freqs[0] - np.abs(self.foff)/2., 
                  self.chan_freqs[-1] + np.abs(self.foff)/2.]
        img = self.ax_wfall.imshow(self.wfall, cmap="viridis",
                                    interpolation='nearest', aspect="auto",
                                    extent=extent, origin="lower")

        self.ax_wfall.set_xlabel("Time [ms]")
        self.ax_wfall.set_ylabel("Frequency [MHz]")

        # get normalized timeseries
        # normalize can be edited if another normalization scheme is needed
        self.ts = normalize(np.sum(self.wfall, axis=0))

        # Plot timeseries
        self.ax_ts.plot(self.time_vals, self.ts, linestyle="-")
        self.ax_ts.set_title("Timeseries")
        self.ax_ts.set_xlim(self.time_vals[0], self.time_vals[-1])
        self.ax_ts.tick_params(axis='x', which='both', bottom=False, 
                                 top=False, labelbottom=False)

        # plot bandpass
        self.bp = np.sum(self.wfall, axis=1)
        self.ax_bp.plot(self.bp, self.chan_freqs,
                        linestyle="-")
        self.ax_bp.set_title("Bandpass")
        self.ax_bp.set_ylim(self.chan_freqs[0], self.chan_freqs[-1])

        self.ax_bp.tick_params(axis='y', which='both', left=False, 
                                 right=False, labelleft=False)

        # some initial values
        self.snr = 1
        self.width = 1

        # if self.text is modified, make sure we update
        self.redraw_textbox()

        if not init:
            # Restore zoom state
            self.ax_wfall.set_xlim(xlim_wfall)
            self.ax_wfall.set_ylim(ylim_wfall)
            self.ax_ts.set_xlim(xlim_xsum)
            self.ax_bp.set_ylim(ylim_ysum)
        else:
            self.navigation.update()

        self.canvas.draw()

    def redraw_textbox(self):
        # sampling time in us
        tsamp_us = self.tsamp*1e3
        w = self.width * self.tsamp

        # this is the text that gets displayed on the top right
        text = f"{self.fname}\nSource: {self.source_name}\n"\
                f"{self.foff:.2f} MHz; {tsamp_us:.2f} us\n\n"\
                f"S/N: {self.snr:.2f}\n"\
                f"width: {self.width} samp [{w:.2f} ms]"

        # initialization
        if not self.text:
            self.text = self.ax_text.text(0.5, 0.9, text,
                              fontsize=TEXT_FONT[1],
                              color="red", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="red", 
                              boxstyle="round,pad=0.5"),
                    ha="center", va="top")
        # else just "set_text"
        else:
            self.text.set_text(text)

        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """
        Handles key presses: 's' for signal, 'm' for noise, 'r' to reset,
        'c' for snr calculation, 'f', 't' and 'z' for RFI cleaning, 
        'h' for help, 'q' to quit
        """
        if event.key == "s" and event.inaxes == self.ax_ts:
            print("Select signal region: Click twice")
            self.mode = "signal"
            self.clicks = []

        elif event.key == "n" and event.inaxes == self.ax_ts:
            print("Select noise region: Click twice")
            self.mode = "noise"
            self.clicks = []

        elif event.key == "r" and event.inaxes == self.ax_ts:
            print("Resetting selections")
            self.update_plots()
            self.mode = None
            self.clicks = []

        elif event.key == "r" and event.inaxes == self.ax_wfall:
            print("Resetting zapping")
            self.wfall = self.orig_wfall.copy()
            self.time_vals = self.time_vals_orig.copy()
            self.chan_freqs = self.chan_freqs_orig.copy()
            self.update_plots(init=True)

        elif event.key == "f" and event.inaxes == self.ax_wfall:
            print("Select frequency region to zap: Click twice")
            self.mode = "frequency"
            self.clicks = []

        elif event.key == "t" and event.inaxes == self.ax_wfall:
            print("Select time region to zap: Click twice")
            self.mode = "time"
            self.clicks = []

        elif event.key == "z" and event.inaxes == self.ax_wfall:
            print("Zap pixel")
            self.clicks = []
            self.zap_pixel(event.xdata, event.ydata)

        elif event.key == "c":
            snr = self._calculate_snr()
            if snr:
                print(f"SNR: {snr:.2f}")

        elif event.key == "h":
            self.show_help()

        elif event.key == "q":
            self.on_close()


    def show_help(self):
        """Creates a help popup."""
        help_window = tk.Toplevel()
        help_window.title("Help")
        help_window.geometry("650x780")  # Set window size (Width x Height)

        # Load help text from package
        help_text = open(os.path.join(os.path.dirname(__file__), 
                                 "help.txt"), encoding="utf-8").read()

        text_widget = tk.Text(help_window, wrap="word", height=10, width=60,
                              font=TEXT_FONT)
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")  # Make it read-only
        text_widget.pack(padx=10, pady=10, fill="both", expand=True)


    def on_click(self, event):
        """Handles mouse clicks to select noise/signal, and RFI masking"""
        # "selecting signal and noise"
        if event.inaxes == self.ax_ts and self.mode in ["signal", "noise"]:
            self.clicks.append(event.xdata)  # Store click position

            # Wait for two clicks
            if len(self.clicks) == 2:
                x1, x2 = sorted(self.clicks)  # Ensure left-to-right order
                x1 = findNearest(x1, self.time_vals)
                x2 = findNearest(x2, self.time_vals)
                x1_arg = findNearest(x1, self.time_vals, True)
                x2_arg = findNearest(x2, self.time_vals, True)
                color = "red" if self.mode == "signal" else "blue"

                # only allow 1 signal fill
                if self.mode == "signal" and len(self.signal_lines) != 0:
                    self.signal_lines.clear()
                    self.update_plots()

                # fill area
                y1,y2 = self.ax_ts.get_ylim()
                fill = self.ax_ts.fill_betweenx([y1, y2], x1, x2, 
                                                  color=color, alpha=0.3)

                # ensure y-lim stays the same
                self.ax_ts.set_ylim(y1, y2)

                if self.mode == "signal":
                    # we selected only 1 point
                    if x1_arg == x2_arg:
                        s = self.ts[x1_arg].tolist()
                    else:
                        s = self.ts[x1_arg:x2_arg].tolist()

                    self.signal_lines.extend([s])
                else: #noise
                    n = self.ts[x1_arg:x2_arg].tolist()
                    self.noise_lines.extend([n])


                self.canvas.draw()
                self.clicks = []  # Reset for next selection

                # Reset mode after selection
                self.mode = None

        elif self.mode in ["frequency", "time"] and event.inaxes == self.ax_wfall:
            if self.mode == "frequency":  # Column selection
                self.clicks.append(event.ydata)
            else:  # Time - Row selection
                self.clicks.append(event.xdata)

            if len(self.clicks) == 2:
                v1, v2 = sorted(self.clicks)

                if self.mode == "frequency":
                    print(f"Frequency region (columns): y1={v1:.2f}, y2={v2:.2f}")
                    v1 = findNearest(v1, self.chan_freqs)
                    v2 = findNearest(v2, self.chan_freqs)
                    mask = np.where((self.chan_freqs >= v1) & (self.chan_freqs <= v2))[0]
                    self.wfall[mask, :] = np.median(self.wfall)
                else: # time
                    print(f"Time region (rows): x1={v1:.2f}, x2={v2:.2f}")
                    v1 = findNearest(v1, self.time_vals)
                    v2 = findNearest(v2, self.time_vals)
                    mask = np.where((self.time_vals >= v1) & (self.time_vals <= v2))[0]
                    self.wfall[:, mask] = np.median(self.wfall)

                self.update_plots()
                self.clicks = []
                self.mode = None

    def zap_pixel(self, t, f):
        """ Zap pixel hovered over """
        t_arg = findNearest(t, self.time_vals, True)
        f_arg = findNearest(f, self.chan_freqs, True)

        self.wfall[f_arg, t_arg] = np.median(self.wfall)

        self.update_plots()
        self.clicks = []
        self.mode = None

    def save_npz(self):
        dt = self.time_vals * 1e-3
        dfs = self.chan_freqs
        dm = self.dm
        bandwidth = np.abs(self.chan_freqs[0] - self.chan_freqs[-1])
        duration = self.time_vals[-1] * 1e-3
        center_f = np.mean(self.chan_freqs)
        
        print(dt, dfs, dm, bandwidth, duration, center_f)
        print("NOT IMPLEMENTED YET")

        wfall = np.array(self.wfall)

        burstmetadata = {
            ### required fields:
            'dt'        : dt,                       # array of time axis -> actually tsamp?
            'dfs'       : dfs,                   # array of frequency channels in MHz
            'DM'        : dm,                              # float of dispersion measure (DM)
            'bandwidth' : bandwidth,                               # float of bandwidth in MHz
            'duration'  : duration,                               # file duration in seconds
            'center_f'  : center_f,                      # burst frequency in MHz, unused, optional,
            ### optional fields:
            'freq_unit' : 'MHz',                                   # string of freqeuncy unit, e.g. 'MHz', optional,
            'time_unit' : 's',                                     # string of time unit, e.g. 'ms', optional,
            'int_unit'  : 'Arb',                                   # string of intensity unit, e.g. 'Jy', optional,
            'telescope' : 'ATA',                                   # string of observing telescope, e.g. 'Arecibo', optional,
            #'burstSN'   :  19.76,                                  # float of signal to noise ratio, optional,
            'tbin'      : self.tsamp,                        # float of time resolution, unused, optional
            'raw_shape' : np.shape(wfall)
        }
        if self.filename:
            npz_name = os.path.splitext(self.filename)[0] + '.npz'
            np.savez(npz_name, wfall=wfall, **burstmetadata)


    def _calculate_snr(self):
        """Calculates the SNR of the ts and displays it."""
        if (not self.signal_lines) or (not self.noise_lines):
            messagebox.showerror("Error", 
                                 "Please define signal and noise levels")
            return

        # I couldn't find a better solution when we have 1 point selected, 
        # vs a range, but this works fine
        if len(self.signal_lines) == 1:
            signal = np.array(self.signal_lines).flatten()
        else:
            signal = np.concatenate([self.signal_lines]).flatten()

        if len(self.noise_lines) == 1:
            noise = np.array(self.noise_lines).flatten()
        else:
            noise  = np.concatenate(self.noise_lines).flatten()

        # SNR calculation, calculate_snr can be modified if needed
        snr = calculate_snr(signal, noise)
        self.snr = snr
        # width of pulse in samples
        self.width = len(signal)

        self.redraw_textbox()
        return snr


def main():
    app = SNIPEApp()
    app.mainloop()

# Run the application
if __name__ == "__main__":
    main()
