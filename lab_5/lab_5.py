
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, CheckboxGroup, Button, Select, ColumnDataSource, Div, Checkbox
from bokeh.plotting import figure
from scipy.signal import iirfilter, filtfilt
import numpy as np

class BokehProgram:
    def __init__(self):
        # default parameters values
        self.INITIAL_AMPLITUDE = 2.0
        self.INITIAL_FREQUENCY = 4.5
        self.INITIAL_PHASE = 0
        self.INITIAL_NOISE_MEAN = 0
        self.INITIAL_NOISE_COVARIANCE = 0.1 
        self.INITIAL_ORDER = 4
        self.INITIAL_CUTOFF = 8
        self.INITIAL_FILTER_TYPE = "butter"
        self.INITIAL_FREQUECY_SAMPLING = 1000
        self.INITIAL_RP = 1
        self.INITIAL_RS = 40

        self.INITIAL_WINDOW_SIZE = 1

        self.INITIAL_SHOW_ORIGINAL = True
        self.INITIAL_SHOW_NOISE = False
        self.INITIAL_SHOW_FILTERED = False
        self.INITIAL_SHOW_LP_FILTERED = False

        
        self.time_start = 0
        self.time_end = 50
        self.sampling_rate = 10000

        #-------------------
        # BOKEH OBJECTS
        #-------------------

        self.signal_div = Div(
            text="""
                <p><b>Signal parameters:</b></p>
            """
        )

        self.filter_div = Div(
            text="""
                <p><b>Filter parameters:</b></p>
            """
        )

        self.lp_filter_div = Div(
            text="""
                <p><b>Low Pass Filter parameters:</b></p>
            """

        )

        self.amplitude_slider = Slider(
            start=-5,
            end=5,
            value=self.INITIAL_AMPLITUDE,
            step=0.05,
            title="Amplitude",
        )

        self.frequency_slider = Slider(
            start=-10,
            end=10,
            value=self.INITIAL_FREQUENCY,
            step=0.05,
            title="Frequency",
        )

        self.phase_slider = Slider(
            start=0,
            end=2 * np.pi,
            value=self.INITIAL_PHASE,
            step=0.01,
            title="Initial Phase",
        )

        self.noise_mean_slider = Slider(
            start=0,
            end=2,
            value=self.INITIAL_NOISE_MEAN,
            step=0.01,
            title="Noise mean",
        )

        self.noise_covariance_slider = Slider(
            start=0,
            end=1.5,
            value=self.INITIAL_NOISE_COVARIANCE,
            step=0.01,
            title="Noise covariance",
        )
        
        self.order_slider = Slider(
            start=1,
            end=10,
            value=self.INITIAL_ORDER,
            step=1, 
            title="Filter Order"
        )

        self.cutoff_slider = Slider(
            start=1,
            end=500, 
            value=self.INITIAL_CUTOFF,
            step=1,
            title="Filter Cutoff Frequency"
        )

        self.filter_type_select = Select(
            value=self.INITIAL_FILTER_TYPE,
            options=[
                "butter",
                "cheby1",
                "cheby2",
                "ellip",
                "bessel",
            ],
            title="Filter Type"
        )

        self.frequency_sampling_slider = Slider(
            value=self.INITIAL_FREQUECY_SAMPLING,
            start=1,
            end=10000,
            step=1,
            title="Frequency Sampling"
        )

        self.rp_slider = Slider(
            title="Allowable attenuation in the passband",
            start=0.1,
            end=10,
            value=self.INITIAL_RP,
            step=0.1
        )

        self.rs_slider = Slider(
            title="Attenuation in the stopband",
            start=10,
            end=100,
            value=self.INITIAL_RS,
            step=1
        )

        self.windows_size_slider = Slider(
            title="Window Size (for low pass filter)",
            start=1,
            end=100,
            value=self.INITIAL_WINDOW_SIZE,
            step=1
        )

        self.show_original_checkbox = Checkbox(
            label="Show Original Signal",
            active=self.INITIAL_SHOW_ORIGINAL
        )

        self.show_noise_checkbox = Checkbox(
            label="Show Noise",
            active=self.INITIAL_SHOW_NOISE
        )
        self.show_filtered_checkbox = Checkbox(
            label="Show Filtered Signal",
            active=self.INITIAL_SHOW_FILTERED
        )
        self.show_lp_filtered_checkbox = Checkbox(
            label="Show Low pass Filtered Signal",
            active=self.INITIAL_SHOW_LP_FILTERED
        )

        self.reset_button = Button(
            label="Reset",
            button_type="success"
        )

        # connect bokeh object with functions
        self.amplitude_slider.on_change('value', self.UpdateSignal)
        self.frequency_slider.on_change('value', self.UpdateSignal)
        self.phase_slider.on_change('value', self.UpdateSignal)
        self.noise_mean_slider.on_change('value', self.UpdateNoise)
        self.noise_covariance_slider.on_change('value', self.UpdateNoise)

        self.show_original_checkbox.on_change('active', self.UpdateLines)
        self.show_noise_checkbox.on_change('active', self.UpdateSignal)
        self.show_filtered_checkbox.on_change('active', self.UpdateLines)
        self.show_lp_filtered_checkbox.on_change('active', self.UpdateLines)

        self.order_slider.on_change('value', self.UpdateFilteredSignal)
        self.cutoff_slider.on_change('value', self.UpdateFilteredSignal)
        self.frequency_sampling_slider.on_change('value', self.UpdateFilteredSignal)
        self.filter_type_select.on_change('value', self.UpdateFilteredSignal)
        self.rp_slider.on_change('value', self.UpdateFilteredSignal)
        self.rs_slider.on_change('value', self.UpdateFilteredSignal)

        self.windows_size_slider.on_change('value', self.UpdateLowPassFilteredSignal)

        self.reset_button.on_click(self.ResetParameters)
        
        # plot object
        self.plot = None
        # bokeh data type object, both axis
        self.source = None
        self.source_filtered = None
        self.source_lp_filtered = None

        
        self.time = None
        self.signal_clear = None
        self.signal_with_noise = None
        self.filtered_signal = None
        self.lp_filtered_signal = None
        self.noise = None
    


    def GenClearSignal(self, amplitude, frequency, init_phase):
        return amplitude * np.sin(2 * np.pi * frequency * self.time + init_phase)

    def GenNoise(self, noise_mean, noise_covariance):
        return np.random.normal(noise_mean, np.sqrt(noise_covariance), self.time.shape)
    
    def FilterSignal(self, signal, fs, order, cuttof, ftype, rp=None, rs=None):
        if (ftype == "butter") or (ftype == "bessel"):
            b, a = iirfilter(order, cuttof, fs=fs, btype="low", ftype=ftype)
        elif ftype == "cheby1":
            b, a = iirfilter(order, cuttof, fs=fs, btype="low", ftype=ftype, rp=rp)
        elif ftype == "cheby2":
            b, a = iirfilter(order, cuttof, fs=fs, btype="low", ftype=ftype, rs=rs)
        elif ftype == "ellip":
           b, a = iirfilter(order, cuttof, fs=fs, btype="low", ftype=ftype, rp=rp, rs=rs)
        

        return filtfilt(b, a, signal)
    
    def LowPassFilter(self, signal, window_size):
        filtered_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
        return filtered_signal

    def GraphAxisInit(self):
        self.time = np.linspace(
            self.time_start,
            self.time_end,
            self.sampling_rate,
            endpoint=True
        )

        self.signal_clear = self.GenClearSignal(
            self.INITIAL_AMPLITUDE,
            self.INITIAL_FREQUENCY,
            self.INITIAL_PHASE
        )

        self.noise = self.GenNoise(
            self.INITIAL_NOISE_MEAN,
            self.INITIAL_NOISE_COVARIANCE
        )

        self.signal_with_noise = self.signal_clear + self.noise

        self.filtered_signal = self.FilterSignal(
            self.signal_with_noise,
            self.INITIAL_FREQUECY_SAMPLING,
            self.INITIAL_ORDER,
            self.INITIAL_CUTOFF,
            self.INITIAL_FILTER_TYPE
        )

        self.lp_filtered_signal = self.LowPassFilter(
            self.signal_with_noise,
            self.INITIAL_WINDOW_SIZE
        )

    
    def PrepareSource(self, x, y):
        return ColumnDataSource(data={'x': x, 'y': y})
    
    def PreparePlot(self):
        self.plot = figure(
            height=800,
            width=1200,
            title="Signal Graph",
            x_axis_label="Time",
            y_axis_label="Signal"
        )

        self.source = self.PrepareSource(self.time, self.signal_clear)
        self.source_filtered = self.PrepareSource(self.time, self.filtered_signal)
        self.source_lp_filtered = self.PrepareSource(self.time, self.lp_filtered_signal)

        self.line = self.plot.line('x', 'y', source=self.source, line_color="blue", legend_label="Original Signal")
        self.line.visible = self.INITIAL_SHOW_ORIGINAL

        self.line_filtered = self.plot.line('x', 'y', source=self.source_filtered, line_color="red", legend_label="Filtered Signal")
        self.line_filtered.visible = self.INITIAL_SHOW_FILTERED

        self.line_lp_filtered = self.plot.line('x', 'y', source=self.source_lp_filtered, line_color="green", legend_label="Low pass Filtered Signal")
        self.line_lp_filtered.visible = self.INITIAL_SHOW_LP_FILTERED

    def UpdateSignal(self, attr, old, new):
        amplitude = self.amplitude_slider.value
        frequency = self.frequency_slider.value
        phase = self.phase_slider.value
        show_noise = self.show_noise_checkbox.active

        self.signal_clear = self.GenClearSignal(
            amplitude,
            frequency,
            phase
        )
        self.signal_with_noise = self.signal_clear + self.noise

        res_signal = self.signal_with_noise if show_noise else self.signal_clear
        self.source.data = {'x': self.time, 'y': res_signal}
        self.UpdateLowPassFilteredSignal(attr, old, new)
        self.UpdateFilteredSignal(attr, old, new)

    def UpdateNoise(self, attr, old, new):
        noise_mean = self.noise_mean_slider.value
        noise_covariance = self.noise_covariance_slider.value
        self.noise = self.GenNoise(noise_mean, noise_covariance)
        self.UpdateSignal(attr, old, new)

    
    def UpdateFilteredSignal(self, attr, old, new):
        order = self.order_slider.value
        cuttof = self.cutoff_slider.value
        ftype = self.filter_type_select.value
        frequency_sampling = self.frequency_sampling_slider.value
        rp = self.rp_slider.value
        rs = self.rs_slider.value
        signal = self.source.data['y']

        if (cuttof >= (frequency_sampling / 2)):
            frequency_sampling = 2 * cuttof + 2
            self.frequency_sampling_slider.value = frequency_sampling

        filtered_signal = self.FilterSignal(signal, frequency_sampling, order, cuttof, ftype,
            rp=rp, rs=rs)
        self.source_filtered.data = {'x': self.time, 'y': filtered_signal}
    
    def UpdateLowPassFilteredSignal(self, attr, old, new):
        windows_size = self.windows_size_slider.value
        signal = self.source.data['y']
        filtered_signal = self.LowPassFilter(signal, windows_size)

        self.source_lp_filtered.data = {'x': self.time, 'y': filtered_signal}
    
    def UpdateLines(self, attr, old, new):
        show_original = self.show_original_checkbox.active
        show_filtered = self.show_filtered_checkbox.active
        show_lp_filtered = self.show_lp_filtered_checkbox.active

        self.line.visible = show_original
        self.line_filtered.visible = show_filtered
        self.line_lp_filtered.visible = show_lp_filtered

    def ResetParameters(self):
        self.amplitude_slider.value = self.INITIAL_AMPLITUDE
        self.frequency_slider.value = self.INITIAL_FREQUENCY
        self.phase_slider.value = self.INITIAL_PHASE
        self.noise_mean_slider.value = self.INITIAL_NOISE_MEAN
        self.noise_covariance_slider.value = self.INITIAL_NOISE_COVARIANCE

        self.order_slider.value = self.INITIAL_ORDER
        self.filter_type_select.value = self.INITIAL_FILTER_TYPE
        self.cutoff_slider.value = self.INITIAL_CUTOFF
        self.frequency_sampling_slider.value = self.INITIAL_FREQUECY_SAMPLING
        self.rp_slider.value = self.INITIAL_RP
        self.rs_slider.value = self.INITIAL_RS

        self.windows_size_slider.value = self.INITIAL_WINDOW_SIZE

        self.show_original_checkbox.active = self.INITIAL_SHOW_ORIGINAL
        self.show_noise_checkbox.active = self.INITIAL_SHOW_NOISE
        self.show_filtered_checkbox.active = self.INITIAL_SHOW_FILTERED
        self.show_lp_filtered_checkbox = self.INITIAL_SHOW_LP_FILTERED

    
    def BuildBokehProgram(self):
        inputs = column(
            self.signal_div,
            self.amplitude_slider,
            self.frequency_slider,
            self.phase_slider,
            self.noise_mean_slider,
            self.noise_covariance_slider,

            self.filter_div,
            self.order_slider,
            self.cutoff_slider,
            self.filter_type_select,
            self.frequency_sampling_slider,
            self.rp_slider,
            self.rs_slider,

            self.lp_filter_div,
            self.windows_size_slider,
            
            self.show_original_checkbox,
            self.show_noise_checkbox,
            self.show_filtered_checkbox,
            self.show_lp_filtered_checkbox,

            self.reset_button
            #sizing_mode='stretch_width'
        )

        layout = row(self.plot, inputs) #, sizing_mode='stretch_width')
        curdoc().add_root(layout)
        curdoc().title = "Harmonic Signal with Noise"

bokeh_prog = BokehProgram()
bokeh_prog.GraphAxisInit()
bokeh_prog.PreparePlot()
bokeh_prog.BuildBokehProgram()

#bokeh serve lab_5.py