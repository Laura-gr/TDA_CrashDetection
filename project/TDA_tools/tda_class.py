from copy import deepcopy
import gudhi as gd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gudhi.representations
from datetime import timedelta
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import periodogram
from scipy.integrate import simps
from scipy.fftpack import fft, fftfreq, ifft


class FinanceTimeSeries:

    def __init__(self,data):
        if type(data)==pd.DataFrame:
            self.time_series = data
        else:
            try:
                self.time_series = pd.DataFrame(data)
            except ValueError:
                raise ValueError("Could not convert data to DataFrame")
        self.return_df = False
        self.scaled = False
        self.return_scaled = False
        self.persistence_norms = pd.DataFrame()
        self.persistence_computed = False
        self.avg_PSD = pd.DataFrame()
        self.PSD_filter_keep = None
        self.PSD_freq_cut = None
        self.Lp_norms_scaling = None
        self.norms_var =pd.DataFrame()
        self.var_filter_keep = None
        self.var_freq_cut = None
    
    def copy(self):
        fts = deepcopy(self)
        return fts
    
    def log_return(self, inplace = False):

        """Compute the log return of self.time_series

        Returns:
            FinanceTimeSeries: (self.time_series : pd.DataFrame) log return if inplace == False, else modify self.time_series
        """

        if self.return_df:
            print("This is already a return DataFrame.")
        else:
            if inplace:
                fts = self
            else:
                fts = self.copy()

            fts.time_series = np.log(fts.time_series.pct_change().dropna() +1)

            fts.return_df = True

            if not inplace:
                return fts
            
    def scale(self, inplace = False):

        """Scaling of self.time_series with StandardScaler

        Returns:
            FinanceTimeSeries: (self.time_series : pd.DataFrame) scale if inplace == False, else modify self.time_series
        """

        if self.scaled:
            print("The DataFrame is already scaled.")
        else :
            if inplace:
                fts = self
            else:
                fts = self.copy()
            
            scaler = StandardScaler()
            fts.time_series = pd.DataFrame(scaler.fit_transform(fts.time_series),
                                           columns = fts.time_series.columns,
                                             index = fts.time_series.index
                                             )
            
            fts.scaled = True

            if not inplace:
                return fts
            
    def scale_log_return(self, inplace = False):

        """Compute log return then scale with StandardScaler self.time_series

        Returns:
            FinanceTimeSeries: (self.time_series : pd.DataFrame) computation of scaled log return if inplace == False, else modify self.time_series
        """

        if self.return_scaled:
            print("The DataFrame is already a scaled log_return.")
        else :
            if inplace:
                fts = self
            else:
                fts = self.copy()

        fts.log_return(inplace = True)
        fts.scale(inplace = True)

        fts.return_scaled = True

        if not inplace:
            return fts
        
    def compute_persistence_norms_seq(self, window_size, p_norms, dimension, scaling = None, inplace = False):

        """Compute the sequence of Lp norms of persistence landscapes, each element of the sequence is computed on a window

        Args:
            window_size (int): size of the rolling window
            p_norms (list): list of integers p for which the Lp norm is computed
            dimension (int): degree of the persistence module for which norms are computed
            scaling (sklearn.preprocessing._data): choice of scaler to scale the norm

        Returns:
            FinanceTimeSeries: (self.persistence_norms: pd.Dataframe) computation of Lp persistence norms if inplace == False, else modify self.persistence_norms
        """

        if not self.return_scaled:
            print("The TimeSeries is not a scaled log_return.")
        else:
            if inplace:
                fts = self
            else:
                fts = self.copy()

            diagrams = {}
            for t in fts.time_series.index[window_size+1:]:
                points = fts.time_series[t-BDay(window_size): t].to_numpy()
                skeleton = gd.RipsComplex(points = points)
                Rips_tree = skeleton.create_simplex_tree(max_dimension = dimension+1)
                dgr = Rips_tree.persistence()

                LS = gd.representations.Landscape()
                L = LS.fit_transform([Rips_tree.persistence_intervals_in_dimension(dimension)])
                
                norms = [np.linalg.norm(L[0], ord = p) for p in p_norms]
                diagrams[t] = norms

            Norms = pd.DataFrame(diagrams).transpose()
            Norms.columns = [f"L{p}_norm" for p in p_norms]

            if scaling is not None:
                scaler = scaling
                Norms = pd.DataFrame(scaler.fit_transform(Norms),
                                     columns = Norms.columns,
                                     index = Norms.index
                                     )

            fts.persistence_norms = Norms
            fts.persistence_computed = True
            fts.Lp_norms_scaling = scaling

            if not inplace:
                return fts

    def avgPSD(self , window_size, freq_cut = None, filter_keep = None, inplace = False):

        """Compute the sequence of the Lp norms' average power spectral density at high or low frequencies, each element of the sequence is computed on a window

        Args:
            window_size (int): size of the rolling window on which each persistence norm is computed
            freq_cut (float): threshold for the frequence cut
            filter_keep (str): choice of filter; 'low' to keep low frequencies, 'high' to keep high frequencies, None to keep all frequencies

        Returns:
            FinanceTimeSeries: (self.avg_PSD: pd.Dataframe) computation of filtered average spectral power density if inplace == False, else modify self.avg_PSD
        """

        if inplace:
            fts = self
        else:
            fts = self.copy()

        
        def avgPSD_total(data, freq_cut = None, filter_keep = None):


            if filter_keep is not None and freq_cut is None:
                print("No frequence cut provided.")
            else:
                (f,S)= periodogram(data,scaling = 'density')
                df_freq = pd.DataFrame((f,S), index = ['frequency','PSD']).transpose()

                if filter_keep == 'low':
                    temp =  df_freq[df_freq['frequency'] < freq_cut]
                    res = simps(temp['PSD'], temp['frequency'])
                    return res
                elif filter_keep == 'high':
                    temp =  df_freq[df_freq['frequency'] > freq_cut]
                    res = simps(temp['PSD'], temp['frequency'])
                    return res
                else:
                    return simps(df_freq['PSD'], df_freq['frequency'])
        
        fts.avg_PSD = fts.persistence_norms.rolling(window_size).agg(
            lambda x : avgPSD_total(x, freq_cut, filter_keep)
            )
        fts.avg_PSD.dropna(inplace = True)
        fts.avg_PSD.columns = ['PSD_'+ col_name for col_name in fts.persistence_norms.columns]
        
        fts.PSD_filter_keep = filter_keep
        fts.PSD_freq_cut = freq_cut

        if filter_keep == None:
            print("No filter selected.")

        if not inplace:
            return fts
    
    def var_freq_filter(self, window_size, freq_cut = None, filter_keep = None, spacing = 1, inplace = False):

        """Compute the sequence of the Lp norms' variance, at high or low frequencies each element of the sequence is computed on a window

        Args:
            window_size (int): size of the rolling window on which each persistence norm is computed
            freq_cut (float): threshold for the frequence cut
            filter_keep (str): choice of filter; 'low' to keep low frequencies, 'high' to keep high frequencies, None to keep all frequencies

        Returns:
            FinanceTimeSeries: (self.avg_PSD: pd.Dataframe) computation of variance if inplace == False, else modify self.norms_var
        """

        if inplace:
            fts = self
        else:
            fts = self.copy()
        
        def var_norms_total(data ,freq_cut = None, filter_keep = None, spacing = 1):
            if filter_keep is not None and freq_cut is None:
                print("No frequence cut provided.")
            else:
                norms_fft = fft(data.values)
                norms_freq = fftfreq(len(norms_fft),spacing)
                
                if filter_keep == 'low':
                    norms_fft[np.abs(norms_freq) > freq_cut] = 0
                    norms_low = np.real(ifft(norms_fft))
                    return norms_low.var()
                elif filter_keep == 'high':
                    norms_fft[np.abs(norms_freq) < freq_cut] = 0
                    norms_high = np.real(ifft(norms_fft))
                    return norms_high.var()
                else:
                    return data.var()
        
        fts.norms_var = fts.persistence_norms.rolling(window_size).agg(
            lambda x : var_norms_total(x, freq_cut, filter_keep, spacing)
            )
        fts.norms_var.dropna(inplace = True)
        fts.norms_var.columns = ['var_'+ col_name for col_name in fts.persistence_norms.columns]

        fts.var_freq_cut = freq_cut
        fts.var_filter_keep = filter_keep

        if filter_keep == None:
            print("No filter selected.")

        if not inplace:
            return fts


def computation_tda(data, window_tda = 50, p_norms = [1], dimension = 1, scaling = None, window_freq = 250, freq_cut = None, filter_keep = None, spacing = 1):

    """Compute the sequence of Lp norms of persistence landscapes, each element of the sequence is computed on a window

        Args:
            window_tda (int): size of the rolling window
            p_norms (list): list of integers p for which the Lp norm is computed
            dimension (int): degree of the persistence module for which norms are computed
            scaling (sklearn.preprocessing._data): choice of scaler to scale the norm
            window_freq (int): size of the rolling window on which each persistence norm is computed
            freq_cut (float): threshold for the frequence cut
            filter_keep (str): choice of filter; 'low' to keep low frequencies, 'high' to keep high frequencies, None to keep all frequencies
            spacing (float): choice of spacing for fft in avg var computation


        Returns:
            FinanceTimeSeries: all attributes computed. To access norms: .persistence_norms. To access avgPSD: .avg_PSD. To access norms var: .norms_var

     """
    fts = FinanceTimeSeries(data)
    fts.scale_log_return(inplace = True)
    fts.compute_persistence_norms_seq(window_tda, p_norms, dimension, scaling, inplace = True)
    fts.avgPSD(window_freq, freq_cut, filter_keep, inplace = True)
    fts.var_freq_filter(window_freq, freq_cut, filter_keep, spacing, inplace = True)

    return fts
