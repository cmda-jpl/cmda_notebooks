from io import BytesIO
import traceback
import os
import warnings
import cartopy.crs as ccrs
import numpy as np
import scipy.ndimage as ndi
import param
import pandas as pd
import datetime as dt
import panel as pn
import requests
import xarray as xr
import hvplot.xarray
import hvplot.pandas
import holoviews as hv
from bokeh.resources import INLINE
from holoviews import opts
from fillna import replace_nans
from holoviews.plotting.util import list_cmaps

warnings.filterwarnings('ignore')
cmaps = list_cmaps(reverse=False, provider='matplotlib')
datasets = pd.read_csv('datasets.csv').drop_duplicates()
variables = pd.read_csv('variables.csv')
names = datasets.dataset_name
categories = datasets.category.unique()
dset_dict = {cat: list(names[datasets.category == cat].unique()) for cat in categories}
seasons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
url_template = """
## Browser URL:
{api_url}
## Plot URL:
{plot_url}
## Plot:
"""

def get_variables(dataset_name, three_dim_only=False):
    if three_dim_only:
        return datasets.long_name[(names == dataset_name) 
                                  & (datasets.dimensions == 3)].dropna().unique()
    return datasets.long_name[names == dataset_name].dropna().unique()

def get_attrs(dataset_name, variable_name):
    return datasets[(names==dataset_name) & (datasets.long_name==variable_name)].to_dict(orient='records')[0]

def gen_time_range(st, et):
    return pd.date_range(st, et, freq='MS')

class NullSubsetter(param.Parameterized):
    def __init__(self, service=None, **params):
        param.Parameterized.__init__(self, **params)
    
class SpatialSubsetter(NullSubsetter):
    latitude_range = param.Range(default=(-90, 90), bounds=(-90, 90))
    longitude_range = param.Range(default=(0, 360), bounds=(0, 360))
        
class TemporalSubsetter(param.Parameterized):
    start_time = param.String('')
    end_time = param.String('')
    
    def __init__(self, service=None, **params):
        self.service = service
        self._update_time_ranges()
        super().__init__(**params)
        
    def _update_time_ranges(self):
        start, end = self.time_range
        self.param['start_time'].label = f'Start Time (Earliest: {start}): '
        self.param['end_time'].label = f'End Time (Latest: {end}): '
        self.start_time = start
        self.end_time = end
        
    @property
    def time_range(self):
        if hasattr(self.service, 'all_selectors'):
            selectors = self.service.all_selectors
        else:
            selectors = [self]
        start_times = [selector.attrs['start'] for selector in selectors]
        end_times = [selector.attrs['end'] for selector in selectors]
        st, et = str(max(start_times)), str(min(end_times))
        return f'{st[:4]}-{st[4:]}', f'{et[:4]}-{et[4:]}'


class SeasonalSubsetter(TemporalSubsetter):
    months = param.ListSelector(default=seasons, objects=seasons)

class SpatialTemporalSubsetter(TemporalSubsetter, SpatialSubsetter):
    pass

class SpatialSeasonalSubsetter(SeasonalSubsetter, SpatialSubsetter):
    pass

class DatasetSelector(param.Parameterized):
    defaults = get_variables('GFDL/ESM2G')
    category = param.ObjectSelector(objects=list(dset_dict.keys()))
    dataset = param.ObjectSelector(default='GFDL/ESM2G', objects=dset_dict['Model: Historical'])
    variable = param.ObjectSelector(default=defaults[0], objects=defaults)
    pressure = param.Integer(-999999, precedence=-1)
    
    def __init__(self, service, three_dim_only=False, **params):
        self.service = service
        self.three_dim_only = three_dim_only
        self._update_variable()
        super().__init__(**params)
    
    @param.depends('category', watch=True)
    def _update_datasets(self):
        self.param['dataset'].objects = dset_dict[self.category]
        self.dataset = list(dset_dict[self.category])[0]

    @param.depends('dataset', watch=True)
    def _update_variable(self):
        self.param['variable'].objects = get_variables(self.dataset, 
                                                       three_dim_only=self.three_dim_only)
        self.variable = get_variables(self.dataset, three_dim_only=self.three_dim_only)[0]

    @param.depends('category', 'dataset', 'variable', watch=True)
    def _update_subsetter(self):
        if hasattr(self, 'time_range'):
            self._update_time_ranges()
        else:
            self.service.subsetter._update_time_ranges()
        if hasattr(self.service, 'start_time2'):
            self.service._update_ref_time_range()
        
    @param.depends('variable', watch=True)
    def _update_pressure(self):
        if self.attrs['dimensions'] > 2:
            self.pressure = 500
            if not self.three_dim_only:
                self.param['pressure'].precedence = 1
        else:
            self.param['pressure'].precedence = -1
            self.pressure = -999999
    
    @property
    def attrs(self):
        return get_attrs(self.dataset, self.variable)

class DatasetSubsetSelector(DatasetSelector, SpatialSubsetter):
    pass

class DatasetAnomalySelector(DatasetSubsetSelector):
    anomaly = param.Boolean(False, label='Use Anomaly')

class DatasetBinsSelector(DatasetSelector):
    nbins = param.Integer(10, label='Number of Bins')

class DatasetMonthSelector(DatasetSelector, SeasonalSubsetter):
    cmap = param.ObjectSelector(objects=cmaps, default='viridis', label='Colormap')

class DatasetMonthSpatialSelector(DatasetSelector, SpatialSeasonalSubsetter):
    pass

class DatasetPresRangeSelector(DatasetSelector):
    pressure = param.Integer(-999999, label='Pressure Min', precedence=-1)
    pressure_max = param.Integer(-999999, label='Pressure Max', precedence=-1)
    
    @param.depends('variable', watch=True)
    def _update_pressure(self):
        for p, v in zip(['pressure', 'pressure_max'], [200, 900]):
            if self.attrs['dimensions'] > 2:
                setattr(self, p, v)
                self.param[p].precedence = 1
            else:
                self.param[p].precedence = -1
                setattr(self, p, -999999)

class DatasetSamplingSelector(DatasetSelector):
    custom_bins = param.Boolean(False, label='Use custom binning specification')
    bin_min = param.String('-999999', precedence=-1)
    bin_max = param.String('-999999', precedence=-1)
    nbins = param.Integer(0, precedence=-1, label='Number of Bins')
    
    @param.depends('custom_bins', watch=True)
    def _update_bins(self):
        for p in ['bin_min', 'bin_max', 'nbins']:
            if self.custom_bins:
                v = '' if p in ['bin_min', 'bin_max'] else 0
                setattr(self, p, v)
                self.param[p].precedence = 1
            else:
                self.param[p].precedence = -1
                v = '-999999' if p in ['bin_min', 'bin_max'] else 0
                setattr(self, p, v)
        
    
class Service(param.Parameterized):
    target_name = 'Target Variable'
    target_selector_cls = DatasetSelector
    selector_names = None
    selector_cls = DatasetSelector
    subsetter_cls = SpatialTemporalSubsetter
    latlon_prefix = ''
    time_prefix = ''
    month_prefix = ''
    latlon_suf = True
    three_dim_only = False
    decode_times = False
    ntargets = param.Integer(0, bounds=(0, 1), precedence=-1)
    nvars = param.Integer(1, bounds=(1, 6), label='Number Of Variables')
    npresses = param.Integer(0, precedence=-1)
    state = param.Integer(0, precedence=-1)
    host = 'http://api.jpl-cmda.org'
    endpoint = '/'
    
    def __init__(self, viewer=None, **params):
        self.viewer = viewer
        t_svc = None if issubclass(self.target_selector_cls, TemporalSubsetter) else self
        svc = None if issubclass(self.selector_cls, TemporalSubsetter) else self
        self.target_selector = self.target_selector_cls(t_svc, three_dim_only=self.three_dim_only, 
                                                        name=self.target_name)
        if not self.selector_names:
            self.dataset_selectors = [self.selector_cls(svc, three_dim_only=self.three_dim_only, 
                                                        name='Dataset 1')]
        else:
            self.dataset_selectors = [
                self.selector_cls(svc, three_dim_only=self.three_dim_only, 
                                  name=self.selector_names[i])
                for i in range(self.nvars)
            ]
        sub_name = '' if self.subsetter_cls is NullSubsetter else 'Subsetting Options'
        self.subsetter = self.subsetter_cls(svc, name=sub_name)
        self.purpose = pn.widgets.TextAreaInput(name='Execution Purpose',
                                                placeholder='Describe execution purpose here (Optional)')
        def press(event):
            self.npresses += 1
        self.plot_button = pn.widgets.Button(name='Generate Data', width=200)
        self.plot_button.on_click(press)
        
        def download():
            buf = BytesIO(self.ds.to_netcdf())
            buf.seek(0)
            return buf
        self.file_download = pn.widgets.FileDownload(callback=download, label='Download Data', 
                                                     filename='data.nc', width=200)
        self.browser_url = pn.pane.Markdown(url_template.format(api_url='', plot_url=''), width=800)
        super().__init__(**params)
    
    @param.depends('nvars', watch=True)
    def _update_datasets(self):
        svc = None if issubclass(self.selector_cls, TemporalSubsetter) else self
        while len(self.dataset_selectors) > self.nvars:
            self.dataset_selectors.pop(-1)
        for i in range(len(self.dataset_selectors), self.nvars):
            name = self.selector_names[i] if self.selector_names else f'Dataset {i+1}'
            self.dataset_selectors.append(self.selector_cls(svc, name=name, 
                                                            three_dim_only=self.three_dim_only))
        if hasattr(self.subsetter, 'time_range'):
            self.subsetter._update_time_ranges()
            
    @param.depends('nvars')
    def select_dataset(self):
        selectors = [pn.Param(selector, name=selector.name, width=450) for selector in self.all_selectors]
        tabs = pn.Tabs(*selectors)
        return tabs
    
    @param.depends('npresses')
    def plot(self):
        if not self.npresses:
            self._pane = pn.pane.HTML('', width=800)
            return self._pane
        self.plot_button.disabled = True
        self.plot_button.name = 'Working...'
        try:
            figure = self.figure
            self._build_output(figure)
        except:
            self._pane.object = traceback.format_exc()
            figure = self._pane
        finally:
            self.plot_button.disabled = False
            self.plot_button.name = 'Generate Data'
        self.state += 1
        return figure
 
    @param.depends('state', watch=True)
    def _update_state(self):
        self.viewer._save_mimebundle()

    def _build_output(self, fig):
        kwargs = {}
        if hasattr(self, 'widgets'):
            kwargs['widgets'] = self.widgets
        buttons = pn.Row(self.plot_button, self.file_download)
        output = pn.Column(pn.Param(self.param, **kwargs),
                           self.select_dataset, self.subsetter,
                           self.purpose, buttons, self.browser_url, 
                           fig)
        if self.viewer is not None:
            self.viewer._panels[self.name] = self.browser_url
        return output
 
    def panel(self):
        return self._build_output(self.plot)
    
    def download_data(self):
        r1 = requests.get(self.url, params=self.query)
        resp = r1.json()
        endpoint = self.endpoint[self.nvars-1] if isinstance(self.endpoint, list) else self.endpoint
        api_url = r1.url.replace(endpoint, '/' + self.html_name)
        plot_url = resp.get('url', '')
        self.browser_url.object = url_template.format(api_url=f'<{api_url}>', 
                                                      plot_url=f'<{plot_url}>')
        url = resp['dataUrl']
        self.file_download.filename = os.path.basename(url)
        r = requests.get(url)
        buf = BytesIO(r.content)
        return self._postprocess_data(xr.open_dataset(buf, decode_times=self.decode_times))

    def v(self, number):
        model, variable = self.query[f'model{number}'], self.query[f'var{number}']
        return f'{model}:{variable}'
    
    def _postprocess_data(self, ds):
        return ds
    
    @property
    def all_selectors(self):
        if self.ntargets:
            return [self.target_selector] + self.dataset_selectors
        return self.dataset_selectors
    
    @property
    def url(self):
        if isinstance(self.endpoint, list):
            return self.host + self.endpoint[self.nvars-1]
        else:
            return self.host + self.endpoint
    
    @property
    def html_name(self):
        if hasattr(self, 'html_base'):
            name = self.html_base
        else:
            name = os.path.basename(self.endpoint)
        return name + '.html'
    
    @property
    def figure(self):
        return pn.pane.HTML('')
    
    @property
    def ds(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
        query_str = '&'.join(['='.join([str(k), str(v)]) for k, v in self.query.items()])
        if query_str not in self._cache:
            self._cache[query_str] = self.download_data()
        return self._cache[query_str]
        
    @property
    def query(self):
        query = dict(purpose=self.purpose.value)
        if not isinstance(self.dataset_selectors[0], SpatialSubsetter):
            query.update(latS=self.subsetter.latitude_range[0],
                         latE=self.subsetter.latitude_range[-1],
                         lonS=self.subsetter.longitude_range[0],
                         lonE=self.subsetter.longitude_range[-1])
        if not isinstance(self.dataset_selectors[0], TemporalSubsetter):
            query.update(timeS=self.subsetter.start_time.replace('-', ''),
                         timeE=self.subsetter.end_time.replace('-', ''))
        if hasattr(self.subsetter, 'months'):
            query.update(months=[self.subsetter.months.index(m) + 1 for m in self.subsetter.months])
        latlon_basenames = ['latS', 'latE', 'lonS', 'lonE']
        latlon_names = [self.latlon_prefix + name for name in latlon_basenames]
        time_basenames = ['timeS', 'timeE']
        time_names = [self.time_prefix + name for name in time_basenames]
        month_basename = 'months'
        month_name = self.month_prefix + month_basename
        for i, selector in enumerate(self.all_selectors):
            mapper = dict(model=selector.dataset, var=selector.attrs['variable'], pres=selector.pressure)
            if 'latS' not in query:
                latlon_vals = [*selector.latitude_range, *selector.longitude_range]
                latlon_vmap = dict(zip(latlon_names, latlon_vals))
                mapper.update(**latlon_vmap)
            if 'timeS' not in query:
                time_range = [t.replace('-', '') for t in selector.time_range]
                mapper.update(**dict(zip(time_names, time_range)))
            if 'months' not in query and hasattr(selector, 'months'):
                mapper[month_name] = [selector.months.index(m) + 1 for m in selector.months]
            if hasattr(selector, 'bin_min'):
                mapper.update(binMin=selector.bin_min, binMax=selector.bin_max, binN=selector.nbins)
            if hasattr(selector, 'pressure_max'):
                mapper['presa'] = selector.pressure_max
            for k, v in mapper.items():
                if k == 'model':
                    v = v.replace('/', '_')
                if k == 'presa' or (self.latlon_suf and k in latlon_basenames):
                    k = k[:-1] + str(i + 1) + k[-1]
                else:
                    k += str(i + 1)
                query[k] = v
        return query
        
    
class TimeSeriesService(Service):
    selector_cls = DatasetSubsetSelector
    subsetter_cls = TemporalSubsetter
    endpoint = '/svc/timeSeries'
    html_base = 'timeSeries8'
    latlon_prefix = 'v'
    latlon_suf = False
    
    def _postprocess_data(self, ds):
        datasets = [self.v(i) for i in range(1, self.nvars+1)]
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        ds = (ds.rename(varIdx='Dataset', monthIdx='time')
                .rename({self.query['var1']: 'variable'})
                .assign_coords(Dataset=datasets, time=times))
        ds['time'] = times
        return ds
    
    @property
    def figure(self):
        ds = self.ds.copy()
        y = 'variable'
        if self.nvars > 1:
            for i in range(2, self.nvars+1):
                if self.query[f'var{i}'] != self.query['var1']:
                    ds[y] = ((ds[y] - ds[y].min('time')) / 
                             (ds[y].max('time') - ds[y].min('time')))
                    ds[y].attrs['units'] = '0-1'
                    ds = ds.rename({y: 'Normalized Variable'})
                    y = 'Normalized Variable'
                    break
        return ds.hvplot.line(x='time', y=y, by='Dataset', 
                              legend='bottom', width=800, height=400)
    
    @property
    def query(self):
        query = dict(**super().query)
        query['nVar'] = self.nvars
        return query

class ScatterHistService(Service):
    selector_names = ['Variable 1', 'Variable 2']
    nvars = param.Integer(2, precedence=-1)
    number_of_samples = param.Integer(1000)
    endpoint = '/svc/scatterPlot2Vars'
    
    def _postprocess_data(self, ds):
        v1, v2 = self.v(1), self.v(2)
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        ds = ds.rename(data1=v1, data2=v2)
        ds = ds.stack(pt=['time', 'lat', 'lon']).reset_index('pt').dropna('pt')
        ds = ds.isel(pt=np.random.choice(np.arange(ds.pt.size), self.query['nSample'], replace=False))
        ds.attrs['corr'] = np.corrcoef(ds[v1], ds[v2])[0, 1]
        return ds
        
    @property
    def figure(self):
        v1, v2 = self.v(1), self.v(2)
        f1 = (self.ds.reset_coords().to_dataframe().hvplot
                  .scatter(x=v1, y=v2, title=f'Correlation: {self.ds.corr:1.2}', width=800, height=300))
        f2 = self.ds.hvplot.hist(y=v1, width=800, height=300, normed=True)
        f3 = self.ds.hvplot.hist(y=v2, width=800, height=300, normed=True)
        return pn.Column(f1, f2, f3)

    @property
    def query(self):
        query = dict(**super().query)
        query['nSample'] = self.number_of_samples
        return query


class DifferencePlotService(Service):
    selector_names = ['Variable 1', 'Variable 2']
    nvars = param.Integer(2, precedence=-1)
    endpoint = '/svc/diffPlot2V'
    html_base = 'diffPlot2Vars'
    cmap1 = param.ObjectSelector(objects=cmaps, default='coolwarm', 
                                 label='Difference Colormap', precedence=0.1)
    cmap2 = param.ObjectSelector(objects=cmaps, default='viridis', 
                                 label='Variable Colormap', precedence=0.1)
    
    def _postprocess_data(self, ds):
        v1, v2 = self.v(1), self.v(2)
        ds = ds.rename(data1=v1, data2=v2)
        ds['diff'] = ds[v1] - ds[v2]
        return ds
        
    @property
    def figure(self):
        v1, v2 = self.v(1), self.v(2)
        f1 = self.ds.hvplot.quadmesh('lon', 'lat', 'diff', title='diff',
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, rasterize=True,
                                     cmap=self.cmap1)
        f2 = self.ds.hvplot.quadmesh('lon', 'lat', v1, title=v1,
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, rasterize=True,
                                     cmap=self.cmap2)
        f3 = self.ds.hvplot.quadmesh('lon', 'lat', v2, title=v2,
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, rasterize=True,
                                     cmap=self.cmap2)
        return pn.Column(f1, f2, f3)

    @property
    def query(self):
        query = dict(**super().query)
        for i in range(1, 4):
            query[f'colorMap{i}'] = 'rainbow'
        return query

    
class RandomForestService(Service):
    nvars = param.Integer(1, bounds=(1, 10), label='Source Variables')
    ntargets = param.Integer(1, bounds=(0, 1), precedence=-1)
    endpoint = '/svc/randomForest'
    
    def _postprocess_data(self, ds):
        vs = self.v(1)
        vt = [self.v(i+2) for i in range(self.nvars)]
        ds = ds.assign_coords(index=vt).rename(importance=f'importance to predict {vs}')
        return ds
        
    @property
    def figure(self):
        vs = self.v(1)
        y = f'importance to predict {vs}'
        return self.ds.hvplot.bar(x='index', y=y, width=800, height=400)

    @property
    def query(self):
        query = dict(**super().query)
        query['nVar'] = self.nvars
        return query


class EOFService(Service):
    decode_times = True
    selector_names = ['Data']
    nvars = param.Integer(1, precedence=-1)
    anomaly = param.Boolean(False, label='Use Anomaly')
    cmap = param.ObjectSelector(objects=cmaps, default='coolwarm', label='Colormap', precedence=0.1)
    endpoint = '/svc/EOF'
    
    def _postprocess_data(self, ds):
        ds = ds.rename(index='EOF')
        ds['varP'] *= 100
        return ds
        
    @property
    def figure(self):
        f1 = self.ds.varP.hvplot.line(x='EOF', y='varP',
                                      title='Variance Explained (%)')
        f2 = self.ds.patterns.hvplot.quadmesh('lon', 'lat', title='EOF',
                                          widget_location='bottom',
                                          projection=ccrs.PlateCarree(), cmap=self.cmap,
                                          crs=ccrs.PlateCarree(), geo=True, width=800,
                                          coastline=True, rasterize=True)
        f3 = self.ds.tser.hvplot.line(x='time', y='tser', title='PC',
                                      widget_location='bottom')
        return pn.Column(f1, f2, f3)

    @property
    def query(self):
        query = dict(**super().query)
        query['anomaly'] = int(self.anomaly)
        return query

    
class JointEOFService(Service):
    selector_names = ['Variable 1', 'Variable 2']
    selector_cls = DatasetAnomalySelector
    nvars = param.Integer(2, precedence=-1)
    cmap = param.ObjectSelector(objects=cmaps, default='coolwarm', label='Colormap', precedence=0.1)
    endpoint = '/svc/JointEOF'
    html_base = 'jointEOF'
    
    def _postprocess_data(self, ds):
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        ds = ds.rename(mode='EOF')
        ds = ds.assign_coords(EOF=ds.EOF.astype(int)+1, time=times)
        ds['time'] = times
        ds['covExplained'] *= 100
        return ds
        
    @property
    def figure(self):
        varf = self.ds.covExplained.hvplot.line(x='EOF', y='covExplained',
                                          title='Covariance Explained (%)')
        tabs = pn.Tabs()
        for i in range(1, 3):
            ef1 = self.ds[f'pattern{i}'].hvplot.quadmesh(
                                            f'lon{i}', f'lat{i}', title='EOF',
                                            widget_location='bottom', cmap=self.cmap,
                                            projection=ccrs.PlateCarree(),
                                            crs=ccrs.PlateCarree(), geo=True, width=800,
                                            coastline=True, rasterize=True)
            ef2 = self.ds[f'amp{i}'].hvplot.line(x='time', y=f'amp{i}', title='PC',
                                                 widget_location='bottom')
            tabs.append((self.v(i), pn.Column(ef1, ef2)))
        return pn.Column(varf, tabs)

    @property
    def query(self):
        query = dict(**super().query)
        for i in range(1, 3):
            query[f'anomaly{i}'] = int(self.all_selectors[i-1].anomaly)
        return query

    
class CorrelationMapService(Service):
    selector_names = ['Variable 1', 'Variable 2']
    nvars = param.Integer(2, precedence=-1)
    lag = param.Integer(0, label='Time Lag in Months')
    cmap = param.ObjectSelector(objects=cmaps, default='viridis', label='Colormap', precedence=0.1)
    endpoint = '/svc/correlationMap'
            
    @property
    def figure(self):
        return self.ds.corr.hvplot.quadmesh('lon', 'lat',
                                            projection=ccrs.PlateCarree(),
                                            crs=ccrs.PlateCarree(), geo=True,
                                            coastline=True, width=800, cmap=self.cmap,
                                            rasterize=True)

    @property
    def query(self):
        query = dict(**super().query)
        query['laggedTime'] = int(self.lag)
        return query

class ConditionalPDFService(Service):
    selector_names = ['Independent Variable', 'Dependent Variable']
    selector_cls = DatasetBinsSelector
    nvars = param.Integer(2, precedence=-1)
    cmap = param.ObjectSelector(objects=cmaps, default='viridis', label='Colormap', precedence=0.1)
    endpoint = '/svc/conditionalPdf'
    
    def _postprocess_data(self, ds):
        return ds.rename(index='yc', indexJ='xc', indexK='x', indexL='y')
        
    @property
    def figure(self):
        v1, v2 = self.v(1), self.v(2)
        meshes = []
        curve = (hv.Curve((self.ds.binsXC, self.ds.median1), 'binsX', 'binsY')
                   .opts(opts.Curve(color='k', line_width=4, tools=['hover'])))
        for i in range(self.ds.xc.size):
            x = self.ds.binsX.isel(x=[i, i+1]).values
            y = self.ds.binsY.isel(xc=[i, i]).values
            z = self.ds.pdf.isel(xc=i).values.reshape(-1, 1)
            submesh = hv.QuadMesh((x, y, z), vdims=['pdf'])
            meshes.append(submesh)
        mesh = hv.Overlay(meshes) * curve
        return mesh.opts(opts.QuadMesh(colorbar=True, width=800, height=400, 
                                       xlabel=v1, ylabel=v2, tools=['hover'], cmap=self.cmap))

    @property
    def query(self):
        query = dict(**super().query)
        query['anomaly'] = 0
        for i, dim in zip(range(1, 3), ['X', 'Y']):
            query[f'nBin{dim}'] = int(self.all_selectors[i-1].nbins)
        return query


class AnomalyService(Service):
    selector_names = ['Source Variable', 'Reference Variable']
    nvars = param.Integer(1, precedence=-1)
    reference_to_remove = param.ObjectSelector(
        default='seasonal cycle',
        objects=['seasonal cycle', 'mean only'],
        label='What reference to remove'
    )
    use_ref = param.Boolean(False, label='Calculate reference from another variable')
    ref_period = param.Boolean(False, label='Calculate reference from different period')
    start_time2 = param.String('', precedence=-1, label='Reference Start Time')
    end_time2 = param.String('', precedence=-1, label='Reference End Time')
    yscale = param.ObjectSelector(objects=['linear', 'log'], label='Y Axis Scale', precedence=0.1)
    cmap = param.ObjectSelector(objects=cmaps, default='viridis', label='Colormap', precedence=0.1)
    endpoint = '/svc/anomaly'

    def __init__(self, **params):
        super().__init__(**params)
        self._update_ref_time_range()

    @param.depends('use_ref', watch=True)
    def _update_use_ref(self):
        if self.use_ref:
            self.nvars = 2
        else:
            self.nvars = 1
    
    @param.depends('ref_period', watch=True)
    def _update_ref_period(self):
        params = ['start_time2', 'end_time2']
        for param in params:
            if self.ref_period:
                self.param[param].precedence = 1
                
            else:
                self.param[param].precedence = -1

    @param.depends('use_ref', watch=True)
    def _update_ref_time_range(self):
        start, end = self.subsetter.time_range
        self.param['start_time2'].label = f'Reference Start Time (Earliest: {start}): '
        self.param['end_time2'].label = f'Reference End Time (Latest: {end}): '
        self.start_time2 = start
        self.end_time2 = end

    def _postprocess_data(self, ds):
        datasets = [selector.dataset for selector in self.dataset_selectors]
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        ds = ds.assign_coords(time=times)
        ds['time'] = times
        return ds.dropna(dim='time', how='all')
        
    @property
    def figure(self):
        v = self.query['var1']
        logy = self.yscale == 'log'
        area_mean = self.ds.weighted(np.cos(np.deg2rad(self.ds.lat))).mean(('lon', 'lat'))
        f1 = area_mean.hvplot.line(x='time', y=v, width=800, height=400, legend='bottom', logy=logy)
        f2 = self.ds.hvplot.quadmesh('lon', 'lat', v, title=f'{v} Anomaly',
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, rasterize=True,
                                     widget_location='bottom', cmap=self.cmap)
        return pn.Column(f1, f2)
    
    @property
    def query(self):
        query = dict(**super().query)
        query['removeSeason'] = 1 if self.reference_to_remove == 'seasonal cycle' else 0
        query['useVar2'] = int(self.nvars > 1)
        query['useTime2'] = int(self.ref_period)
        query['timeS2'] = self.start_time2
        query['timeE2'] = self.end_time2
        if 'model2' not in query:
            keys = ['model', 'var', 'pres']
            for k in keys:
                query[f'{k}2'] = query[f'{k}1']
        return query    
    
    
class MapViewService(Service):
    selector_cls = DatasetMonthSelector
    subsetter_cls = SpatialSubsetter
    time_prefix = 'v'
    month_prefix = 'v'
    endpoint = '/svc/mapView'
    
    def _postprocess_data(self, ds):
        if self.nvars > 1:
            vnames = {self.query[f'var{self.nvars}'] + f'_{(i+1)}': self.v(i+1) for i in range(self.nvars)}
            ds = ds.rename(**vnames)
        return ds
    
    @property
    def figure(self):
        if self.nvars == 1:
            cmap = self.all_selectors[0].cmap
            return self.ds.hvplot.quadmesh('longitude', 'latitude', self.query['var1'],
                                            title=self.query['var1'], geo=True, cmap=cmap,
                                            projection=ccrs.PlateCarree(), crs=ccrs.PlateCarree(), 
                                            coastline=True, width=800, rasterize=True)
        figures = []
        for i in range(1, self.nvars+1):
            v = self.v(i)
            cmap = self.all_selectors[i-1].cmap
            f = self.ds.hvplot.quadmesh(f'longitude_{i}', f'latiitude_{i}', v,
                                         title=v, geo=True, projection=ccrs.PlateCarree(),
                                         crs=ccrs.PlateCarree(), coastline=True, cmap=cmap,
                                         width=800, rasterize=True)
            figures.append((v, f))
        return pn.Tabs(*figures)
    
    @property
    def query(self):
        query = dict(**super().query)
        query['scale'] = 0
        query['nVar'] = self.nvars - 1
        return query

    
class ConditionalSamplingService(Service):
    target_name = 'Physical (Sampled) Variable'
    target_selector_cls = DatasetPresRangeSelector
    selector_cls = DatasetSamplingSelector
    subsetter_cls = SpatialSeasonalSubsetter
    nvars = param.Integer(1, bounds=(1, 2), label='Number Of Variables')
    ntargets = param.Integer(1, bounds=(0, 1), precedence=-1)    
    xscale = param.ObjectSelector(objects=['linear', 'log'], 
                                  label='X (Sampling Variable) Scale', precedence=0.1)
    yscale = param.ObjectSelector(objects=['linear', 'log'], 
                                  label='Y (Sampling Variable or Pressure) Scale', 
                                  precedence=0.1)
    cmap = param.ObjectSelector(objects=cmaps, default='viridis', label='Colormap', precedence=0.1)
    selector_names = ['Environmental Variable 1', 'Environmental Variable 2']
    endpoint = ['/svc/conditionalSampling', '/svc/conditionalSampling2Var']

    @property
    def html_base(self):
        if self.nvars == 2:
            return 'conditionalSampling2Var'
        return 'conditionalSampling'
    
    @property
    def figure(self):
        v1, v2 = self.query['var1'], self.query['var2'] + 'Bin'
        logx = self.xscale == 'log'
        logy = self.yscale == 'log'
        vs = f'{v1}_nSample'
        v1z, vsz = f'{v1}_z', f'{vs}_z'
        if self.nvars == 1:
            f1 = self.ds.hvplot.line(x=v2, y=v1, logx=logx, logy=logy, 
                                     width=800, height=400, title=f'{v1} sorted by {v2}')
            f2 = self.ds.hvplot.line(x=v2, y=vs, logx=logx, logy=logy, 
                                     width=800, height=400, title='Number of Samples')
            return pn.Column(f1, f2)
        else:
            ds = self.ds
            v3 = self.query['var3'] + 'Bin'
            dx, dy = f'{v2}z', f'{v3}z'
            x, y = np.meshgrid(ds[v2], ds[v3])
            z1 = replace_nans(ds[v1].values, 9)
            xx = ndi.zoom(x, 3, prefilter=False)
            yy = ndi.zoom(y, 3, prefilter=False)
            zz1 = ndi.filters.gaussian_filter(ndi.zoom(z1, 3, prefilter=False), 1.4)
            zz2 = ndi.filters.gaussian_filter(ndi.zoom(ds[vs], 3), 1.4)
            zz2[zz2 < 0] = 0
            ds = ds.assign_coords({dx: xx[0], dy: yy[:, 0]})
            ds[v1z] = xr.DataArray(dims=(dy, dx), data=zz1, attrs=ds[v1].attrs)
            ds[vsz] = xr.DataArray(dims=(dy, dx), data=zz2, attrs=ds[vs].attrs)
            lb, ub = np.percentile(ds[vsz], 5), np.percentile(ds[vsz], 95)
            f1 = ds.hvplot.quadmesh(dx, dy, v1z, rasterize=True, width=800, height=400, 
                                    cmap=self.cmap, xlabel=ds.x_labelStr, ylabel=ds.y_labelStr, 
                                    logx=logx, logy=logy,
                                    title=f'{v1} sorted by {v2} and {v3}')
            f2 = ds.hvplot.quadmesh(dx, dy, vsz, rasterize=True, width=800, height=400, 
                                    cmap=self.cmap, xlabel=ds.x_labelStr, ylabel=ds.y_labelStr,
                                    clim=(int(lb), int(ub)), title='Number of Samples', logx=logx, 
                                    logy=logy, logz=True)

            return pn.Column(f1, f2)
    
    @property
    def query(self):
        query = dict(**super().query)
        query.update(scale1=0, scale2=0, scale3=0)
        return query

    
class ZonalMeanService(Service):
    selector_cls = DatasetMonthSpatialSelector
    subsetter_cls = NullSubsetter
    nvars = param.Integer(1, bounds=(1, 8), label='Number Of Variables')
    yscale = param.ObjectSelector(objects=['linear', 'log'], label='Pressure Scale', precedence=0.1)
    xscale = param.ObjectSelector(objects=['linear', 'log'], label='Variable Scale', precedence=0.1)
    time_prefix = 'v'
    month_prefix = 'v'
    latlon_prefix = 'v'
    endpoint = '/svc/zonalMean'
    html_base = 'zonalMean8'

    def _postprocess_data(self, ds):
        if self.nvars < 2:
            ds = ds.rename(latitude='lat')
        else:
            ds = ds.rename(varIdx='Dataset')
        variable = self.query['var1']
        ds = ds.rename({variable: 'variable'})
        vnames = [f'{self.v(i+1)} ({sel.start_time}-{sel.end_time})' 
                  for i, sel in enumerate(self.all_selectors)]
        ds = ds.assign_coords(Dataset=vnames)
        return ds
    
    @property
    def figure(self):
        y = 'variable'
        logx = self.xscale == 'log'
        logy = self.yscale == 'log'
        ds = self.ds.copy()
        if self.nvars > 1:
            for i in range(2, self.nvars+1):
                if self.query[f'var{i}'] != self.query['var1']:
                    ds[y] = ((ds[y] - ds[y].min('lat')) / 
                             (ds[y].max('lat') - ds[y].min('lat')))
                    ds[y].attrs['units'] = '0-1'
                    ds = ds.rename({y: 'Normalized Variable'})
                    y = 'Normalized Variable'
        return ds.hvplot.line(x='lat', y=y, by='Dataset', logy=logy, logx=logx,
                              width=800, height=400, legend='bottom')
    
    @property
    def query(self):
        query = dict(**super().query)
        query['scale'] = 0
        query['nVar'] = self.nvars - 1
        return query

    
class VerticalProfileService(Service):
    subsetter_cls = SpatialSeasonalSubsetter
    three_dim_only = True
    endpoint = '/svc/threeDimVerticalProfile'
    html_base = 'threeDimVarVertical'
    
    @property
    def figure(self):
        ds = self.ds.expand_dims('dummy').assign_coords(plev=self.ds.plev*100)
        x = self.query['var1']
        start, end = self.query['timeS'], self.query['timeE']
        return (ds.hvplot.line(x=x, y='plev', width=800, height=400, 
                               ylabel='Pressure Level (hPa)', title=f'{start}-{end}')
                .opts(invert_yaxis=True))


class RegridService(Service):
    dlat, dlon = param.Parameter(1, label='dlon'), param.Parameter(1, label='dlon')
    yscale = param.ObjectSelector(objects=['linear', 'log'], label='Y Axis Scale', precedence=0.1)
    cmap = param.ObjectSelector(objects=cmaps, default='viridis', label='Colormap', precedence=0.1)
    endpoint = '/svc/regridAndDownload'
    
    @property
    def figure(self):
        v = self.query['var1']
        logy = self.yscale == 'log'
        area_mean = self.ds.weighted(np.cos(np.deg2rad(self.ds.lat))).mean(('lon', 'lat'))
        f1 = area_mean.hvplot.line(x='time', y=v, width=800, height=400, legend='bottom', logy=logy)
        f2 = self.ds.hvplot.quadmesh('lon', 'lat', v, title=f'{v} Anomaly',
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, rasterize=True,
                                     widget_location='bottom', cmap=self.cmap)
        return pn.Column(f1, f2)
    
    @property
    def query(self):
        query = dict(**super().query)
        query.update(dlat=self.dlat, dlon=self.dlon)
        return query
    
    
class RemoteFileService(param.Parameterized):
    url = param.String('', label='NetCDF File URL')
    decode_times = param.Boolean(False, label='Decode Times')
    npresses = param.Integer(0, precedence=-1)
    
    def __init__(self, **params):
        def press(event):
            self.npresses += 1
        self.button = pn.widgets.Button(name='Load Data', width=200)
        self.button.on_click(press)
        super().__init__(**params)
    
    def download_data(self):
        r = requests.get(self.url)
        buf = BytesIO(r.content)
        return xr.open_dataset(buf, decode_times=self.decode_times)
    
    @property
    def ds(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
        key = f'{self.url}/{self.decode_times}'
        if key not in self._cache:
            self._cache[key] = self.download_data()
        return self._cache[key]

    @param.depends('npresses')
    def xr(self):
        if not self.npresses:
            self._pane = pn.pane.HTML('', width=800)
            return self._pane
        self.button.disabled = True
        self.button.name = 'Working...'
        try:
            ds = self.ds
        except:
            self._pane.object = traceback.format_exc()
            ds = self._pane
        finally:
            self.button.disabled = False
            self.button.name = 'Load Data'
        return ds
    
    def panel(self):
        widgets = dict(url=dict(type=pn.widgets.TextAreaInput, width=600))
        return pn.Column(pn.Param(self.param, widgets=widgets), 
                         self.button, self.xr, height=1000)
    
    
class ServiceViewer:
    def __init__(self):
        self.svc = dict(
            time_series=TimeSeriesService(name='Time Series'),
            anomaly=AnomalyService(name='Anomaly'),
            scatter_hist=ScatterHistService(name='Scatter/Histogram'),
            random_forest=RandomForestService(name='Random Forest'),
            difference_plot=DifferencePlotService(name='Difference Map'),
            correlation_plot=CorrelationMapService(name='Correlation Map'),
            eof=EOFService(name='EOF'),
            joint_eof=JointEOFService(name='Joint EOF'),
            pdf=ConditionalPDFService(name='Conditional PDF'),
            map_view=MapViewService(name='Map View'),
            conditional_sampling=ConditionalSamplingService(name='Conditional Sampling'),
            zonal_mean=ZonalMeanService(name='Zonal Mean'),
            vertical_profile=VerticalProfileService(name='Vertical Profile'),
            regrid=RegridService(name='Regrid'),
            open_url=RemoteFileService(name='Open File URL')
        )
        for svc in self.svc.values():
            svc.viewer = self
        self._panels = {}
    
    def __getattr__(self, attr):
        return self.svc[attr]
                
    def view(self):
        return pn.Tabs(*[(svc.name, svc.panel()) for svc in self.svc.values()],
                       dynamic=True, tabs_location='right')
    
    def _save_mimebundle(self):
        obj = pn.Tabs(*[(k, v) for k, v in self._panels.items()],
                      tabs_location='right')
        obj.save('.cmda_data.html', resources='INLINE')

    @property
    def service_names(self):
        return list(self.svc.keys())