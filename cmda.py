from io import BytesIO
import traceback
import cartopy.crs as ccrs
import numpy as np
import param
import pandas as pd
import datetime as dt
import panel as pn
import requests
import xarray as xr
import hvplot.xarray
import hvplot.pandas
import holoviews as hv
from holoviews import opts

datasets = pd.read_csv('datasets.csv').drop_duplicates()
variables = pd.read_csv('variables.csv')
names = datasets.dataset_name
categories = datasets.category.unique()
dset_dict = {cat: list(names[datasets.category == cat].unique()) for cat in categories}

def get_variables(dataset_name):
    return datasets.long_name[names == dataset_name].dropna().unique()

def get_attrs(dataset_name, variable_name):
    return datasets[(names==dataset_name) & (datasets.long_name==variable_name)].to_dict(orient='records')[0]

def gen_time_range(st, et):
    return pd.date_range(st, et, freq='MS')

class SpatialSubsetter(param.Parameterized):
    latitude_range = param.Range(default=(-90, 90), bounds=(-90, 90))
    longitude_range = param.Range(default=(0, 360), bounds=(0, 360))
    
class TemporalSubsetter(param.Parameterized):
    start_time = param.String('')
    end_time = param.String('')
    def __init__(self, service, **params):
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
        start_times = [selector.attrs['start'] for selector in self.service.all_selectors]
        end_times = [selector.attrs['end'] for selector in self.service.all_selectors]
        st, et = str(max(start_times)), str(min(end_times))
        return f'{st[:4]}-{st[4:]}', f'{et[:4]}-{et[4:]}'

class Subsetter(SpatialSubsetter, TemporalSubsetter):
    pass

class DatasetSelector(param.Parameterized):
    defaults = get_variables('GFDL/ESM2G')
    category = param.ObjectSelector(objects=list(dset_dict.keys()))
    dataset = param.ObjectSelector(default='GFDL/ESM2G', objects=dset_dict['Model: Historical'])
    variable = param.ObjectSelector(default=defaults[0], objects=defaults)
    pressure = param.String('N/A', constant=True)
    
    def __init__(self, service, **params):
        self.service = service
        super().__init__(**params)
    
    @param.depends('category', watch=True)
    def _update_datasets(self):
        self.param['dataset'].objects = dset_dict[self.category]
        self.dataset = list(dset_dict[self.category])[0]

    @param.depends('dataset', watch=True)
    def _update_variable(self):
        self.param['variable'].objects = get_variables(self.dataset)
        self.variable = get_variables(self.dataset)[0]

    @param.depends('category', 'dataset', 'variable', watch=True)
    def _update_subsetter(self):
        self.service.subsetter._update_time_ranges()
        if hasattr(self.service, 'start_time2'):
            self.service._update_ref_time_range()
        
    @param.depends('variable', watch=True)
    def _update_pressure(self):
        if self.attrs['dimensions'] > 2:
            self.param['pressure'].constant = False
            self.pressure = '500'
        elif not self.param['pressure'].constant:
                self.pressure = 'N/A'
                self.param['pressure'].constant = True
    
    @property
    def attrs(self):
        return get_attrs(self.dataset, self.variable)

class DatasetSubsetSelector(SpatialSubsetter, DatasetSelector):
    pass

class DatasetAnomalySelector(DatasetSubsetSelector):
    anomaly = param.Boolean(False, label='Use Anomaly')

class DatasetBinsSelector(DatasetSelector):
    nbins = param.Integer(10, label='Number of Bins')

class Service(param.Parameterized):
    selector_names = None
    selector_cls = DatasetSelector
    latlon_prefix = ''
    latlon_suf = True
    ntargets = param.Integer(0, bounds=(0, 1), precedence=-1)
    nvars = param.Integer(1, bounds=(1, 6), label='Number Of Variables')
    npresses = param.Integer(0, precedence=-1)
    host = 'http://ec2-52-53-95-229.us-west-1.compute.amazonaws.com:8080'
    endpoint = '/'
    
    def __init__(self, **params):
        self.target_selector = self.selector_cls(self, name='Target Variable')
        if issubclass(self.selector_cls, SpatialSubsetter):
            subsetter_cls = TemporalSubsetter
        else:
            subsetter_cls = Subsetter
        if not self.selector_names:
            self.dataset_selectors = [self.selector_cls(self, name='Dataset 1')]
        else:
            self.param['nvars'].precedence = -1
            self.dataset_selectors = [
                self.selector_cls(self, name=self.selector_names[i])
                for i in range(self.nvars)
            ]
        self.subsetter = subsetter_cls(self, name='Subsetting Options')
        self.purpose = pn.widgets.TextAreaInput(name='Execution Purpose',
                                                placeholder='Describe execution purpose here (Optional)')
        def press(event):
            self.npresses += 1
        self.plot_button = pn.widgets.Button(name='Generate Plot')
        self.plot_button.on_click(press)
        super().__init__(**params)
    
    @param.depends('nvars', watch=True)
    def _update_datasets(self):
        while len(self.dataset_selectors) > self.nvars:
            self.dataset_selectors.pop(-1)
        for i in range(len(self.dataset_selectors), self.nvars):
            name = self.selector_names[i] if self.selector_names else f'Dataset {i+1}'
            self.dataset_selectors.append(self.selector_cls(self, name=name))
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
        except:
            self._pane.object = traceback.format_exc()
            figure = self._pane
        finally:
            self.plot_button.disabled = False
            self.plot_button.name = 'Generate Plot'
        return figure
    
    def panel(self):
        kwargs = {}
        if hasattr(self, 'widgets'):
            kwargs['widgets'] = self.widgets
        return pn.Row(pn.Column(pn.Param(self.param, **kwargs),
                                self.select_dataset, self.subsetter,
                                self.purpose, self.plot_button), self.plot)
    
    def download_data(self):
        url = requests.get(self.url, params=self.query).json()['dataUrl']
        r = requests.get(url)
        buf = BytesIO(r.content)
        return self._postprocess_data(xr.open_dataset(buf, decode_times=False))

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
        return self.host + self.endpoint
    
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
        query = dict(timeS=self.subsetter.start_time.replace('-', ''),
                     timeE=self.subsetter.end_time.replace('-', ''),
                     purpose=self.purpose.value)
        if not isinstance(self.dataset_selectors[0], SpatialSubsetter):
            query.update(latS=self.subsetter.latitude_range[0],
                         latE=self.subsetter.latitude_range[-1],
                         lonS=self.subsetter.longitude_range[0],
                         lonE=self.subsetter.longitude_range[-1])
        latlon_basenames = ['latS', 'latE', 'lonS', 'lonE']
        latlon_names = [self.latlon_prefix + name for name in latlon_basenames]
        for i, selector in enumerate(self.all_selectors):
            mapper = dict(model=selector.dataset, var=selector.attrs['variable'], pres=selector.pressure)
            if 'latS' not in query:
                latlon_vals = [*selector.latitude_range, *selector.longitude_range]
                latlon_vmap = dict(zip(latlon_names, latlon_vals))
                mapper.update(**latlon_vmap)
            for k, v in mapper.items():
                if k == 'model':
                    v = v.replace('/', '_')
                elif k == 'pres' and v == 'N/A':
                    v = -999999
                if self.latlon_suf and k in latlon_basenames:
                    k = k[:-1] + str(i + 1) + k[-1]
                else:
                    k += str(i + 1)
                query[k] = v
        return query


class TimeSeriesService(Service):
    selector_cls = DatasetSubsetSelector
    endpoint = '/svc/timeSeries'
    latlon_prefix = 'v'
    latlon_suf = False
    
    def _postprocess_data(self, ds):
        datasets = [self.v(i) for i in range(1, self.nvars+1)]
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        ds = (ds.rename(varIdx='Dataset', monthIdx='time')
                 .assign_coords(Dataset=datasets, time=times))
        ds['time'] = times
        return ds
        
    @property
    def figure(self):
        ds = self.ds.copy()
        y = self.query['var1']
        if self.nvars > 1:
            for i in range(2, self.nvars+1):
                if self.query[f'var{i}'] != y:
                    ds[y] = ((ds[y] - ds[y].min('time')) / 
                             (ds[y].max('time') - ds[y].min('time')))
                    ds[y].attrs['units'] = '0-1'
                    ds = ds.rename({y: 'Normalized Variable'})
                    y = 'Normalized Variable'
                    break
        return ds.hvplot.line(x='time', y=y, by='Dataset', 
                              legend='bottom', width=1000, height=500)
    
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
                  .scatter(x=v1, y=v2, title=f'Correlation: {self.ds.corr:1.2}', width=1000, height=300))
        f2 = self.ds.hvplot.hist(y=v1, width=1000, height=300, normed=True)
        f3 = self.ds.hvplot.hist(y=v2, width=1000, height=300, normed=True)
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
                                     width=800, height=400, rasterize=True)
        f2 = self.ds.hvplot.quadmesh('lon', 'lat', v1, title=v1,
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, height=400, rasterize=True)
        f3 = self.ds.hvplot.quadmesh('lon', 'lat', v2, title=v2,
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, height=400, rasterize=True)
        return pn.Column(f1, f2, f3)

    @property
    def query(self):
        query = dict(**super().query)
        for i in range(1, 4):
            query[f'colorMap{i}'] = 'rainbow'
        return query

    
class RandomForestService(Service):
    endpoint = '/svc/randomForest'
    nvars = param.Integer(1, bounds=(1, 10), label='Source Variables')
    ntargets = param.Integer(1, bounds=(0, 1), precedence=-1)
    
    def _postprocess_data(self, ds):
        vs = self.v(1)
        vt = [self.v(i+2) for i in range(self.nvars)]
        ds = ds.assign_coords(index=vt).rename(importance=f'importance to predict {vs}')
        return ds
        
    @property
    def figure(self):
        vs = self.v(1)
        y = f'importance to predict {vs}'
        return self.ds.hvplot.bar(x='index', y=y, width=1000, height=500)

    @property
    def query(self):
        query = dict(**super().query)
        query['nVar'] = self.nvars
        return query


class EOFService(Service):
    selector_names = ['Data']
    nvars = param.Integer(1, precedence=-1)
    anomaly = param.Boolean(False, label='Use Anomaly')
    endpoint = '/svc/EOF'
    
    def _postprocess_data(self, ds):
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        ds = ds.rename(index='EOF').assign_coords(time=times)
        ds['varP'] *= 100
        return ds
        
    @property
    def figure(self):
        f1 = self.ds.varP.hvplot.line(x='EOF', y='varP',
                                      title='Variance Explained (%)')
        f2 = self.ds.patterns.hvplot.quadmesh('lon', 'lat', title='EOF',
                                          widget_location='bottom',
                                          projection=ccrs.PlateCarree(),
                                          crs=ccrs.PlateCarree(), geo=True,
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
    endpoint = '/svc/JointEOF'
    
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
                                            widget_location='bottom',
                                            projection=ccrs.PlateCarree(),
                                            crs=ccrs.PlateCarree(), geo=True,
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
    endpoint = '/svc/correlationMap'
            
    @property
    def figure(self):
        return self.ds.corr.hvplot.quadmesh('lon', 'lat',
                                            projection=ccrs.PlateCarree(),
                                            crs=ccrs.PlateCarree(), geo=True,
                                            coastline=True, width=1000,
                                            height=500, rasterize=True)

    @property
    def query(self):
        query = dict(**super().query)
        query['laggedTime'] = int(self.lag)
        return query

class ConditionalPDFService(Service):
    selector_names = ['Independent Variable', 'Dependent Variable']
    selector_cls = DatasetBinsSelector
    nvars = param.Integer(2, precedence=-1)
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
            submesh = hv.QuadMesh((x, y, z), vdims=['pdf'], kdims=[v1, v2])
            meshes.append(submesh)
        mesh = hv.Overlay(meshes) * curve
        return mesh.opts(opts.QuadMesh(colorbar=True, width=800, height=400,
                                       tools=['hover'], cmap='jet'))

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
        area_mean = self.ds.weighted(np.cos(np.deg2rad(self.ds.lat))).mean(('lon', 'lat'))
        f1 = area_mean.hvplot.line(x='time', y=v, width=800, height=400, legend='bottom')
        f2 = self.ds.hvplot.quadmesh('lon', 'lat', v, title=f'{v} Anomaly',
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, height=400, rasterize=True,
                                     widget_location='bottom')
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


class ServiceViewer:
    def __init__(self):
        self.svc = dict(
            time_series=TimeSeriesService(name='Time Series'),
            anomaly=AnomalyService(name='Anomaly'),
            scatter_hist=ScatterHistService(name='Scatter and Histogram'),
            random_forest=RandomForestService(name='Random Forest Importance'),
            difference_plot=DifferencePlotService(name='Difference Map'),
            correlation_plot=CorrelationMapService(name='Correlation Map'),
            eof=EOFService(name='EOF'),
            joint_eof=JointEOFService(name='Joint EOF'),
            pdf=ConditionalPDFService(name='Conditional PDF'),
        )
    
    def __getattr__(self, attr):
        return self.svc[attr]
                
    def view(self):
        return pn.Tabs(*[(svc.name, svc.panel()) for svc in self.svc.values()])

    @property
    def service_names(self):
        return list(self.svc.keys())
