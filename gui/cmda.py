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

class Service(param.Parameterized):
    selector_names = None
    selector_cls = DatasetSelector
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
            self.dataset_selectors = [self.selector_cls(self, name=name) for name in self.selector_names]
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
            self.dataset_selectors.append(self.selector_cls(self, name=f'Dataset {i+1}'))
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
        return pn.Row(pn.Column(self.param, self.select_dataset,
                                self.subsetter, self.purpose,
                                self.plot_button), self.plot)
    
    def download_data(self):
        url = requests.get(self.url, params=self.query).json()['dataUrl']
        r = requests.get(url)
        buf = BytesIO(r.content)
        return self._postprocess_data(xr.open_dataset(buf))

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
        for i, selector in enumerate(self.all_selectors):
            mapper = dict(model=selector.dataset, var=selector.attrs['variable'], pres=selector.pressure)
            if 'latS' not in query:
                mapper.update(vlatS=selector.latitude_range[0], vlatE=selector.latitude_range[-1],
                              vlonS=selector.longitude_range[0], vlonE=selector.longitude_range[-1])
            for k, v in mapper.items():
                if k == 'model':
                    v = v.replace('/', '_')
                elif k == 'pres' and v == 'N/A':
                    v = -999999
                k += str(i + 1)
                query[k] = v
        return query


class TimeSeriesService(Service):
    selector_cls = DatasetSubsetSelector
    endpoint = '/svc/timeSeries'
    
    def _postprocess_data(self, ds):
        datasets = [selector.dataset for selector in self.dataset_selectors]
        times = gen_time_range(self.subsetter.start_time, self.subsetter.end_time)
        return (ds.rename(varIdx='Dataset', monthIdx='time')
                  .assign_coords(Dataset=datasets, time=times))
    @property
    def figure(self):
        y = self.query['var1']
        return self.ds.hvplot.line(x='time', y=y, by='Dataset', width=1000, height=500)
    
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
                                     width=800, height=400)
        f2 = self.ds.hvplot.quadmesh('lon', 'lat', v1, title=v1,
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, height=400)
        f3 = self.ds.hvplot.quadmesh('lon', 'lat', v2, title=v2,
                                     geo=True, projection=ccrs.PlateCarree(),
                                     crs=ccrs.PlateCarree(), coastline=True,
                                     width=800, height=400)
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
        ds = ds.rename(index='EOF')
        return ds
        
    @property
    def figure(self):
        f1 = self.ds.varP.hvplot.line(x='EOF', y='varP',
                                      title='Variance Explained (%)')
        f2 = self.ds.patterns.hvplot.quadmesh('lon', 'lat', title='EOF',
                                          widget_location='bottom',
                                          projection=ccrs.PlateCarree(),
                                          crs=ccrs.PlateCarree(), geo=True,
                                          coastline=True)
        f3 = self.ds.tser.hvplot.line(x='time', y='tser', title='PC',
                                      widget_location='bottom')
        return pn.Column(f1, f2, f3)

    @property
    def query(self):
        query = dict(**super().query)
        query['anomaly'] = int(self.anomaly)
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
                                            height=500)

    @property
    def query(self):
        query = dict(**super().query)
        query['laggedTime'] = int(self.lag)
        return query
