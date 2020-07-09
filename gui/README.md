# CMDA GUI for Jupyter

![img](https://puu.sh/G5iCA/89ea6c3097.png)

## Implementing your own CMDA Service
This notebook uses the [panel](https://panel.holoviz.org/) library to generate the GUI elements for each CMDA Service. 
Reading through the documentation first is highly recommended.


To create a GUI for a CMDA Service, create a subclass of `Service` in `cmda.py`. You'll need to override the following methods/properties:

### Class Variables
You may optionally override the following class variables:
- `selector_names`: All input datasets are separated by tabs. This variable lets you override the default labels in tabs.
- `selector_cls`: What class to use for the dataset selection widget.
- `ntargets`: Override this if your service separates input datasets into two groups (eg, reference vs target datasets)
- `nvars`: The number of input variables/datasets. Override if the service requires more than one variable.
- `endpoint`: The REST API endpoint for the Service.

`Service` is also a subclass of `param.Parameterized`, so you can add additional GUI elements with additional class variables.

Example:

```python
class EOFService(Service):
    selector_names = ['Data']
    nvars = param.Integer(1, precedence=-1)
    anomaly = param.Boolean(False, label='Use Anomaly')
    endpoint = '/svc/EOF'
```

### Generating the REST API query
The `query` accessor defines how the values stored in the GUI elements get mapped to the query parameters that get passed to the API call. 
The `Service` base class handles most of these, but you will likely need to override this to set additonal parameters specific to the service: 

Example:
```python
    @property
    def query(self):
        query = dict(**super().query)
        query['anomaly'] = int(self.anomaly)
        return query
 ```

### Postprocessing
After the query is formed, the next step is downloading the data from the server into an `xarray.Dataset` object. 
However, it's usually desirable to apply additonal postprocessing to the data which typically entails renaming fields (which can help make the plots nicer)
and/or applying addtional analyses. 

Example:
```python    
    def _postprocess_data(self, ds):
        ds = ds.rename(index='EOF')
        return ds
```

### Plotting
Override the `figure` accessor to determine how the plots should look. Use the `ds` accessor to access the underlying xarray Dataset. 
The return value should be a panel compatible repr, such as `hvplot` objects (with Bokeh). 
If the plot is a matplotlib figure, use a [Matplotlib Pane](https://panel.holoviz.org/reference/panes/Matplotlib.html)

Example:
```python
    @property
    def figure(self):
        f1 = self.ds.patterns.hvplot.quadmesh('lon', 'lat', title='EOF',
                                          widget_location='bottom',
                                          projection=ccrs.PlateCarree(),
                                          crs=ccrs.PlateCarree(), geo=True,
                                          coastline=True)
        f2 = self.ds.tser.hvplot.line(x='time', y='tser', title='PC',
                                      widget_location='bottom')
        return pn.Column(f1, f2)
```
### Custom Dataset Selectors
The default `DatasetSelector` widget includes the basic options for providing the dataset name, variable, and pressure level. 
To add GUI options specific to each dataset, you will need to create a subclass of `DatasetSelector` and then specify it in the `selector_cls` for your
`Service`. For example, the `TimeSeriesService` allows for setting the lat/lon bounds individually for each dataset by doing the following:

```python
class SpatialSubsetter(param.Parameterized):
    latitude_range = param.Range(default=(-90, 90), bounds=(-90, 90))
    longitude_range = param.Range(default=(0, 360), bounds=(0, 360))

class DatasetSubsetSelector(SpatialSubsetter, DatasetSelector):
    pass
```
