"""
Class that builds grid based on either an explicit list of points,
grid parameters, and/or a polygonal shaped license area.
The license area can be redefined, with possible grid adjustments (expand, restrict) during the call.
The call can return all grid points, or only grid points inside the polygon (if trainfull=False),
with well points excluded.
"""
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, Union, Optional, List
import warnings

# Determines if point(s) with coordinate(s) x,y are inside/outside polygon
# using ray-tracing algorithm; returns boolean mask. Borrowed from (last answer)
# https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# copy inside GridBuilder
def ray_tracing(x: np.ndarray, y: np.ndarray, poly):
    n = len(poly)  # n >= 1 points in closed polygon and n edges
    inside = np.zeros(len(x), np.bool_)
    p1x, p1y = poly[0]
    for i in range(1, n + 1):  # for each edge
        p2x, p2y = poly[i % n]
        # print("ray-tracing step",i,": edge",p1x,p1y,"<->",p2x,p2y)
        # locate points in the region to the left of the edge (rays directed || -i)
        idx = np.nonzero((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))[0]
        if p1y != p2y:  # if edge is not horizontal,
            xints = (p2x - p1x) / (p2y - p1y) * (y[idx] - p1y) + p1x  # determine x-intercepts of edge
        if p1x == p2x:  # if edge is vertical,
            inside[idx] = ~inside[idx]  # flip entire region (possibly empty)
        elif idx.size > 0:  # else if region is not empty,
            idxx = idx[x[idx] <= xints]
            inside[idxx] = ~inside[idxx]  # flip on or to the left of the edge
        # else if region is empty (horizontal edges or other reasons), do nothing
        p1x, p1y = p2x, p2y
    return inside


# Unmodified
def build_grid(df_learn: pd.DataFrame, x_max_grid, y_max_grid, x_min_grid, y_min_grid, x_step=1, y_step=1):
    x_grid = range(int(x_min_grid), int(x_max_grid) + 1, x_step)
    y_grid = range(int(y_min_grid), int(y_max_grid) + 1, y_step)

    if df_learn is None or df_learn.shape[0] == 0:
        return np.array([[x, y] for x, y in product(x_grid, y_grid)])
    else:
        return np.array(
            [[x, y] for x, y in product(x_grid, y_grid)
             if not ((df_learn['X'].values == x) & (df_learn['Y'].values == y)).any()]
        )


class GridBuilder:
    eps = 1e-6  # small number (much less than possible step size)

    def __init__(self, grid_points: Optional[
        Union[np.ndarray, List[List[int]], List[List[float]], pd.DataFrame, Dict]] = None,
                 grid_params: Optional[Union[List[int], List[float], np.ndarray]] = None,
                 polygon=None, forceint=False,
                 x_step=0, y_step=0, buffer=0, num: Optional[int] = 31, equalstep=True):
        # Format of grid_params: [x_max, y_max, x_min, y_min, x_step(optional), y_step(optional)]
        # Last line of kwargs are are optional and passed to grid_params_polygon
        # print("init args:",grid_params, x_step, y_step, forceint, buffer, num, equalstep)

        if not grid_points is None:  # use specified grid points
            if isinstance(grid_points, (dict, pd.DataFrame)):
                if all(label in grid_points.keys() for label in ['X', 'Y']):
                    self._grid_points = np.array([grid_points['X'], grid_points['Y']]).T
                else:
                    raise ValueError('unexpected coordinate labels: not X, Y')
            elif isinstance(grid_points, (list, np.ndarray)):
                if isinstance(grid_points, list):
                    grid_points = np.array(grid_points)
                if not self.isnum(grid_points, nofloat=False):
                    raise TypeError('point list is not numeric')
                if len(grid_points.shape) == 2:
                    if grid_points.shape[1] == 2 and grid_points.shape[0] > 0:
                        self._grid_points = grid_points
                    else:
                        raise TypeError('incorrect point list dimensions')
                else:
                    raise TypeError('incorrect point list shape')
            else:
                raise TypeError('unrecognized point list format')
            self._grid_params = self.polygon_bounds(self._grid_points) + [1, 1]
            self._intgrid = self.isnum(self._grid_points)
        elif not grid_params is None:  # make grid using specified grid_params
            if not self.isnum(grid_params, nofloat=False):
                raise TypeError('grid parameters are not numeric')
            if len(grid_params) >= 4 and len(grid_params) <= 6:
                if grid_params[0] > grid_params[2] and grid_params[0] > grid_params[2]:
                    self.make_grid(*grid_params, forceint=forceint)
                else:
                    raise ValueError('grid parameters inconsistent')
            else:
                raise ValueError('wrong number of grid parameters')
        elif not polygon is None:  # make grid using specified polygon
            self._buffer = buffer
            self.make_grid(*self.grid_params_polygon(polygon, x_step, y_step, forceint,
                                                     buffer, num, equalstep), forceint=forceint)
        else:
            raise ValueError('cannot initialize with no inputs (grid_points, grid_params, or polygon)')

        if not polygon is None:
            self._buffer = buffer
            if not self.fits_grid(polygon, buffer):
                warnings.warn('GridBuilder - specified grid does not fit specified polygon')
                # print('GridBuilder warning! Specified grid does not fit specified polygon')
        # Set drillable points based on polygon after grid points are defined
        self._drillable_points = self.inpolygon(polygon)

    # Make rectangular grid (like in build_grid);
    # inception point (x_min, y_min) is included (except with forceint)
    # and (x_max, y_max) is always within the grid
    def make_grid(self, x_max, y_max, x_min, y_min, x_step=1, y_step=1, forceint=False):
        # Round grid parameters really close to integers
        x_max, y_max, x_min, y_min, x_step, y_step = [
            self.roundifclose(param, self.eps) for param in [x_max, y_max, x_min, y_min, x_step, y_step]]
        if x_step == 0 or y_step == 0:
            raise ValueError('cannot make grid: steps too small')
        self._intgrid = True
        if forceint:  # make all coordinates integers
            if isinstance(x_step, (int, np.integer)) and isinstance(y_step, (int, np.integer)):
                x_min, y_min = int(np.floor(x_min)), int(np.floor(y_min))
                x_max, y_max = int(np.ceil(x_max)), int(np.ceil(y_max))
                x_grid = range(x_min, x_max + x_step, x_step)
                y_grid = range(y_min, y_max + y_step, y_step)
            else:
                raise TypeError('cannot initialize with forced integers: steps must be ints')
        else:
            if all(isinstance(x_param, (int, np.integer)) for x_param in [x_step, x_max, x_min]):
                x_grid = range(x_min, x_max + x_step, x_step)
                # print(" x_grid:",list(x_grid))
            else:  # use eps to include max point in numpy arange when (max-min)/step is int
                x_grid = np.arange(x_min, x_max + x_step - self.eps, x_step)
                # print(" x_grid=",x_grid)
                self._intgrid = False
            if all(isinstance(y_param, (int, np.integer)) for y_param in [y_step, y_max, y_min]):
                y_grid = range(y_min, y_max + y_step, y_step)
                # print(" y_grid:",list(y_grid))
            else:
                y_grid = np.arange(y_min, y_max + y_step - self.eps, y_step)
                # print(" y_grid=",y_grid)
                self._intgrid = False
        self._grid_points = np.array([[x, y] for x, y in product(x_grid, y_grid)])
        self._grid_params = [x_max, y_max, x_min, y_min, x_step, y_step]
        # print(" make_grid:", *self._grid_params, "intgrid=", self._intgrid)

    @staticmethod
    def roundifclose(value, tolerance):
        if (not isinstance(value, (int, np.integer))) and abs(value - round(value)) < tolerance:
            return int(np.round(value))
        else:
            return value

    # Return rectangular grid parameters for polygon
    @classmethod
    def grid_params_polygon(cls, polygon: Union[np.ndarray, List[List[int]], List[List[float]], pd.DataFrame],
                            x_step=0, y_step=0,
                            forceint=False, buffer=0, num: Optional[int] = 31, equalstep=True):
        # optional arguments used when x_step, y_step are not specified:
        # num: default minimum number of points along either axis
        # equalstep: do the steps have to be equal?
        # buffer: amount of space between polygon limits and grid edge
        # forceint: rounds steps to nearest integers
        # print(" gpp args:", polygon, "\n ",
        #       x_step, y_step, forceint, buffer, num, equalstep)

        # Find polygon bounds
        [x_max, y_max, x_min, y_min] = cls.polygon_bounds(polygon, buffer)
        # If y_step is not given, assume it is same as x_step
        if x_step != 0 and y_step == 0:
            y_step = x_step
        # Determine x_step and y_step if necessary
        if x_step == 0 or y_step == 0:
            if num > 1:
                if x_step == 0:
                    if x_max > x_min:
                        x_step = (x_max - x_min) / (num - 1)
                    else:
                        raise ValueError('cannot initialize: polygon is vertical')
                if y_step == 0:
                    if y_max > y_min:
                        y_step = (y_max - y_min) / (num - 1)
                    else:
                        raise ValueError('cannot initialize: polygon is horizontal')
            else:
                raise ValueError('cannot initialize: too few steps, num < 2')
            if equalstep:  # set both steps to their minimum
                x_step, y_step = [min(x_step, y_step)] * 2
            if forceint:  # round steps to nearest integer
                if x_step > 0.5 and y_step > 0.5:
                    x_step, y_step = int(np.round(x_step)), int(np.round(y_step))
                else:
                    raise ValueError('cannot force int: steps too small')
        # Check proposed steps
        nx = (x_max - x_min) / x_step + 1
        ny = (y_max - y_min) / y_step + 1
        if nx < 2 or ny < 2: raise ValueError('cannot initialize: need smaller steps')
        # Extend maximum points to include entire polygon & buffer - now done in make_grid
        # if nx % x_step > cls.eps: x_max = x_min + int(nx) * x_step
        # if ny % y_step > cls.eps: y_max = y_min + int(ny) * y_step

        # print(" gpp:", x_max, y_max, x_min, y_min, x_step, y_step)
        return [x_max, y_max, x_min, y_min, x_step, y_step]

    # Determine polygon boundaries with buffer adjustment (first to encounter, check polygon)
    @classmethod
    def polygon_bounds(cls, polygon: Union[np.ndarray, List[List[float]], List[List[int]], pd.DataFrame],
                       buffer=0):
        if not cls.isnum(polygon, nofloat=False):
            raise TypeError('polygon is not numeric')
        apolygon = np.array(polygon)
        if buffer is None: buffer = 0  # if undefined, also set to default
        if len(apolygon.shape) == 2:
            if apolygon.shape[1] == 2 and apolygon.shape[0] >= 2:
                try:
                    x_max = apolygon.T[0].max() + buffer
                    y_max = apolygon.T[1].max() + buffer
                    x_min = apolygon.T[0].min() - buffer
                    y_min = apolygon.T[1].min() - buffer
                except:
                    raise ValueError('cannot find limits of polygon')  # prevented by numerical check
            else:
                raise TypeError('incorrect polygon dimensions')
        else:
            raise TypeError('incorrect polygon shape')
        return [x_max, y_max, x_min, y_min]

    # Returns bool: Does polygon fit inside existing grid?
    def fits_grid(self, polygon: Union[np.ndarray, List[List[float]], List[List[int]], pd.DataFrame],
                  buffer=0):
        [x_max, y_max, x_min, y_min] = self.polygon_bounds(polygon, buffer)
        if self._grid_params[0] >= x_max:
            if self._grid_params[2] <= x_min:
                if self._grid_params[1] >= y_max:
                    if self._grid_params[3] <= y_min:
                        return True
        return False

    # Centralized check if given object consists of int (or float, when nofloat=False)
    @staticmethod
    def isnum(var, nofloat=True):
        var_dtype = np.array(var).dtype
        if nofloat:
            return np.issubdtype(var_dtype, np.integer)
        else:
            return np.issubdtype(var_dtype, np.floating) or np.issubdtype(var_dtype, np.integer)

    @property
    def grid(self):
        return self._grid_points

    @property
    def df_grid(self):
        return pd.DataFrame({'X': self._grid_points.T[0], 'Y': self._grid_points.T[1]})

    # Return array of points excluding those whose coordinates are in df_learn (wells)
    def __call__(self, well_points: Optional[
        Union[np.ndarray, List[List[int]], List[List[float]], pd.DataFrame, Dict]] = None,
                 polygon=None, buffer=None,
                 expand=True, restrict=False, trainfull=True):
        # New polygon is specified: possible reinitialization
        if not polygon is None:
            if buffer is None and not self._buffer is None:
                buffer = self._buffer
            # Alter grid (expand or restrict) using newpolygon and newbuffer
            if (expand and not self.fits_grid(polygon, buffer)) or restrict:
                # print(" polygon=",polygon,"polygon_bounds=",self.polygon_bounds(polygon, buffer))
                new_grid_params = self.grid_params_polygon(polygon,
                                                           self._grid_params[4], self._grid_params[5],
                                                           forceint=self._intgrid, buffer=buffer)
                # print(" expand=",expand,"intgrid=",self._intgrid,"buffer=",buffer,"bound gp:",new_grid_params)
                if expand and not restrict:  # expand grid
                    new_grid_params[:4] = [
                        max(new_grid_params[0], self._grid_params[0]),
                        max(new_grid_params[1], self._grid_params[1]),
                        min(new_grid_params[2], self._grid_params[2]),
                        min(new_grid_params[3], self._grid_params[3])]
                # print(" new gp:", new_grid_params, " old gp:", self._grid_params)
                self.make_grid(*new_grid_params, forceint=self._intgrid)
            # Set drillable points based on new polygon after grid is remade
            self._drillable_points = self.inpolygon(polygon)

        # Check list of excluded (drilled) points - same check as grid_points in init
        if not well_points is None:  # use specified grid points
            if isinstance(well_points, (dict, pd.DataFrame)):
                if all(label in well_points.keys() for label in ['X', 'Y']):
                    well_points = np.array(well_points[['X', 'Y']])
                else:
                    raise ValueError('unexpected coordinate labels: not X, Y')
            elif isinstance(well_points, (list, np.ndarray)):
                if isinstance(well_points, list):
                    well_points = np.array(well_points)
                if not self.isnum(well_points, nofloat=False):
                    raise TypeError('point list is not numeric')
                if len(well_points.shape) == 2:
                    if not (well_points.shape[1] == 2 and well_points.shape[0] > 0):
                        raise TypeError('incorrect point list dimensions')
                else:
                    raise TypeError('incorrect point list shape')
            else:
                raise TypeError('unrecognized point list format')

        # Return train_points with well_points excluded
        if trainfull:
            train_points = self._grid_points
        else:
            train_points = self._drillable_points
        if well_points is None:
            return train_points
        else:
            # regular way (simple via list comprehension, slow on big grids)
            return np.array([p for p in train_points
                             if not np.any((well_points.T[0] == p[0]) & (well_points.T[1] == p[1]))])

            # alternative 1: numpy way, fast but only works for integer-valued arrays
            # dims = np.maximum(train_points.max(0),well_points.max(0))+1
            # return well_points[~np.in1d(np.ravel_multi_index(well_points.T,dims),np.ravel_multi_index(train_points.T,dims))]

            # alternative 2: fast but weird pandas way
            # pd_train_points = pd.DataFrame({'X':train_points.T[0],'Y':train_points.T[1]})
            # pd_well_points = pd.DataFrame({'X':well_points.T[0],'Y':well_points.T[1]})
            # return np.array(pd.concat([pd_train_points,pd_well_points,pd_well_points]).drop_duplicates(keep=False))

    # Construct inside mask for given polygon to restrict drillable points
    def inpolygon(self, polygon: Optional[Union[np.ndarray, List[List[float]], List[List[int]], pd.DataFrame]]):
        if polygon is None:
            self._polygon = None
            self._buffer = None
            return self._grid_points
        else:
            # Checking of polygon done in polyugon_bounds before inpolygon call
            self._polygon = np.array(polygon)
            inside = self.ray_tracing(self._grid_points.T[0],
                                      self._grid_points.T[1], self._polygon)
            if inside.sum() == 0:
                warnings.warn('GridBuilder - polygon area is empty')
                # print('GridBuilder warning! Polygon area is empty')
            # later add minimum number of points requirement
            return self._grid_points[inside]

    # Returns boolean mask of point(s) with coordinate(s) inside the polygon
    @staticmethod
    def ray_tracing(x: np.ndarray, y: np.ndarray, poly):
        n = len(poly)  # n >= 1 points in closed polygon and n edges
        inside = np.zeros(len(x), np.bool_)
        p1x, p1y = poly[0]
        for i in range(1, n + 1):  # for each edge
            p2x, p2y = poly[i % n]
            # print("ray-tracing step",i,": edge",p1x,p1y,"<->",p2x,p2y)
            # locate points in the region to the left of the edge (rays directed || -i)
            idx = np.nonzero((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))[0]
            if p1y != p2y:  # if edge is not horizontal,
                xints = (p2x - p1x) / (p2y - p1y) * (y[idx] - p1y) + p1x  # determine x-intercepts of edge
            if p1x == p2x:  # if edge is vertical,
                inside[idx] = ~inside[idx]  # flip entire region (possibly empty)
            elif idx.size > 0:  # else if region is not empty,
                idxx = idx[x[idx] <= xints]
                inside[idxx] = ~inside[idxx]  # flip on or to the left of the edge
            # else if region is empty (horizontal edges or other reasons), do nothing
            p1x, p1y = p2x, p2y
        return inside

    # Return df_metric (must include coordinates) restricted to drillable points
    def drillable(self, df_metric: Optional[Union[pd.DataFrame, Dict]]):
        if isinstance(df_metric, dict):
            df_metric = pd.DataFrame(df_metric)
        elif not isinstance(df_metric, pd.DataFrame):
            raise TypeError('incorrect type of df_metric')

        pd_drillable = pd.DataFrame({'X': self._drillable_points.T[0],
                                     'Y': self._drillable_points.T[1]})
        return df_metric.merge(pd_drillable, on=['X', 'Y'], how='inner')


"""
#Usage, manual testing, and visualization - see test_grid_builder for unit tests
from matplotlib import pyplot as plt

#Initialize using separate build_grid (the old way)
# pts = build_grid(None,243,930,216,901)#,2,2) #bigger steps
# pd_pts = pd.DataFrame({'X':pts.T[0],'Y':pts.T[1]})
# rasterbouwer = GridBuilder(pd_pts)
#Or initialize using class initializer with grid_params
# print("\ncall: rasterbouwer = GridBuilder(grid_params=[243,930,216,901])")
# rasterbouwer = GridBuilder(grid_params=[243,930,216,901])
# pts = rasterbouwer.grid
# pd_pts = rasterbouwer.df_grid

#Polygon as list of coordinates (last point links back to first)
# polygon = [[217, 901], [229, 901], [241, 913], #[241, 901] #corner with left-out initial well
#            [241, 928], [235, 928],
#            #[228, 921], [228, 910], [223, 910], [220, 915], [228, 915], #inner loop extruded
#            [228, 921], [225, 927], [217, 927]] #integer coordinates over edge points
polygon = [[217.5, 901.5], [229, 901.5], 
           [241.5, 914], [241.5, 928.5], [235, 928.5],
           #[228.1, 921.5], [228.1, 910.5], [223, 910.5], [225, 914.5], [228.1, 914.5], #inner loop extruded
           [228.1, 921.5], [225.5, 926.5], [217.5, 926.5]] #avoids all edge points
polygon_part = [[217.5, 910.5], [238, 910.5], #cut off here
           [241.5, 914], [241.5, 928.5], [235, 928.5],
           #[228.1, 921.5], [228.1, 910.5], [223, 910.5], [225, 914.5], [228.1, 914.5], #inner loop extruded
           [228.1, 921.5], [225.5, 926.5], [217.5, 926.5]] #cut off smaller version of above

# print("\ncall: rasterbouwer = GridBuilder(grid_params=[243,930,216,901])")
# rasterbouwer = GridBuilder(grid_params=[243,930,216,901])
print("\ncall: rasterbouwer = GridBuilder(grid_params=[243,930,216,901], polygon=polygon)")
rasterbouwer = GridBuilder(grid_params=[243,930,216,901], polygon=polygon)
#pts = rasterbouwer.grid
#pd_pts = rasterbouwer.df_grid

print("\ncall: rasterbouwer2 = GridBuilder(polygon=polygon_part, x_step=1, buffer=0.5)")
rasterbouwer2 = GridBuilder(polygon=polygon_part, x_step=1, buffer=0.5)
#pd_pts = rasterbouwer2.df_grid

#Redefining polygon
wells = pd.DataFrame({'X':[219,219,239,239],'Y':[904,925,904,925]})
pts_nw = rasterbouwer(wells)
pd_pts_nw = pd.DataFrame({'X':pts_nw.T[0],'Y':pts_nw.T[1]})
print("\ncall: rasterbouwer2(wells, polygon)")
pts_nwp = rasterbouwer2(wells, polygon)
# print("\ncall: rasterbouwer2(wells, polygon, expand=False)")
# pts_nwp = rasterbouwer2(wells, polygon, expand=False)
pd_pts_nwp = pd.DataFrame({'X':pts_nwp.T[0],'Y':pts_nwp.T[1]})
print("\ncall: rasterbouwer(wells, polygon_part)")
pts_pnw = rasterbouwer(wells, polygon_part)
# print("\ncall: rasterbouwer(wells, polygon_part, 0.5, restrict=True)")
# pts_pnw = rasterbouwer(wells, polygon_part, 0.5, restrict=True)
pd_pts_pnw = pd.DataFrame({'X':pts_pnw.T[0],'Y':pts_pnw.T[1]})

pd_pts = rasterbouwer.df_grid
pd_pts2 = rasterbouwer2.df_grid

#Some calculations to produce some metric as a pandas data frame,
#followed by restriction to polygon area for drilling
pd_pts_drill = rasterbouwer.drillable(pd_pts_pnw)
pd_pts_drill2 = rasterbouwer2.drillable(pd_pts_nwp)

polygonplot = np.array(polygon + [polygon[0]]).T
polygonplot_part = np.array(polygon_part + [polygon_part[0]]).T
plt.figure(figsize=(5,5), dpi=100)
plt.plot(polygonplot[0],polygonplot[1])
plt.plot(polygonplot_part[0],polygonplot_part[1])
#plt.plot(pts.T[0],pts.T[1],'+',color='xkcd:grey')
plt.plot(pd_pts['X'],pd_pts['Y'],'+',color='xkcd:grey')
plt.plot(pd_pts2['X'],pd_pts2['Y'],'x',color='xkcd:grey')
#plt.plot(pts_nw.T[0],pts_nw.T[1],'go') #full grid minus wells
plt.plot(wells['X'],wells['Y'],'ro') #wells
#plt.plot(pts_nwp.T[0],pts_nwp.T[1],'yo') #full grid minus wells
plt.plot(pd_pts_drill['X'],pd_pts_drill['Y'],'yo') #restricted grid
#plt.plot(pts_pnw.T[0],pts_pnw.T[1],'co') #also full grid minus wells
plt.plot(pd_pts_drill2['X'],pd_pts_drill2['Y'],'c.') #restricted grid
"""