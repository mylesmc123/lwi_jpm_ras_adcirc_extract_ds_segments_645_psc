#!/usr/bin/env python3
from datetime import datetime, timedelta, timezone
import numpy as np


class Extract:
    def __init__(self, filename: str, pointfile: str, coldstart: datetime):
        import netCDF4 as nc

        self.__filename = filename
        self.__ncfile = nc.Dataset(self.__filename)
        self.__pointfile = pointfile
        self.__coldstart = coldstart
        self.__nnode = None
        self.__nelem = None
        self.__nodes = None
        self.__elements = None
        self.__centroid = None
        self.__tree = None
        self.__variable = None
        self.__extract_points = None
        self.__n_stations = None
        self.__point_id = None
        self.__point_indices = None
        self.__read_mesh()
        self.__find_variable()
        self.__get_dimensions()
        self.__read_points()
        self.__find_point_indices()

    def __is_inside(self, x, y, element_index) -> bool:
        import math

        nodes = self.__elements[element_index]
        n1 = nodes[0] - 1
        n2 = nodes[1] - 1
        n3 = nodes[2] - 1
        xx = np.array((self.__nodes[0][n1], self.__nodes[0][n2], self.__nodes[0][n3]))
        yy = np.array((self.__nodes[1][n1], self.__nodes[1][n2], self.__nodes[1][n3]))

        s0 = abs(
            (xx[1] * yy[2] - xx[2] * yy[1])
            - (x * yy[2] - xx[2] * y)
            + (x * yy[1] - xx[1] * y)
        )
        s1 = abs(
            (x * yy[2] - xx[2] * y)
            - (xx[0] * yy[2] - xx[2] * yy[0])
            + (xx[0] * y - x * yy[0])
        )
        s2 = abs(
            (xx[1] * y - x * yy[1])
            - (xx[0] * y - x * yy[0])
            + (xx[0] * yy[1] - xx[1] * yy[0])
        )
        tt = abs(
            (xx[1] * yy[2] - xx[2] * yy[1])
            - (xx[0] * yy[2] - xx[2] * yy[0])
            + (xx[0] * yy[1] - xx[1] * yy[0])
        )
        return s0 + s1 + s2 <= tt

    def __interpolation_weight(self, x, y, element_index):

        nodes = self.__elements[element_index]
        n1 = nodes[0] - 1
        n2 = nodes[1] - 1
        n3 = nodes[2] - 1
        xx = np.array((self.__nodes[0][n1], self.__nodes[0][n2], self.__nodes[0][n3]))
        yy = np.array((self.__nodes[1][n1], self.__nodes[1][n2], self.__nodes[1][n3]))

        denom = (yy[1] - yy[2]) * (xx[0] - xx[2]) + (xx[2] - xx[1]) * (yy[0] - yy[2])
        w0 = ((yy[1] - yy[2]) * (x - xx[2]) + (xx[2] - xx[1]) * (y - yy[2])) / denom
        w1 = ((yy[2] - yy[0]) * (x - xx[2]) + (xx[0] - xx[2]) * (y - yy[2])) / denom
        w2 = 1.0 - w1 - w0
        return [n1, n2, n3, w0, w1, w2]

    def __find_point_indices(self):
        self.__point_indices = []
        for p in self.__extract_points:
            _, idx = self.__tree.query([p[0], p[1]], k=10)
            found = False
            for d in idx:
                found = self.__is_inside(p[0], p[1], d)
                if found:
                    self.__point_indices.append(
                        self.__interpolation_weight(p[0], p[1], d)
                    )
                    break
            if not found:
                self.__point_indices.append([-9999, -9999, -9999, 0, 0, 0])

    def __read_points(self):
        import csv

        self.__extract_points = []
        with open(self.__pointfile, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "fd_id" in row.keys():
                    pid = int(row["fd_id"])
                else:
                    pid = int(row["id"])
                x = float(row["x"])
                y = float(row["y"])
                if "tag" in row.keys():
                    tag = int(row["tag"])
                else:
                    tag = 0
                self.__extract_points.append([x, y, tag, pid])
        self.__n_stations = len(self.__extract_points)

    def __find_variable(self):
        candidate_variables = [
            "zeta",
            "swan_HS",
            "swan_TPS",
            "swan_TM01",
            "swan_DIR",
            "zeta_max",
            "swan_HS_max",
            "swan_TPS_max",
            "swan_TM01_max",
            "swan_DIR_max",
        ]
        units = ["m", "m", "s", "s", "deg", "m", "m", "s", "s", "deg"]
        datum = [
            "navd88 2009.55",
            "m",
            "n/a",
            "n/a",
            "n/a",
            "navd88 2009.55",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
        ]
        standard_name = [
            "sea_surface_height_above_geoid",
            "sea_surface_wave_significant_height",
            "smoothed peak period",
            "mean wave direction",
            "maximum water surface elevationabove geoid",
            "maximum significant wave height",
            "maximum smoothed peak period",
            "maximum TM01 mean wave period",
            "maximum mean wave direction",
        ]
        long_name = [
            "water surface elevation above geoid",
            "significant wave height",
            "sea_surface_wave_period_at_variance_spectral_density_maximum",
            "sea_surface_wave_to_direction",
            "maximum_sea_surface_height_above_geoid",
            "maximum_sea_surface_wave_significant_height",
            "maximum_sea_surface_wave_period_at_variance_spectral_density_maximum",
            "maximum_sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
            "maximum_sea_surface_wave_to_direction",
        ]
        is_timeseries = [True, True, True, True, False, False, False, False, False]

        for i in range(len(candidate_variables)):
            if (
                candidate_variables[i] + "_transpose" in self.__ncfile.variables
                or candidate_variables[i] in self.__ncfile.variables
            ):
                if candidate_variables[i] + "_transpose" in self.__ncfile.variables:
                    self.__variable = candidate_variables[i] + "_transpose"
                else:
                    self.__variable = candidate_variables[i]
                self.__units = units[i]
                self.__datum = datum[i]
                self.__standard_name = standard_name[i]
                self.__long_name = long_name[i]
                self.__is_timeseries = is_timeseries[i]
                return
        raise RuntimeError("Could not locate a valid ADCIRC variable")

    def __get_dimensions(self):
        self.__n_step = self.__ncfile.dimensions["time"].size

    def __read_mesh(self):
        from scipy.spatial import KDTree

        x = self.__ncfile["x"][:]
        y = self.__ncfile["y"][:]
        self.__nodes = np.ascontiguousarray([x, y])
        self.__elements = np.ascontiguousarray(self.__ncfile["element"][:])
        self.__nelem = self.__elements.shape[0]
        self.__nnode = x.shape[0]
        self.__centroid = np.ascontiguousarray(
            np.zeros((self.__elements.shape[0], 2), dtype=float)
        )

        for i in range(self.__nelem):
            nn = self.__elements[i]
            n1 = nn[0] - 1
            n2 = nn[1] - 1
            n3 = nn[2] - 1
            x_c = (
                self.__nodes[0][n1] + self.__nodes[0][n2] + self.__nodes[0][n3]
            ) / 3.0
            y_c = (
                self.__nodes[1][n1] + self.__nodes[1][n2] + self.__nodes[1][n3]
            ) / 3.0
            self.__centroid[i][0] = x_c
            self.__centroid[i][1] = y_c

        self.__tree = KDTree(self.__centroid)

    @staticmethod
    def __normalize_weights(z0, z1, z2, w0, w1, w2):
        if z0 > -999 and z1 > -999 and z2 > -999:
            return w0, w1, w2
        elif z0 < -999 and z1 < -999 and z2 < -999:
            return 0.0, 0.0, 0.0
        elif z0 < -999 and z1 > -999 and z2 > -999:
            f = 1.0 / (w1 + w2)
            w0 = 0.0
            w1 *= f
            w2 *= f
        elif z0 > -999 and z1 < -999 and z2 > -999:
            f = 1.0 / (w0 + w2)
            w1 = 0.0
            w0 *= f
            w2 *= f
        elif z0 > -999 and z1 > -999 and z2 < -999:
            f = 1.0 / (w0 + w1)
            w0 *= f
            w1 *= f
            w2 = 0.0
        elif z0 > -999 and z1 < -999 and z2 < -999:
            w0 = 1.0
            w1 = 0.0
            w2 = 0.0
        elif z0 < -999 and z1 > -999 and z2 < -999:
            w0 = 0.0
            w1 = 1.0
            w2 = 0.0
        elif z0 < -999 and z1 < -999 and z2 > -999:
            w0 = 0.0
            w1 = 0.0
            w2 = 1.0
        return w0, w1, w2

    def __read_transpose_variable(self, variable_name) -> dict:
        record = {}
        for pt in self.__point_indices:
            for pt2 in pt:
                if pt2 not in record.keys():
                    if pt2 > 0:
                        record[pt2] = np.ascontiguousarray(
                            self.__ncfile[self.__variable][:, pt2]
                        )
                    else:
                        record[pt2] = None
        return record

    def __extract_from_transpose_variable(self) -> tuple:
        import netCDF4 as nc

        records = self.__read_transpose_variable(self.__variable)

        data = np.zeros((self.__n_stations, self.__n_step), dtype=float)

        all_times = self.__ncfile["time"][:]
        time = np.zeros((self.__n_step), dtype=int)
        for i in range(len(all_times)):
            time[i] = datetime.timestamp(
                self.__coldstart + timedelta(seconds=all_times[i])
            )
            
        for j in range(self.__n_stations):
            pt0 = self.__point_indices[j][0]
            pt1 = self.__point_indices[j][1]
            pt2 = self.__point_indices[j][2]
            w00 = self.__point_indices[j][3]
            w01 = self.__point_indices[j][4]
            w02 = self.__point_indices[j][5]

            z0 = records[pt0]
            z1 = records[pt1]
            z2 = records[pt2]

            for i in range(len(all_times)):
                if z0 is None:
                    z00 = -99999.0
                else:
                    z00 = z0[i]

                if z1 is None:
                    z01 = -99999.0
                else:
                    z01 = z1[i]

                if z2 is None:
                    z02 = -99999.0
                else:
                    z02 = z2[i]

                w0, w1, w2 = Extract.__normalize_weights(z00, z01, z02, w00, w01, w02)
                if w0 == 0.0 and w1 == 0.0 and w2 == 0.0:
                    data[j][i] = -99999.0
                else:
                    data[j][i] = z00 * w0 + z01 * w1 + z02 * w2
        return time, data

    def __extract_from_standard_variable(self) -> tuple:
        import netCDF4 as nc

        time = np.zeros((self.__n_step), dtype=int)
        data = np.zeros((self.__n_stations, self.__n_step), dtype=float)
        all_times = self.__ncfile["time"][:]

        for i in range(self.__n_step):
            if self.__is_timeseries:
                record = np.ascontiguousarray(self.__ncfile[self.__variable][i])
            else:
                record = np.ascontiguousarray(self.__ncfile[self.__variable])
            sim_time = self.__coldstart + timedelta(seconds=all_times[i])
            time[i] = datetime.timestamp(sim_time)
            for j in range(self.__n_stations):
                z0 = record[self.__point_indices[j][0]]
                z1 = record[self.__point_indices[j][1]]
                z2 = record[self.__point_indices[j][2]]
                w0, w1, w2 = Extract.__normalize_weights(
                    z0,
                    z1,
                    z2,
                    self.__point_indices[j][3],
                    self.__point_indices[j][4],
                    self.__point_indices[j][5],
                )
                if w0 == 0.0 and w1 == 0.0 and w2 == 0.0:
                    data[j][i] = -99999.0
                else:
                    data[j][i] = z0 * w0 + z1 * w1 + z2 * w2
        return time, data

    def __write_output_netcdf(
        self, output_file: str, time: np.ndarray, data: np.ndarray
    ) -> None:
        import netCDF4 as nc

        ds = nc.Dataset(output_file, "w", format="NETCDF4")
        time_dim = ds.createDimension("time", self.__n_step)
        station_dim = ds.createDimension("nstation", self.__n_stations)

        timevar = ds.createVariable("time", "i8", ("time"))
        timevar.units = "seconds since 1970-01-01 00:00:00"

        lat = ds.createVariable("latitude", "f8", ("nstation"), zlib=True, complevel=2)
        lat.reference = "EPSG:4326"
        lon = ds.createVariable("longitude", "f8", ("nstation"), zlib=True, complevel=2)
        lon.reference = "EPSG:4326"
        pid = ds.createVariable("point_id", "i4", ("nstation"), zlib=True, complevel=2)
        tag = ds.createVariable(
            "point_type", "i4", ("nstation"), zlib=True, complevel=2
        )
        tag.types = "0=gate, 1=levee, 2=roadway"

        datavar = ds.createVariable(
            "values",
            "f8",
            ("nstation", "time"),
            fill_value=-99999,
            zlib=True,
            complevel=2,
            chunksizes=(1, self.__n_step),
        )

        datavar.adcirc_type = self.__variable
        datavar.standard_name = self.__standard_name
        datavar.long_name = self.__long_name
        datavar.units = self.__units
        datavar.datum = self.__datum

        for i in range(self.__n_stations):
            lon[i] = self.__extract_points[i][0]
            lat[i] = self.__extract_points[i][1]
            tag[i] = self.__extract_points[i][2]
            pid[i] = self.__extract_points[i][3]

        timevar[:] = time
        datavar[:, :] = data
        ds.close()

    def __write_output_csv(
        self, output_file: str, time: np.ndarray, data: np.ndarray
    ) -> None:
        import csv

        with open(output_file, "w") as out:
            writer = csv.writer(out)
            writer.writerow(["PID", "Longitude", "Latitude", "Value"])
            for i in range(self.__n_stations):
                writer.writerow(
                    [
                        self.__extract_points[i][3],
                        self.__extract_points[i][0],
                        self.__extract_points[i][1],
                        # TODO the csv writer is currently only writing a single value based on this data[i][0] slicing. 
                        # Feature or Bug?
                        data[i][0],
                    ]
                )

    def __write_output_dss(
        self, output_file: str, time: np.ndarray, data: np.ndarray
    ) -> None:
        from pydsstools.heclib.dss import HecDss
        from pydsstools.core import TimeSeriesContainer
        import os.path
        import numpy as np
        import datetime
        

        # Write Average Timeseries to DSS.
        dss_file = f'{output_file.split(".")[0]}.dss'
        head, tail = os.path.split(output_file)
        # aPart = f"RAS_Model:{RFC_Gages_dict[gage][1]}"
        bPart = f"{tail.split('.')[0]}"
        pathname = f"//{bPart}/STAGE//IR-MONTH/ADCIRC/"
        tsc = TimeSeriesContainer()
        tsc.pathname = pathname
        # tsc.units ="m"
        tsc.units ="FEET"
        tsc.type = "INST"
        tsc.interval = -1

        # Average Each Time Ordinate Value
        avg_values_List = []
        data[data==-99999] = np.nan
        
        # Transpose data to get each timestep as an array through the points. WantedStructure (time, each point value)
        # Current structure has each point as an array through time. existingStructure (point, value for each timestep)
        dataT = data.T
        for step_meter in dataT:
            # Convert each value from meter to feet.
            step_feet = step_meter * 3.28084
            avg_values_List.append(np.nanmean(step_feet))
            # avg_values_List.append(np.nanmean(step_meter))
        
        # Convert Times to Format required by DSS.
        times_List = []
        for step in time:
            # print (datetime.datetime.fromtimestamp(step))
            times_List.append(datetime.datetime.fromtimestamp(step).strftime("%d%b%Y %H:%M:%S"))

        tsc.values = avg_values_List
        tsc.numberValues = len(tsc.values)
        tsc.times = times_List
        # tsc.startDateTime = times_List[0]
        
        with HecDss.Open(dss_file) as fid:
            status = fid.put_ts(tsc)
    
    def __write_output_hdf(
        self, output_file : str, time : np.ndarray, data : np.ndarray, hdf_fn : str, ras_startTime : str, ras_endTime: str
    ) -> None:
        import os.path
        import numpy as np
        import pandas as pd
        from datetime import datetime
        import h5py

        print (f'Updating the downstream boundary condtion timeseries for: {hdf_fn}.')

        # pickle data and time for testing.
        # import pickle
        # with open('/twi/work/projects/p00667_louisiana_rtf/ras/lffs/louisiana_real_time_forecasting/src/system/scripts/dev/temp/extract_pointfile_data.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('/twi/work/projects/p00667_louisiana_rtf/ras/lffs/louisiana_real_time_forecasting/src/system/scripts/dev/temp/extract_pointfile_time.pickle', 'wb') as handle:
        #     pickle.dump(time, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        with h5py.File(hdf_fn,  "a") as hf:
        # Ensure Results dataset is removed from HDF file. RAS unsteady solver will not run if Results dataset present.
            try:
                del hf['Results']
            except:
                print ('HDF file has no results dataset, and is ready to be used as a RAS unsteady run input file.')
        
            # Testing setup (using only the Amite River flow hydrograph).
            # gage = list(RFC_Gages_dict.keys())[1:][-1]
            # parametersList = ['Flow']

            # Use file name to link HDF BC paths to RFC data.
            head, tail = os.path.split(self.__pointfile)
            pt_fn = tail.split(".")[0]
            ras_model = pt_fn.split("_")[0]
            twoD_area = pt_fn.split("_")[1]
            bc_line = pt_fn.split("_")[2]
            print (bc_line)
            hdf_bc_path = f'/Event Conditions/Unsteady/Boundary Conditions/Stage Hydrographs/2D: {twoD_area} BCLine: {bc_line}'
            
            # Current HDF Dataset
            hdf_bc_ds = hf[hdf_bc_path]
            
            # Average Each Time Ordinate Value
            avg_values_List = []
            data[data==-99999] = np.nan
                    
            # Transpose data to get each timestep as an array through the points. WantedStructure (time, each point value)
            # Current structure has each point as an array through time. existingStructure (point, value for each timestep)
            dataT = data.T
            for step_meter in dataT:
                # Convert each value from meter to feet.
                step_feet = step_meter * 3.28084
                avg_values_List.append(np.nanmean(step_feet))
            
            # Convert Times to datetime format.
            times_list_dt = []
            for step in time:
                times_list_dt.append(datetime.fromtimestamp(step))
            
            # Set up dataframe for extracted adcirc data to update HDF values with.           
            times_values_dict = {
                'datetime'  : times_list_dt,
                'values'    : avg_values_List
                }
            df = pd.DataFrame(times_values_dict)

            # Cut data to RAS run simulation time window.          
            ras_run_startTime_dt = datetime.strptime(ras_startTime, '%d%b%Y %H%M')
            ras_run_endTime_dt = datetime.strptime(ras_endTime, '%d%b%Y %H%M')
            df_trim = df.loc[df['datetime'].between(str(ras_run_startTime_dt),str(ras_run_endTime_dt))]
            df_trim = df_trim.reset_index(drop=True)

            # Convert datetimes to integers in days from startTime referenced as 0.        
            referenceTime = df_trim.loc[0]['datetime']
            df_trim['deltaTime'] = df_trim['datetime'] - referenceTime
            df_trim['seconds'] = df_trim['deltaTime'].map(lambda v: v.total_seconds())

            # Convert seconds to days. 86,400 seconds per day.
            df_trim['days'] = df_trim['seconds'] / 86400

            # Create numpy array of shape needed for HDF
            df_trimmer = df_trim.drop(['datetime','deltaTime','seconds'], axis=1)
            df_to_hdf = df_trimmer[['days', 'values']]
            np_to_hdf = df_to_hdf.to_numpy()

            # Create a new temp dataset based on numpy array.
            hdf_temp_path = '/Event Conditions/Unsteady/Boundary Conditions/Stage Hydrographs/2D: Foo BCLine: Bar'
            
            try:
                del hf[hdf_temp_path]
                print ('deleted expired temp dataset, ready to create new dataset.')
            
            except: 
                print ('temp dataset does not exist, ready to create new temp dataset.')
            
            hdf_temp_ds = hf.create_dataset(hdf_temp_path, data=np_to_hdf)

            # Pull attrs from expired dataset. update time info.
            for name, value in hdf_bc_ds.attrs.items():
                
                if name == 'Start Date':
                    value = ras_startTime.encode('utf-8')
                
                elif name == 'End Date':
                    value = ras_endTime.encode('utf-8')
                    
                hdf_temp_ds.attrs.modify(name, value)
                print (name,': ', value)

            #  Delete expired production dataset
            del hf[hdf_bc_path]

            # Rename the temp dataset to production dataset
            hf[hdf_bc_path] = hf[hdf_temp_path]

            # Delete temp dataset.
            del hf[hdf_temp_path]
        

    def extract(self, output_file, output_format, hdf_fn=None, ras_startTime=None, ras_endTime=None) -> None:

        if "transpose" in self.__variable:
            time, data = self.__extract_from_transpose_variable()
        else:
            time, data = self.__extract_from_standard_variable()

        if output_format == "netcdf":
            self.__write_output_netcdf(output_file, time, data)
        elif output_format == "csv":
            self.__write_output_csv(output_file, time, data)
        elif output_format == "dss":
            self.__write_output_dss(output_file, time, data)
        elif output_format == "ras":
            self.__write_output_hdf(output_file, time, data, hdf_fn, ras_startTime, ras_endTime)
