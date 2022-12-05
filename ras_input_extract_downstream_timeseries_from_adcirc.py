#!/usr/bin/env python3

def getColdStart(adcirc_fort63_file):
# Get cold start from adcirc fort.63 file:
    import xarray as xr
    from datetime import datetime
    ds = xr.open_dataset(adcirc_fort63_file, drop_variables=['neta','nvel'], chunks={"node": 1000})
    coldstart = ds.time.base_date
    coldstart = datetime.fromisoformat(coldstart)
    return coldstart

def getPointFilesFromDir(dir):
    import glob, os
    pointFilesList = glob.glob(rf'{dir}/*.txt')
    return pointFilesList

def main():
    import argparse
    from datetime import datetime, timedelta, timezone
    from util.extract import Extract
    import os.path

    p = argparse.ArgumentParser(description="File to use for extraction of ADCIRC time series")
    p.add_argument(
        "--adcirc_wse_file", help="The ADCIRC file to extract timeseries data from.", 
        required=True, 
        type=str
    )

    p.add_argument(
        "--ras_hdf", help="The filename for the HEC-RAS temp plan HDF file (p##.tmp.hdf) to update boundary condition timeseries data to.", 
        required=False, 
        type=str
    )
    
    p.add_argument(
        "--point_file_dir",
        help="Name of the directory containing the point files to extract coordinates from. The directory should only contain .txt files that are point files.",
        required=True,
        type=str,
    )
    p.add_argument(
        "--coldstart",
        help="Cold start time for ADCIRC simulation",
        required=False,
        type=datetime.fromisoformat,
    )

    p.add_argument(
        "--ras_start", help="RAS simulation start time. Will be used to update RAS pertinent data and HDF files. (Format Ex: 19Sep2022 0600)", 
        required=False, type=str
    )

    p.add_argument(
        "--ras_end", help="RAS simulation start time. Will be used to update RAS pertinent data and HDF files. (Format Ex: 19Sep2022 0600)", 
        required=False, type=str
    )
    
    p.add_argument(
        "--outdir", help="Name of output directory to place output files", required=True, type=str
    )
    p.add_argument(
        "--format",
        help="Format to use. Either netcdf, csv, dss, hdf",
        required=False,
        default="netcdf",
        type=str,
    )
    p.add_argument(
        "--sim_number",
        help="Sim number used for copying output hdf to a unique directory",
        required=False,
        type=str,
    )

    args = p.parse_args()

    # ...The user is sending UTC, so make python do the same
    if args.coldstart is None:
        coldstart = getColdStart(args.adcirc_wse_file)
    else: coldstart = args.coldstart
        
    coldstart_utc = datetime(
        coldstart.year,
        coldstart.month,
        coldstart.day,
        coldstart.hour,
        coldstart.minute,
        0,
        tzinfo=timezone.utc,
    )

    pointFilesList = getPointFilesFromDir(args.point_file_dir)
    for pointFile in pointFilesList:
        head, tail = os.path.split(pointFile)
        outputFile = os.path.join(args.outdir, tail.split(".")[0] + ".nc")
        print (f'\nExtracting {tail.split(".")[0]}')

        # if hdf output validate and extract using additional required arguments.
        if args.format == 'hdf':

            if (args.ras_start is None) or (args.ras_end is None) or (args.ras_hdf is None) or (args.sim_number is None):
                p.error("--output = hdf --> Requires the arguments: --ras_hdf, --sim_number, --ras_start, and --ras_end.")

            try:
                datetime.strptime(args.ras_start, '%d%b%Y %H%M')  
            except:
                print (f'\nERROR: --ras_start {args.ras_start} not the correct format required (Format Ex: 19Sep2022 0600).\n')
                exit()

            try:
                datetime.strptime(args.ras_end, '%d%b%Y %H%M')
            except:
                print (f'\nERROR: --ras_end {args.ras_end} not the correct format required (Format Ex: 19Sep2022 0600).\n')
                exit()
            


            # extract using additional required arguments for RAS HDF output.
            extractor = Extract(args.adcirc_wse_file, pointFile, coldstart_utc)
            extractor.extract(outputFile, args.format, args.ras_hdf, args.ras_start, args.ras_end)
            

        # else if not hdf output, extract using less arguments.
        else:
            extractor = Extract(args.adcirc_wse_file, pointFile, coldstart_utc)
            extractor.extract(outputFile, args.format)
    
    # copy output HDF file to unique directory based on sim_number
    # Currently only implemented for HDF.
    import shutil
    if args.format == 'hdf':
        head, tail = os.path.split(args.ras_hdf)
        dest = os.path.join(args.outdir,args.sim_number,tail)
        # dest_dir = 
        try:
            shutil.copyfile(args.ras_hdf, dest)

        except IOError as io_err:
            os.makedirs(os.path.join(args.outdir,args.sim_number),exist_ok=True)
            shutil.copyfile(args.ras_hdf, dest)
    

if __name__ == "__main__":
    main()
