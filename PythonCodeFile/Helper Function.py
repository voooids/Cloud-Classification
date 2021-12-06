def read_nc(nc_file):
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')
    f_radiances = np.vstack([file.variables[name][:] for name in radiances])
    f_properties = np.vstack([file.variables[name][:] for name in properties])
    f_rois = file.variables[rois][:]
    f_labels = file.variables[labels][:]
    f_lats = file.variables[coordinates[0]][:]
    f_longs = file.variables[coordinates[1]][:]
    
    file.close()
    return f_radiances, f_properties, f_rois, f_labels, f_lats, f_longs
