import os
from sanpy.base.project_functions import load_project


project_path = ''
output_path = ''

# DONT EDIT BELOW THIS LINE
# =========================
project = load_project(project_path)

if not os.path.isdir(output_path):
    os.mkdir(output_path)

# Greens function FORCESOLUTION file
for i, station in enumerate(project.stations.index):
    net = project.stations['net'][station]
    sta = project.stations['sta'][station]
    lat = project.stations['lat'][station]
    lon = project.stations['lon'][station]
    depth = project.stations['elv'][station] / -1000.0

    force_file_name = '{}_FORCESOLUTION_{}.{}'.format(i, net, sta)

    with open(os.path.join(output_path, force_file_name), 'w') as _file:
        _file.write('FORCE {:06d}\n'.format(i))
        _file.write('time shift: 0.0\n')
        _file.write('f0: 0.0\n')
        _file.write('latorUTM: {}\n'.format(lat))
        _file.write('lonorUTM: {}\n'.format(lon))
        _file.write('depth: {}\n'.format(depth))
        _file.write('factor force source: 1.0\n')
        _file.write('component dir vect source E: 0.0\n')
        _file.write('component dir vect source N: 0.0\n')
        _file.write('component dir vect source Z_UP: -1.0\n')

# STATIONS file
with open(os.path.join(output_path, 'STATIONS'), 'w') as _file:
    for station in project.stations.index:
        net = project.stations['net'][station]
        sta = project.stations['sta'][station]
        lat = project.stations['lat'][station]
        lon = project.stations['lon'][station]
        depth = project.stations['elv'][station] * -1.0 

        _file.write('{} {} {:.4f} {:.4f} 0.0000 {:.4f}\n'.format(
            sta, net, lat, lon, depth))
