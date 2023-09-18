import os
import sys
import yaml

from sanpy.base.project_classes import (Control_Project,
                                        Correlation_Project,
                                        Preprocessing_Project,
                                        Stacking_Project)

from sanpy.base.project_functions import save_project


def read_parfile(parfile):
    with open(parfile, 'r') as _file:
        par = yaml.load(_file)

    return par


if __name__ == '__main__':
    option = sys.argv[1]
    parfile = sys.argv[2]

    par = read_parfile(parfile)

    if option == 'preprocessing':
        P = Preprocessing_Project(par)
    elif option == 'correlation':
        P = Correlation_Project(par)
    elif option == 'stacking':
        P = Stacking_Project(par)
    elif option == 'control':
        P = Control_Project(par)

    P.setup()

    project_file = os.path.join(os.path.dirname(parfile),
                                f"{P.par['name']}_{option}.pkl")

    save_project(P, project_file)
