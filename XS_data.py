import h5py
import numpy as np
import scipy.io

mat = scipy.io.loadmat('XS_data.mat')

print(mat['ABS_ref'])
print(type(mat['ABS_ref']))
print(mat['ABS_ref'].shape)

with h5py.File("XS_data_old.hdf5", "w") as f:
    for key, value in mat.items():
        # eliminate entries starting with __ which are created by the mat importer
        if '__' not in key:
            if key == 'DABS_Na':
                key = 'ABS_der_Na'
            if key == 'DCHI_Na':
                key = 'CHI_der_Na'
            if key == 'DFIS_Na':
                key = 'FIS_der_Na'
            if key == 'DKAPPA_Na':
                key = 'KAPPA_der_Na'
            if key == 'DNU_Na':
                key = 'NU_der_Na'
            if key == 'DSCAT_Na':
                key = 'SCAT_der_Na'
            if key == 'DTR_Na':
                key = 'TR_der_Na'

            if key == 'DABS_fuel':
                key = 'ABS_der_fuel'
            if key == 'DCHI_fuel':
                key = 'CHI_der_fuel'
            if key == 'DFIS_fuel':
                key = 'FIS_der_fuel'
            if key == 'DKAPPA_fuel':
                key = 'KAPPA_der_fuel'
            if key == 'DNU_fuel':
                key = 'NU_der_fuel'
            if key == 'DSCAT_fuel':
                key = 'SCAT_der_fuel'
            if key == 'DTR_fuel':
                key = 'TR_der_fuel'

            # if 'SCAT' in key:
            #     f.create_dataset(key, data=np.transpose(value))
            # else:
            #     f.create_dataset(key, data=value)

            f.create_dataset(key, data=value)

    # f.create_dataset('Na', data=[[600, 800]])
    f.create_dataset('Na', data=[600])
    f.create_dataset('fuel', data=[600])
    f['ABS_ref'].attrs['temperature'] = 600.0


with h5py.File("XS_data.hdf5", "w") as f:
    ref = f.create_group('ref')
    der = f.create_group('der')
    der_Na = der.create_group('Na')
    der_fuel = der.create_group('fuel')

    der_Na.attrs['temperature'] = 600.0
    der_fuel.attrs['temperature'] = 600.0

    for key, value in mat.items():
        if '_ref' in key:
            key = key.split('_')[0]
            if 'SCAT' in key:
                ref.create_dataset(key, data=np.transpose(value))
            else:
                ref.create_dataset(key, data=value)

        # eliminate entries starting with __ which are created by the mat importer
        if '__' not in key:
            if '_Na' in key:
                if key == 'DABS_Na':
                    key = 'ABS'
                if key == 'DCHI_Na':
                    key = 'CHI'
                if key == 'DFIS_Na':
                    key = 'FIS'
                if key == 'DKAPPA_Na':
                    key = 'KAPPA'
                if key == 'DNU_Na':
                    key = 'NU'
                if key == 'DSCAT_Na':
                    key = 'SCAT'
                if key == 'DTR_Na':
                    key = 'TR'

                if 'SCAT' in key:
                    der_Na.create_dataset(key, data=np.transpose(value))
                else:
                    der_Na.create_dataset(key, data=value)

            if '_fuel' in key:
                if key == 'DABS_fuel':
                    key = 'ABS'
                if key == 'DCHI_fuel':
                    key = 'CHI'
                if key == 'DFIS_fuel':
                    key = 'FIS'
                if key == 'DKAPPA_fuel':
                    key = 'KAPPA'
                if key == 'DNU_fuel':
                    key = 'NU'
                if key == 'DSCAT_fuel':
                    key = 'SCAT'
                if key == 'DTR_fuel':
                    key = 'TR'

                if 'SCAT' in key:
                    der_fuel.create_dataset(key, data=np.transpose(value))
                else:
                    der_fuel.create_dataset(key, data=value)

    f.attrs['eg'] = f['der/Na/ABS'].shape[1]
    # f.create_dataset('Na', data=[[600, 800]])
    # f.create_dataset('Na', data=[600])
    # f.create_dataset('fuel', data=[600])
    # f['ABS_ref'].attrs['temperature'] = 600.0
    print(f['der/Na/ABS'])
    print(f['der/Na'].attrs['temperature'])
