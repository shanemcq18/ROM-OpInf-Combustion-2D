# boris_QB_matrices.py

import os
import h5py
import matplotlib.pyplot as plt

import config
import utils


H5FILE = "boris_matrices.h5"
t = utils.load_time_domain()



def append_rom_info(k, r, l):
    rom = utils.load_rom(k, r, l)
    x0_ = utils.load_projected_data(k, r)[0][:,0]

    # Plot the integrated ROM
    X_ = rom.predict(x0_, t, config.U)
    for i in range(X_.shape[0]):
        plt.plot(t, X_[i,:], lw=1)
    plt.title(fr"$k = {k:d}$, $r = {r:d}$")
    utils.save_figure(fr"rom_k{k:d}_r{r:d}.pdf")

    prefix = f"k{k:d}_r{r:d}"
    with h5py.File(H5FILE, 'a') as hf:
        if "time" not in hf:
            hf.create_dataset("time", data=t)
        hf.create_dataset(f"{prefix}/x0", data=x0_)
        hf.create_dataset(f"{prefix}/cr", data=rom.c_)
        hf.create_dataset(f"{prefix}/Ar", data=rom.A_)
        hf.create_dataset(f"{prefix}/Hr", data=rom.H_)
        hf.create_dataset(f"{prefix}/Br", data=rom.B_)


def print_attrs(name, obj):
    shift = name.count('/') * '    '
    print(shift + name)
    for key, val in obj.attrs.items():
        print(shift + '    ' + f"{key}: {val}")


def main():
    utils.reset_logger()
    if os.path.isfile(H5FILE):
        os.remove(H5FILE)
    append_rom_info(5000, 11, 19744)
    append_rom_info(10000, 22, 34610)
    append_rom_info(10000, 28, 85363)
    with h5py.File(H5FILE, 'r') as hf:
        hf.visititems(print_attrs)


if __name__ == "__main__":
    main()
