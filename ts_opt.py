import os
import time
import torch
import warnings
import numpy as np

from xtb.ase.calculator import XTB
from mace.calculators import MACECalculator
from ase.utils.forcecurve import fit_images, plotfromfile
from ase.io.trajectory import TrajectoryReader
from ase.calculators.mixing import SumCalculator
from sella import Sella
from sella import IRC
from ase.calculators.gaussian import Gaussian
from x3dase.x3d import X3D
from ase.io import read
from ase.vibrations import Vibrations
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
import matplotlib.pyplot as plt
from ase.io import read, write, animation
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.mep.autoneb import AutoNEB
from ase.optimize.bfgs import BFGS
from natsort import ns, natsorted

from neuralneb.painn.painn import PaiNN
from neuralneb import utils
import configparser
from argparse import ArgumentParser
import traceback
import jax

jax.config.update('jax_platform_name', 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_mep(images, save_dir, rxn_name, job_type, interval=50):
    """
    save the mep png and search pathway gif
    """
    fit = fit_images(images)
    fit.plot()
    plt.savefig(os.path.join(save_dir, f'{rxn_name}_{job_type}.png'), dpi=300)
    animation.write_animation(os.path.join(save_dir, f'{rxn_name}_{job_type}.gif'), images=images, writer='pillow', interval=interval) # save the gif of dimer path
    plt.close()

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def delete_files_with_extension(folder_path, extension):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension) and '0' in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

def sella_refine_irc(rxn_names, input_path, output_path, calculator, redine_f_max, irc_f_max, steps):

    title = 'name\trefine_time\trefine_steps\tirc_time\tirc_steps\tEf\tEr\tflag\n'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for fs in rxn_names:
        rxn_name = fs.split('.')[0]
        # work folder
        rxn_output_path = os.path.join(output_path, rxn_name)
        if not os.path.exists(rxn_output_path):
            os.makedirs(rxn_output_path)
        else:
            print(f'{rxn_name} already exists, skip.')
            continue

        ref_file_dir = os.path.join(rxn_output_path, f'{rxn_name}_ref.traj')
        ref_ts_dir = os.path.join(rxn_output_path, f'{rxn_name}_ts.xyz')
        irc_file_dir = os.path.join(rxn_output_path, f'{rxn_name}_irc.traj')

        max_retries = 4
        for attempt in range(1, max_retries+1):
            print(f'Refining loop {attempt} for rxn_{rxn_name}...')
            try:
                if os.path.exists(ref_file_dir):
                    os.remove(ref_file_dir)
                if os.path.exists(ref_ts_dir):
                    os.remove(ref_ts_dir)

                atoms = read(os.path.join(input_path, str(fs)))
                atoms.calc = calculator

                time_start = time.time()
                dyn = Sella(atoms, internal=True, trajectory=ref_file_dir, eta=1e-4, gamma=0.4)
                dyn.run(fmax=redine_f_max, steps=steps)
                time_end = time.time()
                sella_ref_time = time_end - time_start
                print(f'{rxn_name} sella_ref_time: {round(sella_ref_time, 2)} s')
                traj = TrajectoryReader(ref_file_dir)
                write(ref_ts_dir, traj[-1])

            except Exception as e:
                traceback.print_exc()
                print(f'refine error on {rxn_name}')
                if attempt <= max_retries:
                    print(f'Attempt {attempt} failed on {rxn_name} REFINE. Retrying...')
                    time.sleep(3)
                    continue
                else:
                    print(f'REFINE error on {rxn_name} after {max_retries} attempts.')
                    continue

            print(f'Calculating IRC for {rxn_name}...')
            try:
                if os.path.exists(irc_file_dir):
                    os.remove(irc_file_dir)

                irc_inits = traj[-1]
                irc_inits.calc = calculator
                opt = IRC(irc_inits, trajectory=irc_file_dir, dx=0.05, eta=1e-4, gamma=0.4)

                opt.run(fmax=irc_f_max, steps=steps, direction='forward')
                flag = len(TrajectoryReader(irc_file_dir))
                opt.run(fmax=irc_f_max, steps=steps, direction='reverse')

                irc_traj = TrajectoryReader(irc_file_dir)

                arrange_irc = []
                for i in range(flag - 1, -1, -1):
                    arrange_irc.append(irc_traj[i])
                arrange_irc.extend(irc_traj[flag:])

                try:
                    visualize_mep(arrange_irc, rxn_output_path, rxn_name, job_type='irc')
                except:
                    warnings.warn("visualize sella irc mep failed.", UserWarning)

                nframes = len(irc_traj)
                optimized_sella = read(irc_file_dir, ":")[-1 * nframes:]

                x3d = X3D(optimized_sella, bond=True)
                x3d.write(os.path.join(rxn_output_path, f"optimized_sella_irc_{rxn_name}.html"))

                break

            except Exception as e:
                traceback.print_exc()
                if attempt <= max_retries:
                    print(f'Attempt {attempt} failed on {rxn_name} IRC. Retrying...')
                    time.sleep(3)
                    continue
                else:
                    print(f'IRC error on {rxn_name} after {max_retries + 1} attempts.')
                    continue

def main(args, DEVICE):

    input_path = args.input_path
    output_path = args.output_path + f'/{args.model_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the model and initialize the calculator
    try:
        if args.model_name == 'painn_nms':
            config = read_config(args.model_path + f'/{args.model_name}/config.txt')
            statedict = torch.load(args.model_path + f'/{args.model_name}/painn.sd', map_location=DEVICE)
            new_statedict = {}
            for k, v in statedict.items():
                name = k[7:] if k.startswith('module.') else k
                new_statedict[name] = v
            model = PaiNN(int(config['DEFAULT']['num_interactions']), int(config['DEFAULT']['hidden_state_size']),
                          float(config['DEFAULT']['cutoff']))
            model.load_state_dict(new_statedict)
            model.eval()
            calculator = utils.MLCalculator(model)

        elif args.model_name == 'mace_nms':
            calculator = MACECalculator(model_paths=args.model_path + f'/{args.model_name}/mace_nms.model', device='cuda')

        elif args.model_name == 'mace_nonms':
            calculator = MACECalculator(model_paths=args.model_path + f'/{args.model_name}/mace_nonms.model', device='cuda')

        else:
            calculator = None
            raise ValueError(
                f"Unknown model name: {args.model_name}. Please choose from 'mace_nms', 'mace_nonms', 'painn_nms'.")

    except Exception as e:
        print('An error was encountered while loading the model')

    rxn_names = [n for n in os.listdir(input_path) if n.endswith('.xyz')]
    rxn_names = natsorted(rxn_names, alg=ns.PATH)
    print(f'number of jobs: {len(rxn_names)}.')

    if calculator is not None:
        sella_refine_irc(rxn_names, input_path, output_path, calculator, redine_f_max=0.04, irc_f_max=0.1, steps=300)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--input_path", nargs="?", default=r'./example/input')
    parser.add_argument("--output_path", nargs="?", default=r'./example/output')
    parser.add_argument("--model_name", nargs="?", default='painn_nms',
                        choices=['mace_nms', 'mace_nonms', 'painn_nms'],
                        help="Select the model name for transition state optimization. "
                             "Options include: 'mace_nms', 'mace_nonms', 'painn_nms'")
    parser.add_argument("--model_path", nargs="?", default=r'./models')
    args = parser.parse_args()

    main(args, DEVICE)
