#!/usr/bin/env python3

import argparse
import sys

import ase
import ase.io
import equistore
import numpy as np
import torch
from rascaline import AtomicComposition, LodeSphericalExpansion, SphericalExpansion
from rascaline.utils import PowerSpectrum
from sklearn.linear_model import RidgeCV
from skmatter.preprocessing import StandardFlexibleScaler

from radial_basis import KspaceRadialBasis

torch.set_default_dtype(torch.float64)

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the arguments
parser.add_argument("-p", "--prefix", type=str, default="test", help="The prefix")
parser.add_argument(
    "-r", "--rcut", type=float, default=4.0, help="The training rcut value"
)
parser.add_argument("-s", "--stride", type=int, default=1, help="The stride")
parser.add_argument("-n", "--epochs", type=int, default=500, help="number of epochs")
parser.add_argument("-l", "--layer", type=int, default=128, help="The layer size")
parser.add_argument("-b", "--batch", type=int, default=128, help="The batch size")
parser.add_argument(
    "-v",
    "--exponents",
    type=str,
    default="0136",
    help='The potential exponents to be considered (string, e.g. "016")',
)

# Parse the arguments
args = parser.parse_args()

# Store the arguments in variables
PREFIX = args.prefix
RCUT = args.rcut
STRIDE = args.stride
N_EPOCHS = args.epochs
LAYER_SIZE = args.layer
BATCH_SIZE = args.batch
EXPONENTS = args.exponents

####
# UTILITY FUNCTION DEFINITIONS
####


def split_mono(f):
    symbols_a = []
    positions_a = []
    symbols_b = []
    positions_b = []
    on_first = True
    prev_s = "C"
    for s, p in zip(f.symbols, f.positions):
        if s == "C" and prev_s != "C":
            on_first = False
        prev_s = s
        if on_first:
            symbols_a.append(s)
            positions_a.append(p)
        else:
            symbols_b.append(s)
            positions_b.append(p)
    # NB : this assumes monomers are in the right order, i.e. that a "CA" molecule will
    # have "C" first and "A" next this is not true so these assignments may be wrong
    mono_a = ase.Atoms(symbols_a, positions_a)
    mono_a.info["distance"] = 0
    mono_a.info["distance_initial"] = 0
    mono_a.info["label"] = f.info["label"][0]
    mono_a.info["energy"] = f.info["energyA"]
    mono_b = ase.Atoms(symbols_b, positions_b)
    mono_b.info["distance"] = 0
    mono_b.info["distance_initial"] = 0
    mono_b.info["label"] = f.info["label"][1]
    mono_b.info["energy"] = f.info["energyB"]
    return mono_a, mono_b


########
# LOADS STUFF
##########

print("Loading structures")
all_frames = ase.io.read(
    "/scratch/loche/datasets/bioDimers_forces_relaxed_pbc.xyz",
    "::" + str(STRIDE),
)

# Use all for showing the whole dataset
select_labels = "all"

if select_labels != "all":
    idx = [i for i, f in enumerate(all_frames) if f.info["label"] == select_labels]
    frames = [f for f in all_frames if f.info["label"] == select_labels]
else:
    frames = all_frames
    idx = list(range(len(all_frames)))


idxpair = 0
d = frames[0].info["distance"]
for f in frames:
    if f.info["distance"] < d:
        idxpair += 1
    d = f.info["distance"]
    f.info["indexAB"] = idxpair


mono = []
for f in frames:
    mono_a, mono_b = split_mono(f)
    # this avoids adding "wrong" monomers. see split_mono. ugly workaround but it works
    if mono_a.info["label"] == mono_b.info["label"]:
        mono.append(mono_a)
        mono.append(mono_b)

emono = []
for m in mono:
    emono.append(m.info["energy"])
euni, eidx = np.unique(emono, return_index=True)
mono_unique = [mono[idx] for idx in eidx]


mono_true_unique = [mono_unique[0]]
emono = [mono_unique[0].info["energy"]]
for m in mono:
    if np.abs(np.array(emono) - m.info["energy"]).min() > 2e-3:
        mono_true_unique.append(m)
        emono.append(m.info["energy"])

# THESE ARE THE FINAL FRAMES WE USE
combined_frames = frames + mono_true_unique

# COMPOSITION BASELINING
co_calculator = AtomicComposition(per_structure=True)
co_descriptor = co_calculator.compute(combined_frames)
co = co_descriptor.keys_to_properties("species_center")

y = np.array([f.info["energy"] for f in combined_frames])
x = co[0].values

rcv = RidgeCV(alphas=np.geomspace(1e-8, 1e2, 10), fit_intercept=False)
rcv.fit(x, y)

yp_base = rcv.predict(x)

y_diff = y - yp_base

#########
# COMPUTE FEATURES
#########

for f in combined_frames:
    f.cell = [30, 30, 30]
    f.pbc = True


# RHO
hypers_sr = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

# LODE 1
print("Computing rhoi")
calculator_soap = SphericalExpansion(**hypers_sr)

try:
    descriptor_soap = equistore.load(PREFIX + "-descriptor_soap.npz")
except:
    print(" ... didn't find restart")
    ps_soap = PowerSpectrum(calculator_soap)
    descriptor_soap = ps_soap.compute(combined_frames)
    descriptor_soap = descriptor_soap.keys_to_properties("species_center")
    equistore.save(PREFIX + "-descriptor_soap.npz", descriptor_soap)

# LODE 1
hypers_lr = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 1.0,
    "radial_basis": {"Gto": {}},
    "potential_exponent": 1,
    "center_atom_weight": 1.0,
}

# Use monomial basis for LODE
rad = KspaceRadialBasis(
    "monomial_spherical",
    max_radial=hypers_lr["max_radial"],
    max_angular=hypers_lr["max_angular"],
    projection_radius=hypers_lr["cutoff"],
    orthonormalization_radius=hypers_lr["cutoff"],
)

k_cut = 1.2 * np.pi / hypers_lr["atomic_gaussian_width"]
spline_points = rad.spline_points(cutoff_radius=k_cut, requested_accuracy=1e-8)

hypers_lr["radial_basis"] = {
    "TabulatedRadialIntegral": {
        "points": spline_points,
        "center_contribution": [0.0 for _ in range(hypers_lr["max_radial"])],
    }
}

# LODE 1
hypers_lr["potential_exponent"] = 1
print("Computing V1i")
try:
    descriptor_lode1 = equistore.load(PREFIX + "-descriptor_lode1.npz")
except:
    print(" ... didn't find restart")
    calculator_lode1 = PowerSpectrum(
        calculator_soap, LodeSphericalExpansion(**hypers_lr)
    )
    descriptor_lode1 = calculator_lode1.compute(combined_frames)
    descriptor_lode1 = descriptor_lode1.keys_to_properties("species_center")
    equistore.save(PREFIX + "-descriptor_lode1.npz", descriptor_lode1)

# LODE 3
hypers_lr["potential_exponent"] = 3
print("Computing V3i")
try:
    descriptor_lode3 = equistore.load(PREFIX + "-descriptor_lode3.npz")
except:
    print(" ... didn't find restart")
    calculator_lode3 = PowerSpectrum(
        calculator_soap, LodeSphericalExpansion(**hypers_lr)
    )
    descriptor_lode3 = calculator_lode3.compute(combined_frames)
    descriptor_lode3 = descriptor_lode3.keys_to_properties("species_center")
    equistore.save(PREFIX + "-descriptor_lode3.npz", descriptor_lode3)

# LODE 6
hypers_lr["potential_exponent"] = 6
print("Computing V6i")
try:
    descriptor_lode6 = equistore.load(PREFIX + "-descriptor_lode6.npz")
except:
    print(" ... didn't find restart")
    calculator_lode6 = PowerSpectrum(
        calculator_soap, LodeSphericalExpansion(**hypers_lr)
    )
    descriptor_lode6 = calculator_lode6.compute(combined_frames)
    descriptor_lode6 = descriptor_lode6.keys_to_properties("species_center")
    equistore.save(PREFIX + "-descriptor_lode6.npz", descriptor_lode6)


rho2i = descriptor_soap.block().values
rhov1i = descriptor_lode1.block().values
rhov3i = descriptor_lode3.block().values
rhov6i = descriptor_lode6.block().values

del descriptor_soap
del descriptor_lode1
del descriptor_lode3
del descriptor_lode6

####
# Standardizing features
####
print("Standardizing features")
scaler = StandardFlexibleScaler()

rho2i = scaler.fit_transform(rho2i)
rhov1i = scaler.fit_transform(rhov1i)
rhov3i = scaler.fit_transform(rhov3i)
rhov6i = scaler.fit_transform(rhov6i)

####
# Join Features and convert to torch
####
device = "cpu"

lfeats = []
if "0" in EXPONENTS:
    lfeats.append(torch.tensor(rho2i))
if "1" in EXPONENTS:
    lfeats.append(torch.tensor(rhov1i))
if "3" in EXPONENTS:
    lfeats.append(torch.tensor(rhov3i))
if "6" in EXPONENTS:
    lfeats.append(torch.tensor(rhov6i))
feats = torch.hstack(lfeats)

del rho2i
del rhov1i
del rhov3i
del rhov6i

####
# Training
####
print("Setting up training")

distance = np.array([f.info["distance"] for f in combined_frames])
emono = np.array(
    [
        (
            f.info["energyA"] + f.info["energyB"]
            if "energyA" in f.info
            else f.info["energy"]
        )
        for f in combined_frames
    ]
)

itrain = np.array(
    [
        i
        for i in range(len(combined_frames))
        if combined_frames[i].info["distance"]
        - combined_frames[i].info["distance_initial"]
        < RCUT
    ]
)
itest = np.array(
    [
        i
        for i in range(len(combined_frames))
        if combined_frames[i].info["distance"]
        - combined_frames[i].info["distance_initial"]
        >= RCUT
    ]
)

targets = torch.tensor((y - yp_base))

rhoi = calculator_soap.compute(combined_frames)
rhoi = rhoi.keys_to_properties(["species_neighbor"])
smpl = rhoi.keys_to_samples(["species_center"])[0].samples

structure_samples = torch.tensor(
    smpl.values.reshape(len(smpl), len(smpl.names)), dtype=torch.int32, device=device
)


def loss_mse(predicted, actual):
    return torch.sum((predicted.flatten() - actual.flatten()) ** 2)


layer_size = LAYER_SIZE
energy_model = torch.nn.Sequential(
    torch.nn.Linear(feats.shape[-1], layer_size),
    torch.nn.Tanh(),
    torch.nn.LayerNorm(layer_size),
    torch.nn.Linear(layer_size, layer_size),
    torch.nn.Tanh(),
    torch.nn.LayerNorm(layer_size),
    torch.nn.Linear(layer_size, 1),
)

try:
    energy_model.load_state_dict(torch.load(PREFIX + "-model.torch"))
    print("Loaded initial weights from checkpoint")
except:
    print("Starting from scratch")


# optimizer = torch.optim.AdamW(energy_model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(energy_model.parameters(), lr=1e-3)

# Decay learning rate by a factor of 0.5 every 50 epochs after step 300
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)

print("Running training with ntrain=", len(itrain))
n_epochs = N_EPOCHS
batch_size = BATCH_SIZE

batch_idx = itrain.copy()
for epoch in range(n_epochs):
    np.random.shuffle(batch_idx)

    # manual dataloder...
    for ibatch in range(len(batch_idx) // batch_size):
        batch_structs = batch_idx[ibatch * batch_size : (ibatch + 1) * batch_size]
        batch_sel = np.where(np.isin(structure_samples[:, 0], batch_structs))[0]
        batch_samples = structure_samples[batch_sel, 0]
        batch_feats = feats[batch_sel]
        batch_tgt = targets[batch_structs]

        predicted = energy_model(batch_feats)
        predicted_structure = torch.zeros((len(targets), 1), device=device)
        predicted_structure.index_add_(0, batch_samples, predicted)
        loss = loss_mse(predicted_structure[batch_structs], batch_tgt)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    if epoch >= 1200:
        scheduler.step()

    if epoch % 10 == 0:
        predicted = energy_model(feats)
        predicted_structure = torch.zeros((len(targets), 1), device=device)
        predicted_structure.index_add_(0, structure_samples[:, 0], predicted)

        print(
            "Epoch:",
            epoch,
            "Energy RMSE: train ",
            np.sqrt(loss.detach().cpu().numpy().flatten()[0] / len(itrain)),
            "test",
            np.sqrt(
                loss_mse(predicted_structure[itest], targets[itest])
                .detach()
                .cpu()
                .numpy()
                .flatten()[0]
                / len(itest)
            ),
        )

        sys.stdout.flush()

        for i, f in enumerate(combined_frames):
            f.info["predicted_energy"] = (
                yp_base[i] + predicted_structure[i].detach().cpu().numpy().flatten()[0]
            )
            f.info["predicted_binding"] = (
                yp_base[i]
                + predicted_structure[i].detach().cpu().numpy().flatten()[0]
                - emono[i]
            )
            f.info["binding"] = y[i] - emono[i]
            f.info["error_binding"] = (
                yp_base[i]
                + predicted_structure[i].detach().cpu().numpy().flatten()[0]
                - y[i]
            )
            if i in itrain:
                f.info["split"] = "train"
            else:
                f.info["split"] = "test"

        torch.save(energy_model.state_dict(), PREFIX + "-checkpoint.torch")
        ase.io.write(PREFIX + "-checkpoint.xyz", combined_frames)

torch.save(energy_model.state_dict(), PREFIX + "-model.torch")

predicted = energy_model(feats)
predicted_structure = torch.zeros((len(targets), 1), device=device)
predicted_structure.index_add_(0, structure_samples[:, 0], predicted)

for i, f in enumerate(combined_frames):
    f.info["predicted_energy"] = (
        yp_base[i] + predicted_structure[i].detach().cpu().numpy().flatten()[0]
    )
    f.info["predicted_binding"] = (
        yp_base[i]
        + predicted_structure[i].detach().cpu().numpy().flatten()[0]
        - emono[i]
    )
    f.info["binding"] = y[i] - emono[i]
    f.info["error_binding"] = (
        yp_base[i] + predicted_structure[i].detach().cpu().numpy().flatten()[0] - y[i]
    )
    if i in itrain:
        f.info["split"] = "train"
    else:
        f.info["split"] = "test"

ase.io.write(PREFIX + "-final.xyz", combined_frames)
