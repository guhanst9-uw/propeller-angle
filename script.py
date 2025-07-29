import numpy as np

# All ring atoms per base type
RING_ATOMS = {
    'purine':     ['N9','C8','N7','C5','C4','N3','C2','N1'],
    'pyrimidine': ['N1','C2','O2','N3','C4','C5','C6']
}
SUGAR_ATOM = "C1'"
REF_ATOM = {'purine':'C8','pyrimidine':'C6'}

def read_pdb_file(filename):
    residues = {}
    base_type = {}
    with open(filename) as f:
        for line in f:
            # skip non-atom lines
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            res_num = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            key = (chain_id, res_num, res_name)
            if key not in residues:
                residues[key] = {}
                # classify base type by last letter
                letter = res_name[-1].upper()
                base_type[key] = 'purine' if letter in 'AG' else 'pyrimidine'
            residues[key][atom_name] = np.array([x,y,z])
    return residues, base_type

def base_frame(coords, btype):
    # plane normal by PCA
    pts = np.stack([coords[a] for a in RING_ATOMS[btype]])
    center = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - center)
    z = Vt[-1]
    z /= np.linalg.norm(z)
    # orient z toward sugar
    sugar_vec = coords[SUGAR_ATOM] - center
    if np.dot(z, sugar_vec) < 0:
        z = -z
    # y-axis
    y = coords[REF_ATOM[btype]] - center
    y -= np.dot(y, z) * z
    y /= np.linalg.norm(y)
    return center, y, z

def torsion(z1, z2, axis):
    # signed torsion of z2→z1 about axis
    b2 = axis / np.linalg.norm(axis)
    n1 = np.cross(axis, z2) # note swap: z2→z1
    n2 = np.cross(axis, z1)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2)
    return np.degrees(np.arctan2(y, x))

def normalize_propeller(phi):
    # bring into (-90, +90)
    if phi > 90:
        return phi - 180
    if phi < -90:
        return phi + 180
    return phi

def calculate_propeller_angle(f1, f2):
    _, y1, z1 = f1
    _, y2, z2 = f2
    # flip if y-axes disagree
    if np.dot(y1, y2) < 0:
        y2 = -y2
    axis = y1 + y2
    axis /= np.linalg.norm(axis)
    raw = torsion(z1, z2, axis)
    return normalize_propeller(raw)

def find_base_pairs(residues):
    # group residues by chain
    chains = {}
    for key in residues:
        chain = key[0]
        chains.setdefault(chain, []).append(key)
    c1, c2 = sorted(chains.keys())
    # match pairs by order in chain 1 and reverse order in chain 2
    r1 = sorted(chains[c1], key=lambda x: x[1])
    r2 = sorted(chains[c2], key=lambda x: x[1])
    pairs = []
    for b1, b2 in zip(r1, reversed(r2)):
        pairs.append((b1, b2))
    return pairs

def process_pdb(filename):
    # read residues and their base types from the PDB file
    residues, base_type = read_pdb_file(filename)
    frames = {}
    for res_key, atoms in residues.items():
        # make local frame for each base
        btype = base_type[res_key]
        frame = base_frame(atoms, btype)
        frames[res_key] = frame
    # find base pairs
    pairs = find_base_pairs(residues)
    for i, (r1, r2) in enumerate(pairs, 1):
        if r1 in frames and r2 in frames:
            angle = calculate_propeller_angle(frames[r1], frames[r2])
            print(f"Base Pair {i}: {r1[2]}{r1[1]}-{r2[2]}{r2[1]} Propeller Angle = {angle:.4f}")

if __name__ == "__main__":
    process_pdb("pdb/1bna.pdb")