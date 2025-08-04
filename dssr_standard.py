import numpy as np

# all ring atoms per base type
RING_ATOMS = {
    'purine':     ['N9','C8','N7','C5','C4','N3','C2','N1'],
    'pyrimidine': ['N1','C2','N3','C4','C5','C6', 'O2'] 
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
            
            # split by whitespace for more robust parsing
            parts = line.split()
            if len(parts) < 9:
                continue
                
            atom_name = parts[2]
            res_name = parts[3]
            chain_id = parts[4]
            res_num = int(parts[5])
            x = float(parts[6])
            y = float(parts[7])
            z = float(parts[8])
            
            # handle cases where chain ID might be missing or merged with res_num
            if not chain_id.isalpha():
                # fallback to position-based
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                if not chain_id:
                    chain_id = 'A'
                res_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            
            # only process DNA bases
            if res_name not in ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']:
                continue
                
            key = (chain_id, res_num, res_name)
            if key not in residues:
                residues[key] = {}
                # classify base type by last letter
                if res_name in ['DA', 'A']:
                    base_type[key] = 'purine'
                elif res_name in ['DG', 'G']:
                    base_type[key] = 'purine'
                elif res_name in ['DT', 'T']:
                    base_type[key] = 'pyrimidine'
                elif res_name in ['DC', 'C']:
                    base_type[key] = 'pyrimidine'
                else:
                    letter = res_name[-1].upper()
                    base_type[key] = 'purine' if letter in 'AG' else 'pyrimidine'
                    
            residues[key][atom_name] = np.array([x,y,z])
                
    return residues, base_type

def base_frame(coords, btype):
    # check if we have the required atoms
    required_ring_atoms = [atom for atom in RING_ATOMS[btype] if atom in coords]
    if len(required_ring_atoms) < 4:
        raise ValueError(f"Insufficient ring atoms: only found {required_ring_atoms}")
    
    if SUGAR_ATOM not in coords:
        raise ValueError(f"Missing sugar atom: {SUGAR_ATOM}")
    
    if REF_ATOM[btype] not in coords:
        raise ValueError(f"Missing reference atom: {REF_ATOM[btype]}")
    
    # plane normal by PCA
    pts = np.stack([coords[a] for a in required_ring_atoms])
    center = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - center)
    z = Vt[-1]
    z /= np.linalg.norm(z)

    # orient z away from sugar
    sugar_vec = coords[SUGAR_ATOM] - center
    if np.dot(z, sugar_vec) > 0:
        z = -z

    # y-axis
    y = coords[REF_ATOM[btype]] - center
    y -= np.dot(y, z) * z
    y /= np.linalg.norm(y)

    return center, y, z

def torsion(z1, z2, axis):
    axis = axis / np.linalg.norm(axis)
    z1_proj = z1 - np.dot(z1, axis) * axis
    z2_proj = z2 - np.dot(z2, axis) * axis
    z1_proj /= np.linalg.norm(z1_proj)
    z2_proj /= np.linalg.norm(z2_proj)

    x = np.dot(z1_proj, z2_proj)
    y = np.dot(np.cross(z2_proj, z1_proj), axis)  # flipped cross order here

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
        z2 = -z2

    axis = y1 + y2
    if np.linalg.norm(axis) < 1e-8:
        axis = y1  # fallback if vectors nearly opposite
    axis /= np.linalg.norm(axis)
    
    # force axis direction from base1 to base2 (optional)
    # flip axis if dot with y1 is negative to keep consistent direction
    if np.dot(axis, y1) < 0:
        axis = -axis

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
    
    # skip terminal residues that might be incomplete
    valid_r1 = []
    valid_r2 = []
    
    for res in r1:
        if res in residues and len(residues[res]) > 5:
            valid_r1.append(res)
    
    for res in r2:
        if res in residues and len(residues[res]) > 5:
            valid_r2.append(res)
    
    pairs = []
    min_valid = min(len(valid_r1), len(valid_r2))
    for i in range(min_valid):
        pairs.append((valid_r1[i], valid_r2[-(i+1)]))
    
    return pairs

def process_pdb(filename):
    print(f"\nPDB File: {filename}")
    
    # read residues and their base types from the PDB file
    residues, base_type = read_pdb_file(filename)
    
    print(f"Found {len(residues)} DNA residues")
    
    frames = {}
    skipped = []
    
    for res_key, atoms in residues.items():
        # make local frame for each base
        btype = base_type[res_key]
        required_ring_atoms = [atom for atom in RING_ATOMS[btype] if atom in atoms]
        if len(required_ring_atoms) < 4:
            skipped.append((res_key, f"Not enough ring atoms: only found {required_ring_atoms}"))
            continue
        
        if SUGAR_ATOM not in atoms:
            skipped.append((res_key, f"Missing sugar atom: {SUGAR_ATOM}"))
            continue
        
        if REF_ATOM[btype] not in atoms:
            skipped.append((res_key, f"Missing reference atom: {REF_ATOM[btype]}"))
            continue
            
        frame = base_frame(atoms, btype)
        frames[res_key] = frame
    
    if skipped:
        print(f"Skipped {len(skipped)} residues due to missing atoms:")
        for res_key, error in skipped:
            print(f"  {res_key}: {error}")
    
    # find base pairs
    pairs = find_base_pairs(residues)
    
    print("\n")
    
    for i, (r1, r2) in enumerate(pairs, 1):
        if r1 in frames and r2 in frames:
            angle = calculate_propeller_angle(frames[r1], frames[r2])
            print(f"Base Pair {i}: {r1[2]}{r1[1]}-{r2[2]}{r2[1]} Propeller Angle = {angle:.4f}")

    print("\n")
    

if __name__ == "__main__": 
    process_pdb("pdb/1bna.pdb") 