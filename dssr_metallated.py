import numpy as np

# all ring atoms per base type
RING_ATOMS = {
    'purine':     ['N9','C8','N7','C5','C4','N3','C2','N1'],
    'pyrimidine': ['N1','C2','N3','C4','C5','C6', 'O2'] 
}
SUGAR_ATOM = "C1'"
REF_ATOM = {'purine':'C8','pyrimidine':'C6'}

# metal ions to look for different types of metallated dna
METAL_IONS = ['AG', 'HG', 'CU', 'ZN', 'NI', 'PT']

def read_pdb_file(filename):
    residues = {}
    base_type = {}
    metals = {}
    
    with open(filename) as f:
        for line in f:
            # skip non-atom lines
            if not line.startswith("ATOM"):
                continue
            
            # use position-based parsing for this format
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            if not chain_id or chain_id.isdigit():
                chain_id = 'A'  # default chain
            
            # handle the case where there's no chain ID and res_num starts at position 22
            try:
                res_num_str = line[22:26].strip()
                if not res_num_str:
                    # try alternative positions
                    res_num_str = line[23:27].strip()
                res_num = int(res_num_str)
            except (ValueError, IndexError):
                # fallback
                parts = line.split()
                if len(parts) < 8:
                    continue

                res_num = None
                coord_start = None

                for i, part in enumerate(parts[3:7]):  # reasonable range
                    if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                        res_num = int(part)
                        coord_start = i + 4
                        break

                # skip if no valid residue number was found or not enough coordinates
                if res_num is None or coord_start + 2 >= len(parts):
                    continue

                x = float(parts[coord_start])
                y = float(parts[coord_start + 1])
                z = float(parts[coord_start + 2])
            
            else:
                # standard position-based coordinates
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
             
            # process metal ions
            if atom_name in METAL_IONS or res_name in METAL_IONS:
                key = (chain_id, res_num, atom_name)
                metals[key] = np.array([x, y, z])
                continue
            
            # process DNA bases (including modified ones)
            if res_name not in ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C', 'DT5', 'DA3', 'DC5', 'DG3', 'DC1', 'DM1']:
                continue
                
            key = (chain_id, res_num, res_name)
            if key not in residues:
                residues[key] = {}
                # classify base type by last letter or specific residue name
                if res_name in ['DA', 'A', 'DA3']:
                    base_type[key] = 'purine'
                elif res_name in ['DG', 'G', 'DG3']:
                    base_type[key] = 'purine'
                elif res_name in ['DT', 'T', 'DT5', 'DM1']:  # DM1 is modified thymine
                    base_type[key] = 'pyrimidine'
                elif res_name in ['DC', 'C', 'DC5', 'DC1']:
                    base_type[key] = 'pyrimidine'
                else:
                    letter = res_name[-1].upper()
                    base_type[key] = 'purine' if letter in 'AG' else 'pyrimidine'
                    
            residues[key][atom_name] = np.array([x,y,z])
                
    return residues, base_type, metals

def base_frame(coords, btype):
    # check if we have the required atoms
    required_ring_atoms = [atom for atom in RING_ATOMS[btype] if atom in coords]
    if len(required_ring_atoms) < 4:
        raise ValueError(f"Not enough ring atoms: only found {required_ring_atoms}")
    
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
    y = np.dot(np.cross(z2_proj, z1_proj), axis)  

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

    # flip axis if dot with y1 is negative to keep direction
    if np.dot(axis, y1) < 0:
        axis = -axis

    raw = torsion(z1, z2, axis)
    return normalize_propeller(raw)

# find pairs by sequential pairing within chains or between chains
def find_sequential_pairs(residues, base_type):
    # group residues by chain
    chains = {}
    for key in residues:
        chain = key[0]
        chains.setdefault(chain, []).append(key)
    
    pairs = []
    
    if len(chains) >= 2:
        # inter-chain pairing (antiparallel)
        c1, c2 = sorted(chains.keys())[:2]
        
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
        
        min_valid = min(len(valid_r1), len(valid_r2))
        for i in range(min_valid):
            pairs.append((valid_r1[i], valid_r2[-(i+1)]))
    
    else:
        # intra-chain pairing (less common for DNA)
        for chain_residues in chains.values():
            sorted_residues = sorted(chain_residues, key=lambda x: x[1])
            for i in range(0, len(sorted_residues)-1, 2):
                if i+1 < len(sorted_residues):
                    pairs.append((sorted_residues[i], sorted_residues[i+1]))
    
    return pairs

def process_pdb(filename):
    print(f"\nPDB File: {filename}")
    
    # read residues, their base types, and metals from the PDB file
    residues, base_type, metals = read_pdb_file(filename)
    
    print(f"Found {len(residues)} DNA residues")
    print(f"Found {len(metals)} metal ions")
    
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
            
        try:
            frame = base_frame(atoms, btype)
            frames[res_key] = frame
        except ValueError as e:
            skipped.append((res_key, str(e)))
    
    if skipped:
        print(f"Skipped {len(skipped)} residues due to missing atoms:")
        for res_key, error in skipped:
            print(f"  {res_key}: {error}")
    
    sequential_pairs = find_sequential_pairs(residues, base_type)
    pairs = [(r1, r2, None) for r1, r2 in sequential_pairs]
    
    print(f"\nFound {len(pairs)} base pairs")
    print("\nPropeller Angles:")
    
    for i, pair_info in enumerate(pairs, 1):
        if len(pair_info) == 3:
            r1, r2, metal = pair_info
            metal_info = f" (via {metal})" if metal else ""
        else:
            r1, r2 = pair_info
            metal_info = ""
            
        if r1 in frames and r2 in frames:
            angle = calculate_propeller_angle(frames[r1], frames[r2])
            print(f"Base Pair {i}: {r1[2]}{r1[1]}-{r2[2]}{r2[1]}{metal_info} Propeller Angle = {angle:.4f}°")
        else:
            missing = []
            if r1 not in frames:
                missing.append(str(r1))
            if r2 not in frames:
                missing.append(str(r2))

    print("\n")

    print("\nBase Pair Order Check:")
    for i, (r1, r2) in enumerate(sequential_pairs, 1):
        print(f"Pair {i}: {r1[2]}{r1[1]} - {r2[2]}{r2[1]}")

if __name__ == "__main__": 
    process_pdb("metallated/sampleModels/model1_NoMetal.pdb")  # change (to try sample models, use metallated/sampleModels/model1_****.pdb) 