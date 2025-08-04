import os
import tempfile
import numpy as np
import dssr_metallated

# extracts individual models from a multi-model PDB file
# returns a generator that yields (model_number, model_lines) tuples.
def extract_models_from_trajectory(trajectory_file):  
    with open(trajectory_file, 'r') as f:
        current_model = None
        model_lines = []
        header_lines = []
        
        for line in f:
            line = line.strip()
            
            # store header information
            if line.startswith('CRYST1') and not header_lines:
                header_lines.append(line)
            
            # start of a new model
            elif line.startswith('MODEL'):
                if current_model is not None and model_lines:
                    # yield the previous model
                    yield current_model, header_lines + model_lines
                
                # parse model number
                current_model = int(line.split()[1])
                
                model_lines = [line]
            
            # end of current model
            elif line.startswith('ENDMDL'):
                if current_model is not None:
                    model_lines.append(line)
                    yield current_model, header_lines + model_lines
                
                # reset for next model
                model_lines = []
            
            # regular atom/ter lines
            elif current_model is not None and (line.startswith('ATOM') or line.startswith('TER')):
                model_lines.append(line)
        
        # handle case where file doesn't end with ENDMDL
        if current_model is not None and model_lines:
            yield current_model, header_lines + model_lines

# write model lines to a temporary PDB file
def write_temporary_pdb(model_lines, temp_file):
    with open(temp_file, 'w') as f:
        for line in model_lines:
            f.write(line + '\n')

# process all models in a trajectory file and calculate propeller angles.
def process_trajectory(trajectory_file, output_file=None, max_models=None):
    results = []
    processed_count = 0

    # open output file if specified
    output_handle = None
    if output_file:
        output_handle = open(output_file, 'w')
        output_handle.write("Model\tPair\tBase1\tBase2\tPropeller_Angle\n")

    print(f"Processing: {trajectory_file}")
    if max_models:
        print(f"Maximum models to process: {max_models}")

    # create temporary file for individual models
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
        temp_filename = temp_file.name

    model_count = 0
    for model_num, model_lines in extract_models_from_trajectory(trajectory_file):
        model_count += 1

        # check if we've reached the maximum number of models
        if max_models and processed_count >= max_models:
            break

        processed_count += 1

        # write current model to temporary file
        write_temporary_pdb(model_lines, temp_filename)

        # process the model
        print(f"\n" + "="*60)
        total_models_str = str(max_models) if max_models else "?"
        print(f"Processing Model {model_num} ({processed_count}/{total_models_str})")
        print("="*60)

        # get the results
        model_results = process_pdb_return_results(temp_filename)

        # store results
        for pair_num, pair_data in enumerate(model_results, 1):
            r1, r2, angle = pair_data
            result_entry = {
                'model': model_num,
                'pair': pair_num,
                'base1': f"{r1[2]}{r1[1]}",
                'base2': f"{r2[2]}{r2[1]}",
                'angle': angle
            }
            results.append(result_entry)

            # write to output file if specified
            if output_handle:
                output_handle.write(f"{model_num}\t{pair_num}\t{result_entry['base1']}\t{result_entry['base2']}\t{angle:.4f}\n")

    # clean up temporary file
    if os.path.exists(temp_filename):
        os.unlink(temp_filename)

    # close output file if opened
    if output_handle:
        output_handle.close()

    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print("="*60)
    print(f"Total models processed: {processed_count}")
    print(f"Total base pairs analyzed: {len(results)}")

    if output_file:
        print(f"Results saved to: {output_file}")

    return results

# modified version of process_pdb that returns results instead of just printing
def process_pdb_return_results(filename):
    # read residues and base types (ignore metals)
    residues, base_type, _ = dssr_metallated.read_pdb_file(filename)
    
    frames = {}
    
    for res_key, atoms in residues.items():
        # make local frame for each base
        btype = base_type[res_key]
        required_ring_atoms = [atom for atom in ['N9','C8','N7','C5','C4','N3','C2','N1'] if btype == 'purine' and atom in atoms] + \
                              [atom for atom in ['N1','C2','N3','C4','C5','C6', 'O2'] if btype == 'pyrimidine' and atom in atoms]
        
        if len(required_ring_atoms) < 4:
            continue
        
        if "C1'" not in atoms:
            continue
        
        ref_atom = 'C8' if btype == 'purine' else 'C6'
        if ref_atom not in atoms:
            continue
            
        frame = dssr_metallated.base_frame(atoms, btype)
        frames[res_key] = frame

    sequential_pairs = dssr_metallated.find_sequential_pairs(residues, base_type)
    pairs = [(r1, r2) for r1, r2 in sequential_pairs]

    # calculate propeller angles and return results
    results = []
    for r1, r2 in pairs:
        if r1 in frames and r2 in frames:
            angle = dssr_metallated.calculate_propeller_angle(frames[r1], frames[r2])
            results.append((r1, r2, angle))
    
    return results

if __name__ == "__main__":
    trajectory_file = "metallated/md_SingleCHgT_solv_nWI.pdb"
    output_file = "results/propeller_angles_results_CHgT.txt"
    
    results = process_trajectory(
        trajectory_file=trajectory_file,
        output_file=output_file,
        max_models=100000  
    )