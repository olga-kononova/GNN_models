"""
Main file for features creation
All other files has to be synchronized with this file
"""
import pandas as pd
import mendeleev as mv

if mv.__version__ < "0.7":
    elements = mv.elements.get_all_elements()

    elements_table = {str(el.symbol): (el.period, el.group_id)
                      for el in mv.elements.get_all_elements()
                      if not pd.isna(el.group_id)}
else:
    from mendeleev.fetch import fetch_table
    elements = fetch_table('elements')

    elements_table = {el: (p, int(g))
                      for el, p, g in zip(elements["symbol"], elements["period"], elements["group_id"])
                      if not pd.isna(g)}

## +1 is added for any non-standard element which will have p=0 and g=0
PERIOD_NUM = 7 + 1
GROUP_NUM = 18 + 1

## 1-4 matches neighbours amount and 0 for >4 neighbours
NEIGHBOURS_LEN = 5

## 1-3 matches raical electrons and 0 for >3 electrons
NUM_ELECTRONS = 4

BOND_TYPES = {"UNK": 0,
              "AROMATIC": 1,
              "SINGLE": 2,
              "DOUBLE": 3,
              "TRIPLE": 4
              }

BOND_STEREO = {"UNK": 0,
               "STEREOE": 1,
               "STEREONONE": 2,
               "STEREOZ": 3 
               }

CHIRAL_TAG = {"CHI_UNSPECIFIED": 0,
              "CHI_TETRAHEDRAL_CCW": 1,
              "CHI_TETRAHEDRAL_CW": 2  
             }

formal_charge = {0: 0,
                 1: 1,
                 -1: 2,
                 10: 3} # for abnormal charge

ATOM_SYMBOLS = {'Unk': 0}
for i, el in enumerate(sorted(elements_table.keys())):
    ATOM_SYMBOLS[el] = i + 1

HYBRIDIZATION = {"UNK": 0,
                 "S": 1,
                 "SP": 2,
                 "SP2": 3,
                 "SP3": 4,
                 "SP3D": 5,
                 "SP3D2": 6
                 }


def get_features_dim():
    """
    :return: number of atomic features, number of edge features
    """
    return len(ATOM_SYMBOLS) + PERIOD_NUM + GROUP_NUM + NEIGHBOURS_LEN + len(HYBRIDIZATION) + len(formal_charge) + 5, \
           len(BOND_TYPES)


def get_features_description():
    return "Atomic: Symbol ({}), Period ({}), Group ({}), Neighbours ({}), "\
                    "Hybridization ({}), Chiral tag ({}), Formal charge({}), "\
                    "Radical Electrons, Rings ({}); " \
           "Edge: Bond type ({})".format(len(ATOM_SYMBOLS),
                                         PERIOD_NUM,
                                         GROUP_NUM,
                                         NEIGHBOURS_LEN,
                                         len(HYBRIDIZATION),
                                         len(CHIRAL_TAG),
                                         len(formal_charge),
                                         NUM_ELECTRONS,
                                         5,
                                         len(BOND_TYPES))


def get_edges_from_bonds(molecule_bonds):
    begins = []
    ends = []
    features = []
    for b in molecule_bonds:
        begins.append(b.GetBeginAtomIdx())
        ends.append(b.GetEndAtomIdx())
        features.append(get_bond_features(b))
        begins.append(b.GetEndAtomIdx())
        ends.append(b.GetBeginAtomIdx())
        features.append(get_bond_features(b))
    return begins, ends, features


def get_atomic_features(atom):
    atom_symbol = str(atom.GetSymbol())
    pg = elements_table.get(atom_symbol, (0, 0))

    atom_symbol_feature = [0] * len(ATOM_SYMBOLS)
    atom_name_id = ATOM_SYMBOLS.get(atom_symbol, 0)
    atom_symbol_feature[atom_name_id] = 1

    period = [0] * PERIOD_NUM
    period[pg[0]] = 1
    group = [0] * GROUP_NUM
    group[pg[1]] = 1

    hybrid_feature = [0] * len(HYBRIDIZATION)
    hybrid_id = HYBRIDIZATION.get(str(atom.GetHybridization()), 0)
    hybrid_feature[hybrid_id] = 1

    chiral_features = [0] * len(CHIRAL_TAG)
    chiral_id = CHIRAL_TAG.get(str(atom.GetChiralTag()), 0)
    chiral_features[chiral_id] = 1

    charge = [0] * len(formal_charge)
    charge[formal_charge.get(atom.GetFormalCharge(), 3)] = 1

    neighbors_feature = [0] * NEIGHBOURS_LEN
    n_count = len(atom.GetNeighbors())
    if n_count > 4:
        n_count = 0
    neighbors_feature[n_count] = 1

    radical_electrons = [0] * NUM_ELECTRONS
    e_count = atom.GetNumRadicalElectrons()
    if e_count > 3:
        e_count = 0
    radical_electrons[e_count] = 1

    features = atom_symbol_feature + \
               period + \
               group + \
               hybrid_feature + \
               chiral_features + \
               neighbors_feature + \
               charge + \
               radical_electrons 

    features.append(int(atom.IsInRingSize(3)))
    features.append(int(atom.IsInRingSize(4)))
    features.append(int(atom.IsInRingSize(5)))
    features.append(int(atom.IsInRingSize(6)))
    features.append(int(atom.GetIsAromatic()))

    return features


def get_bond_features(bond):
    bond_type = [0] * len(BOND_TYPES)
    bond_type_id = BOND_TYPES.get(str(bond.GetBondType()), 0)
    bond_type[bond_type_id] = 1
    
    conjugate = [int(bond.GetIsConjugated())]
    
    stereo = [0] * len(BOND_STEREO)
    stereo_id = BOND_STEREO.get(str(bond.GetStereo()), 0)
    stereo[stereo_id] = 1

    return bond_type + conjugate + stereo


def featurize_molecule(rdk_mol):
    x = [get_atomic_features(atom) for atom in rdk_mol.GetAtoms()]
    start_idx, end_idx, edge = get_edges_from_bonds(rdk_mol.GetBonds())
    
    return dict(x=x,
                edge_attr=edge,
                edge_index=[start_idx, end_idx])
