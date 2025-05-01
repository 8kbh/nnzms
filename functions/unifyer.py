import numpy as np
import pandas as pd

def check_nan(value):
    return pd.isna(value)

def unify_shape(shape, check_nan = check_nan):
    shape_types = {
        'cubic': 'cubic',
        'octahedral': 'octahedral',
        'rods': 'rods',
        'sheet': 'sheet',
        'flower': 'flower',
        
        'truncated cuboctahedron': 'polyhedral',
        'decahydron': 'polyhedral',
        'nanopolyhedra': 'polyhedral',
        'truncated octahedron': 'polyhedral',
        'icosahedral': 'polyhedral',
        'tetrahedral': 'polyhedral',
        'cuboctahedral': 'polyhedral',
    
        'hexahedrons': 'cubic',
        'cuboid': 'cubic',
    
        'lenticular': 'ellipsoidal',
        'Ovoid-like': 'ellipsoidal',
        'ovoid': 'ellipsoidal',
    
        'erythrocyte-like': 'plates',
        'saucer-like': 'plates',
        'circular': 'plates',
    
        'mulberry-like clusters': 'nanocluster',
        'pseudo sphere-like aggregates': 'nanocluster',
        'clusters': 'nanocluster',
        'trimer': 'nanocluster',
        'aggregated': 'nanocluster',
    
        'rod-like particles': 'rods',
    
        'gel': 'bulk',
    
        'nanocables': 'nanofiber',
    
        'nanotextiles': 'nanofiber',
        'fibrillar network': 'nanofiber',
    
        'nanotubes@nanoflowers': 'flower',
    
        'cachet-like': 'tile',
        'concave': 'tile',
    
        'column': 'block',
    
        'spherical or oval': 'spherical',
        'rounded': 'spherical',
    
        'porous nanohybrid network': 'nanohybrid',
    
        'colloidal': 'nanocomposite',
    
        'shell': 'cage',
    
        ########################
        
        'sheet-like': 'sheet',
        'nanosheets': 'sheet',
        'Nanosheets': 'sheet',
        'nanosheet': 'sheet',
        'sheets': 'sheet',
        'sheetlike': 'sheet',
        'sheet-shaped': 'sheet',
        '2D nanosheet': 'sheet',
        'two-dimensional': 'sheet',
        'Thinly layered': 'sheet',
        '2D nanosheet': 'sheet',
    
        'hexagonal nanoplates': 'plates',
        'nanoplates': 'plates',
        'plate-like': 'plates',
        'square': 'plates',
        'rectangular': 'plates',
        'triangular plates': 'plates',
        'nanotriangles': 'plates',
        'hexagonal plate-like': 'plates',
        'lamellar': 'sheet',
        'hexagonal bilayer': 'sheet',
    
        'urchin-like': 'branched',
        'coral-like': 'branched',
        'forest': 'branched',
        'besom-like': 'branched',
        'multipod': 'branched',
        'nanowires with thorns': 'branched',
        'quasi-coral': 'branched',
        'algae-like': 'branched',
        'cauliflower-like': 'branched',
        
        'flower-like': 'flower',
        'nanoflower': 'flower',
        'nanoflowers': 'flower',
        'flower-shaped': 'flower',
        'flowerlike': 'flower',
        'ball-flower': 'flower',
        'dandelion-like': 'flower',
        'nanostructured hierarchical flower blooms': 'flower',
        'dahlia-like': 'flower',
        'Nanoflowers': 'flower',
    
        'nanorods': 'rods',
        'nanowires': 'rods',
        'Nanowires': 'rods',
        'nanowire': 'rods',
        'rods/wires': 'rods',
        'rod': 'rods',
        'rod-like': 'rods',
        'short rod-like': 'rods',
        'hexagonal rod': 'rods',
        'hexagonal rod-like': 'rods',
        'rods and pseudo spherical': 'rods',
        'rods and spheroids': 'rods',
    
        'quasi-spherical': 'spherical',
        'pseudo spherical': 'spherical',
        'pseudo-spherical': 'spherical',
        'spherical or ellipsoidal': 'spherical',
        'spheroidal or ellipsoidal': 'spherical',
        'elliptical or sphere': 'spherical',
        'spherical-like': 'spherical',
        'sphere-shaped': 'spherical',
        'spherical or spheroidal': 'spherical',
        'spherical and spindle': 'spherical',
        'Nanospheres': 'spherical',
        'nanospheres': 'spherical',
        'Spherical': 'spherical',
        'spheroidal': 'spherical',
        'round': 'spherical',
        'globular': 'spherical',
        'semi-spherical': 'spherical',
        'near-cubic': 'cubic',
    
        'nanoflakes': 'flakes',
        'flakes': 'flakes',
        'flake': 'flakes',
        'flaky': 'flakes',
        'flake-like': 'flakes',
        'wrinkled flakes': 'flakes',
        
        'octahedron': 'octahedral',
        'octahedrons': 'octahedral',
    
        'nanotubes': 'tubular',
        'nanotube': 'tubular',
        'belt-like': 'tubular',
        'belt': 'tubular',
        'nanobelts': 'tubular',
        'nanobelt': 'tubular',
        'nanoribbons': 'tubular',
        'nanoribbon': 'tubular',
        'ribbons': 'tubular',
    
        'cube': 'cubic',
        'nanocubes': 'cubic',
        'nanocube': 'cubic',
        'cubes': 'cubic',
        'cubic or polyhedral': 'cubic',
        'cubic and polyhedral': 'cubic',
        'quasi-cubic': 'cubic',
        'concave-cubic': 'cubic',
        'cube-like': 'cubic',
    
        'polyhedral': 'polyhedral',
        'polyhedron-shaped': 'polyhedral',
        'concave polyhedral': 'polyhedral',
        
        'nanocomposite': 'nanocomposite',
        'nanocomposites': 'nanocomposite',
        
        'nanohybrid': 'nanohybrid',
        'nanohybrids': 'nanohybrid',
    
        'nanoclusters': 'nanocluster',
        'nanocluster': 'nanocluster',
    
        'nanofibers': 'nanofiber',
        'nanofiber': 'nanofiber',
    
        'nanostructures': 'nanostructure',
        'nanostructure': 'nanostructure',
        'nanostructured': 'nanostructure',
    
        'core-shell': 'core-shell',
        'core-satellite': 'core-shell',
        
        'disk': 'plates',
        'nanodisks': 'plates',
    
        'prisms': 'prism',
        'prism-shaped': 'prism',
        'hexagonal prism': 'prism',
        'double cone hexagonal prism': 'prism',
        'hexagonal': 'prism',
        'slender bipyramidal hexagonal prism': 'prism',
        'bipyramidal': 'prism',
    
        'dodecahedron': 'dodecahedral',
        'dodecahedral': 'dodecahedral',
        'rhombic dodecahedral': 'dodecahedral',
        'rhombic dodecahedron': 'dodecahedral',
    
        'nanodots': 'dots',
        'dots-like': 'dots',
        
        'quantum dots': 'dots',
    
        'monomer': 'dots',
        'discrete': 'dots',
        
        'chain-like': 'chain',
        'Cyclic': 'chain',
        'helical': 'chain',
        'clew-like': 'chain',
        'toroid': 'chain',
        'uniparallel G-quadruplex': 'chain',
    
        'block': 'block',
        'blocky': 'block',
        'blockish': 'block',
    
        'dendritic': 'dendritic',
        'dendrite-like': 'dendritic',
        'dendritic-like': 'dendritic',
    
        'needle-like': 'needle',
        'nanoneedles': 'needle',
        'needle bundle-like': 'needle',
    
        'spindle-shaped': 'spindle',
        'spindle': 'spindle',
        'fusiform': 'spindle',
        'spindle and spherical': 'spindle',
        'rice-like': 'spindle',
    
        'ellipsoidal': 'ellipsoidal',
        'prolate ellipsoidal': 'ellipsoidal',
    
        'grain-like': 'grain',
        
        'vesicles': 'vesicle',
        'vesicle-like': 'vesicle',
        'vesicular': 'vesicle',
    
        'nanocages': 'cage',
        'nanoballoons': 'cage',
    
        'angular': 'irregular',
        'irregular': 'irregular',
        'irregular bulk': 'irregular',
        'irregular nanoparticles': 'irregular',
        'irregular polygon': 'irregular',
        'multi-shaped': 'irregular',
        'anisotropic': 'irregular',
        'nonspherical and interconnected': 'irregular',
        'wrinkled structures': 'irregular',
        
        'multibranched': 'branched',
        'porous branched-structure': 'branched',
    
        'tubular': 'tubular',
    
        'star-shaped': 'star',
        'nanostar': 'star',
        'nucleus-star heterogeneous nanostructure': 'star',
    
        'tile-like': 'tile',
        
        'graphene-like': 'graphene',
    
        'wrinkle paper-like': 'flakes',
        
        'cigar-shaped': 'elongated',
        'dumbbell': 'elongated',
        'elongate': 'elongated',
    
        'needle bundle-like': 'needle',
        
        'wheat sheaf-like': 'flower',
        
        'bamboo-like': 'tubular',
    
        'porous': 'porous',
        'nanosponges': 'porous',
        
        'chain-like': 'chain',
        
        'bulk': 'bulk',
        'massive': 'bulk',
    
        'chain-like': 'chain',  # Removed duplicate under 'nanocluster'
    
        'bipyramidal': 'polyhedral',  # Changed from 'prism'
    
        'concave': 'irregular',  # Changed from 'tile'
    
        'column': 'elongated',  # Changed from 'block'
    
        'spherical or ellipsoidal': 'ellipsoidal',
        'spheroidal or ellipsoidal': 'ellipsoidal',
        'elliptical or sphere': 'ellipsoidal',
    
        'hexagonal': 'plates',  # Changed from 'prism'
    
        'wheat sheaf-like': 'branched',  # Changed from 'flower'
    
        'nanoparticles': 'nanostructure',
        'Nanowires': 'nanofiber',
        'nanocrystals': 'nanostructure',
        'Nanosheets': 'sheet',
        'Nanoflowers': 'flower',
        'single twin': 'nanostructure',
        'particles': 'nanostructure',
        '2D nanonetwork': 'nanostructure',
        '3D-network': 'nanostructure',
        'cylindrical': 'rods',
        'hexamer': 'nanocluster',
        'hierarchical': 'nanostructure',
        'nanodiamonds': 'nanostructure',
        'particles-like': 'nanostructure',
    }

    if check_nan(shape):
        return np.nan

    shape_types = {key.lower(): value for key, value in shape_types.items()}
    try:
        return shape_types[shape.lower()]
    except:
        print(shape)
        return np.nan


def unify_activity(activity, check_nan=check_nan):
    activity_map = {
        # Glutathione peroxidase and variants
        'glutathione peroxidase': 'glutathione peroxidase',
        'glutathione peroxidase-like': 'glutathione peroxidase',
        'glutathione peroxidase (gpx)': 'glutathione peroxidase',
        'glutathione peroxidase 1': 'glutathione peroxidase',
        'glutathione peroxidase-1': 'glutathione peroxidase',
        'glutathione peroxidase ii': 'glutathione peroxidase',
        'glutathione peroxidase mimetic': 'glutathione peroxidase',
        'gpx': 'glutathione peroxidase',
        'gpx-1': 'glutathione peroxidase',
        'gsh-px': 'glutathione peroxidase',
        'gsh peroxidase': 'glutathione peroxidase',
        'non-se gsh-px': 'glutathione peroxidase',
        'seleno-dependent glutathione peroxidase': 'glutathione peroxidase',
        'selenium-dependent glutathione peroxidase': 'glutathione peroxidase',
        'selenium-independent glutathione peroxidase': 'glutathione peroxidase',
        'se-independent glutathione peroxidase': 'glutathione peroxidase',
        'glutathion peroxidase': 'glutathione peroxidase',
        'glutathione peroxidase-like': 'glutathione peroxidase',

        # Catalase
        'catalase': 'catalase',
        'catalase-like': 'catalase',
        'catalase and laccase': 'catalase',
        'catalase/peroxidase': 'catalase',
        'cat': 'catalase',

        # Superoxide dismutase
        'superoxide dismutase': 'superoxide dismutase',
        'superoxide dismutase-like': 'superoxide dismutase',
        'manganese superoxide dismutase': 'superoxide dismutase',
        'sod': 'superoxide dismutase',
        'sod and cat': 'superoxide dismutase',
        'sod/pod': 'superoxide dismutase',
        'copper-zinc superoxide dismutase': 'superoxide dismutase',
    
        # Peroxidase
        'peroxidase': 'peroxidase',
        'peroxidase-like': 'peroxidase',
        'peroxidase-oxidase': 'peroxidase',
        'catalase-peroxidase': 'peroxidase',
        'guaiacol peroxidase': 'peroxidase',
        'guaiacol peroxidase': 'peroxidase',
        'peroxidase/oxidase': 'peroxidase',
        'hydroxyl radical scavenger': 'peroxidase',
        'mn2+ peroxidase': 'peroxidase',
        'manganese peroxidase': 'peroxidase',
        'mn peroxidase': 'peroxidase',
        'iodide peroxidase': 'peroxidase',
        'tryptophan peroxidase': 'peroxidase',
        'fenton': 'peroxidase',
        'peroxygenase': 'peroxidase',
        'ascorbate peroxidase': 'peroxidase',
        'ascorbic acid peroxidase (apod)-like activity': 'peroxidase',
        'ascorbic acid oxidase (aao)-like activity': 'peroxidase',
        'chlorogenic acid peroxidase': 'peroxidase',
        'dopa peroxidase': 'peroxidase',
        'chloroperoxidase': 'peroxidase',
        'ferulic acid peroxidase': 'peroxidase',
        'glucose peroxidase': 'peroxidase',
        'glutathione-dependent peroxidase': 'peroxidase',
        'iodide-peroxidase': 'peroxidase',
        'lignin peroxidase': 'peroxidase',
        'myeloperoxidase': 'peroxidase',
        'nadh peroxidase': 'peroxidase',
        'pod': 'peroxidase',
        'pod-like': 'peroxidase',
        'hrp': 'peroxidase',
        'hrp-like': 'peroxidase',
        'guaiacol oxidase': 'peroxidase',
        'phenoloxidase': 'peroxidase',
        'polyphenoloxidase': 'peroxidase',
        'polyphenol oxidase': 'peroxidase',
        'glucose oxidase': 'peroxidase',
        'glucose oxidase-like': 'peroxidase',
    
        # Oxidase
        'oxidase': 'oxidase',
        'oxidase-like and peroxidase-like': 'oxidase',
        'ascorbic acid oxidase': 'oxidase',
        'ascorbate oxidase': 'oxidase',
        'cholesterol oxidase': 'oxidase',
        'diamine oxidase': 'oxidase',
        'ethanol oxidase': 'oxidase',
        'monooxidase': 'oxidase',
        'nadh oxidase': 'oxidase',
        'nadph oxidase': 'oxidase',
        'oxidative coupling of methane': 'oxidase',
        'oxidase, peroxidase, and catalase': 'oxidase',
        'oxalate oxidase': 'oxidase',
        'chromogen oxidase': 'oxidase',
        'cytochrome c oxidase': 'oxidase',
        'cco': 'oxidase',
        'cco-like': 'oxidase',
    
        # Glutathione S-transferase
        'glutathione s-transferase': 'glutathione s-transferase',
        'gst': 'glutathione s-transferase',
        'gsh-s-transferase': 'glutathione s-transferase',
        'gsta4-4': 'glutathione s-transferase',
    
        # Glutathione reductase
        'glutathione reductase': 'glutathione reductase',
        'gsh reductase': 'glutathione reductase',
        'glutathione transferase': 'glutathione reductase',
    
        # Misc common
        'rna splicing': 'rna splicing',
        'rnase': 'rnase',
        'dnase': 'dnase',
        'dnase i': 'dnase',
        'lipase': 'lipase',
        'protease': 'protease',
        'proteolytic': 'protease',
        'peptidase': 'protease',
        'amidase': 'amidase',
        'esterase': 'esterase',
        'photooxidase': 'oxidase',
        'reductase': 'reductase',
        'thioredoxin': 'thioredoxin',
        'thioredoxin reductase': 'thioredoxin reductase',
        'thioredoxin peroxidase': 'thioredoxin peroxidase',
        'thiol peroxidase': 'thioredoxin peroxidase',
        'thioredoxin-dependent thiol peroxidase': 'thioredoxin peroxidase',
    
        # Catch-all
        'unknown': 'unknown',
        'non-peroxidase': 'unknown',
        'hydrolysis': 'hydrolase',
        'hydrolase': 'hydrolase',
        'phosphatase': 'phosphatase',
        'acid phosphatase': 'phosphatase',
        'alkaline phosphatase': 'phosphatase',
        'alkaline phosphatase-like': 'phosphatase',
        'phosphotriesterase': 'phosphatase',
        'phosphotriesterase-like': 'phosphatase',
        'phospholipid hydroperoxide glutathione peroxidase': 'glutathione peroxidase',
        'phospholipid hydroperoxide cysteine peroxidase': 'glutathione peroxidase',
        'phosphodiester bond cleavage': 'phosphodiesterase',
        'phosphodiesterase': 'phosphodiesterase',
    }
    
    activity_map.update({
        # Extensions and Additions
    
        # Already Existing Classes
        'glutathione-peroxidase': 'glutathione peroxidase',
        'glutathione peroxidase-4': 'glutathione peroxidase',
        'phgpx': 'glutathione peroxidase',
        'gsh-dependent peroxidase': 'glutathione peroxidase',
        'gsh-peroxidase': 'glutathione peroxidase',
        'catalase-like and gsh peroxidase-like': 'glutathione peroxidase',
        'non-se-gpx': 'glutathione peroxidase',
        'glutathione peroxidase (gpx)-like': 'glutathione peroxidase',
        'non-selenium-dependent glutathione peroxidase': 'glutathione peroxidase',
    
        'peroxidase, oxidase': 'peroxidase',
        'peroxidase, catalase, superoxide dismutase': 'peroxidase',
        'catalase and peroxidase': 'peroxidase',
        'peroxidase and superoxide dismutase': 'peroxidase',
        'oxidase and peroxidase': 'oxidase',
        'peroxidase, catalase, oxidase': 'oxidase',
    
        'superoxide dismutase, catalase, glutathione peroxidase': 'superoxide dismutase',
        'superoxide dismutase and catalase': 'superoxide dismutase',
        'superoxide dismutase (sod) and catalase (cat)': 'superoxide dismutase',
    
        'cuzn-sod': 'superoxide dismutase',
        'mnp': 'peroxidase',  # manganese peroxidase (mnp)
    
        'gox-like': 'oxidase',  # glucose oxidase-like
    
        'thiocyanate oxidation': 'oxidase',
        'indole-3-acetic acid oxidase': 'oxidase',
        'indoleacetic acid oxidase': 'oxidase',
        'auxin oxidase': 'oxidase',
        'iaa oxidase': 'oxidase',
        'iaa-oxidase': 'oxidase',
        'pseudo-nad(p)h oxidase': 'oxidase',
        'nadh-oxidase': 'oxidase',
        'nadph peroxidase': 'peroxidase',
        'polyamine oxidase': 'oxidase',
        'cytochrome c-oxidase': 'oxidase',
        'prx': 'peroxidase',
        'lipid peroxidase': 'peroxidase',
    
        'photooxidase': 'oxidase',
        'photocatalytic': 'oxidase',
    
        'peroxynitrite scavenger': 'peroxidase',
        'peroxynitrite reductase': 'reductase',
    
        'chlorination': 'oxidase',
        'iodination': 'oxidase',
        'iodinating': 'oxidase',
        'haloperoxidase': 'oxidase',
    
        'co2 reduction': 'reductase',
        'co oxidation': 'oxidase',
        'reduction of aryl azides': 'reductase',
        'thioredoxin-dependent thiol peroxidase': 'thioredoxin peroxidase',
    
        'ethene-forming enzyme': 'oxidase',  # assuming 'ethylene-forming enzyme' activity
    
        # Hydrolytic enzymes
        'hydrolytic': 'hydrolase',
        'acetal hydrolysis': 'hydrolase',
        'cellulase': 'hydrolase',
        'invertase': 'hydrolase',
        'β-glucosidase': 'hydrolase',
        'β-glucuronidase': 'hydrolase',
        'amylase': 'hydrolase',
        'beta-amylase': 'hydrolase',
    
        # Dehydrogenases
        'dehydrogenase': 'dehydrogenase',
        'glucose dehydrogenase': 'dehydrogenase',
        'glucose-6-phosphate dehydrogenase': 'dehydrogenase',
        '11β-hydroxysteroid dehydrogenase': 'dehydrogenase',
    
        # Misc classes
        'aldh3': 'aldehyde dehydrogenase',
        'aldh1 plus aldh2': 'aldehyde dehydrogenase',
        'adenyl cyclase': 'cyclase',
        'atpase': 'atpase',
        'ribonuclease': 'rnase',
        'nuclease': 'rnase',
        'polymerase': 'polymerase',
        'raft polymerization': 'polymerase',
        'organophosphorus hydrolase': 'hydrolase',
        'chlorophyllase': 'hydrolase',
        'lipoxygenase': 'oxidase',
        'monooxygenase': 'oxidase',
        'violaxanthin de-epoxidase': 'oxidase',
        'p-450 complex': 'oxidase',  # Cytochrome P450
        'epoxidase': 'oxidase',
        'epoxidation': 'oxidase',
    
        # Radical scavenging and antioxidant classes
        'dpph radical scavenging': 'antioxidant',
        'hydroxyl radical scavenging': 'antioxidant',
        'nitric oxide scavenger': 'antioxidant',
        'singlet oxygen scavenger': 'antioxidant',
        'antioxidase': 'antioxidant',
        'total antioxidant capacity': 'antioxidant',
    
        # Proteases and related
        'chymotrypsin-like': 'protease',
        'collagenase': 'protease',
        'lysozyme': 'protease',
        'protease': 'protease',
    
        # Other hydrolases
        'amidase': 'amidase',
        'esterase': 'esterase',
        'phosphorylase': 'phosphorylase',
        'transphosphorylation': 'phosphorylase',
    
        # Miscellaneous
        'diene conjugates': 'oxidative stress marker',
        'malondialdehyde': 'oxidative stress marker',
        'ar': 'reductase',  # assuming aldose reductase
        'orr': 'oxidase',  # oxygen reduction reaction
        'thioredoxin': 'thioredoxin',
        'gr': 'glutathione reductase',
        'nadph-cytochrome c reductase': 'reductase',
        'cytochrome c reductase': 'reductase',
        'glutathione oxidase': 'oxidase',
        'c-gpx': 'glutathione peroxidase',
        'cgpx': 'glutathione peroxidase',
        'antimicrobial': 'antimicrobial',
        'po': 'peroxidase',
        'phenylalanine ammonia-lyase': 'lyase',
        'pal': 'lyase',
        'ppo': 'oxidase',  # polyphenol oxidase
        'polyphenolase': 'oxidase',
        'laccase': 'laccase',
        'syringaldazine oxidase': 'oxidase',
        'auxin oxidase': 'oxidase',
        'hydroxyl radical scavenger': 'antioxidant',
        'peroxide genesis': 'oxidase',
        'oph-like': 'hydrolase',  # organophosphorus hydrolase-like
        'uricase': 'oxidase',
    })
    
    activity_map.update({
        'superoxide peroxidase': 'peroxidase',
        'adenosine deaminase': 'deaminase',
        'hematin catalysis': 'catalysis',
        'indoleacetate oxidase': 'oxidase',
        'ethylene-forming enzyme': 'ethylene forming enzyme',
        'glutathione': 'glutathione',
        'butyrylcholinesterase': 'esterase',
        'olefin metathesis': 'metathesis',
        'quinol peroxidase': 'peroxidase',
        'nitroreductase': 'reductase',
        'catalase (cat), superoxide dismutase (sod), and glutathione peroxidase (gpx)': 'multiple antioxidant enzymes',
        'tryptophan peroxidase-oxidase': 'oxidase',
        'polygalacturonate trans-eliminase': 'lyase',
        'deiodinase': 'deiodinase',
        'coniferyl alcohol peroxidase': 'peroxidase',
        'paraoxonase': 'hydrolase',
        'dmop oxidation': 'oxidation',
        'cat-like': 'catalase',
        'mn(ii) oxidation': 'oxidation',
        'β-1,3 glucanase': 'glucanase',
        'monophenolase': 'oxidase',
        'tyrosinase': 'oxidase',
        'gamma-glutamyltransferase': 'transferase',
        'nadh-pod': 'peroxidase',
        'glutamine synthetase': 'synthetase',
        'chitinase': 'chitinase',
    })

    if check_nan(activity):
        return np.nan

    activity_types = {key.lower(): value for key, value in activity_map.items()}
    try:
        return activity_types[activity.lower()]
    except:
        print(activity)
        return np.nan


def unify_polymer(value, check_nan=check_nan):
    polymers_map = {
        # Polyvinylpyrrolidone (PVP)
        'poly(n-vinylpyrrolidone)': 'Poly(N-Vinylpyrrolidone)',
        'pvp': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl pyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'poly(vinylpyrrolidone)': 'Poly(N-Vinylpyrrolidone)',
        'poly(vinyl pyrrolidone)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl pyrrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (pvp k30, mw = 40000 da)': 'Poly(N-Vinylpyrrolidone)',
        'poly-(n-vinyl-2-pyrrolidone) (pvp∙k30)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (pvp-10)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinypyrrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'poly(vinyl pyrrolidone) (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (pvp, k30, mw=40000)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (pvp, mw 30,000)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl pyrrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'polyclar at': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpolypyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl polypyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl polyprolidone': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl poly-pyrolidone (pvpp)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpolypyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'poly(vinylpyrrolidone)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl pyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl pyrrolidone (pvp)': 'Poly(N-Vinylpyrrolidone)',
            # Polyvinylpyrrolidone (PVP)
        'polyvinylpyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'Polyvinylpyrrolidone': 'Poly(N-Vinylpyrrolidone)',
        'Poly(vinylpyrrolidone) (PVP)': 'Poly(N-Vinylpyrrolidone)',
        'Polyvinylpyrrolidone (PVP, K-30)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl pyrrolidone (PVP-K30)': 'Poly(N-Vinylpyrrolidone)',
        'PVP (K-30)': 'Poly(N-Vinylpyrrolidone)',
        'PVP-40': 'Poly(N-Vinylpyrrolidone)',
        'Poly (vinyl pyrrolidone) (PVP)': 'Poly(N-Vinylpyrrolidone)',
        'Poly(N-vinyl-2-pyrrolidone) (PVP)': 'Poly(N-Vinylpyrrolidone)',
        'Crosslinked polyvinylpyrrolidone (40%)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (PVP, K30)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpyrrolidone (PVP, K30, MW ≈ 3800)': 'Poly(N-Vinylpyrrolidone)',
        'PVP K30': 'Poly(N-Vinylpyrrolidone)',
        'polyvinyl-pyrrolidone (PVP-10)': 'Poly(N-Vinylpyrrolidone)',
        'polyvinylpolypyrrolidone (PVPP)': 'Poly(N-Vinylpyrrolidone)',
        'PVPP': 'Poly(N-Vinylpyrrolidone)',
    
        # Polyethylene glycol (PEG)
        'peg': 'poly(ethylene glycol)',
        'peg-200': 'poly(ethylene glycol)',
        'peg-400': 'poly(ethylene glycol)',
        'peg 4000': 'poly(ethylene glycol)',
        'peg 20000': 'poly(ethylene glycol)',
        'peg 20,000': 'poly(ethylene glycol)',
        'peg-4000': 'poly(ethylene glycol)',
        'polyethylene glycol 4000': 'poly(ethylene glycol)',
        'polyethylene glycol-400': 'poly(ethylene glycol)',
        'polyethylene glycol 20000': 'poly(ethylene glycol)',
        'polyethylene glycol (peg 1000)': 'poly(ethylene glycol)',
        'polyethylene glycol (peg)': 'poly(ethylene glycol)',
        'polyethylene glycol (peg-4000)': 'poly(ethylene glycol)',
        'polyethylene glycol 20,000': 'poly(ethylene glycol)',
        'polyethyleneglycol-8000': 'poly(ethylene glycol)',
        'polyethylene glycol (peg 4.6k)': 'poly(ethylene glycol)',
        'polyethyleneglycol 3550': 'poly(ethylene glycol)',
        'polyethylene glycol 2000': 'poly(ethylene glycol)',
        'poly(ethylene oxide)': 'poly(ethylene glycol)',
        'poly(ethylene glycol)': 'poly(ethylene glycol)',
        'poly(ethylene glycol) 4000': 'poly(ethylene glycol)',
        'peo20ppo70peo20': 'poly(ethylene glycol)',
        # Polyethylene glycol (PEG)
        'polyethylene glycol': 'poly(ethylene glycol)',
        'Polyethylene glycol': 'poly(ethylene glycol)',
        'PEG-3,4-dihydroxy benzyl amine': 'poly(ethylene glycol)',
        'PEG–vitamin E': 'poly(ethylene glycol)',
        'Poly(ethylene glycol) (PEG 4.6k)': 'poly(ethylene glycol)',
        'Polyethyleneglycol (PEG) 4000': 'poly(ethylene glycol)',
        'Polyethylene glycol 20,000 (PEG 20,000)': 'poly(ethylene glycol)',
        'poly eth yl ene gly -col(4000)': 'poly(ethylene glycol)',
        'polyethylene glycol dicarboxylic acid': 'poly(ethylene glycol)',
        'PEG 600': 'poly(ethylene glycol)',
        'PEG-SH': 'poly(ethylene glycol)',
        'methoxy PEG thiol': 'poly(ethylene glycol)',
        'PEG-b-PVIm': 'poly(ethylene glycol)',
        'PEG-pLL': 'poly(ethylene glycol)',
        'PEG-pLL 50': 'poly(ethylene glycol)',
        'poly(ethylene glycol)-b-poly(L-glutamic acid)': 'poly(ethylene glycol)',
        'poly(ethylene glycol dimethylacrylate)': 'poly(ethylene glycol)',
        'poly(ethyleneglycol)methylether': 'poly(ethylene glycol)',
        # Chitosan
        'chitosan': 'chitosan',
        'chitosan (cs)': 'chitosan',
    
        # Polydopamine (PDA)
        'polydopamine': 'poly(dopamine)',
        'poly(dopamine)': 'poly(dopamine)',
        'pda': 'poly(dopamine)',
        'pppda': 'poly(dopamine)',
    
        # Pluronic F127
        'pluronic f127': 'pluronic f127',
        'pluronic@f127': 'pluronic f127',
        'pluronic block copolymer f127': 'pluronic f127',
        'f127': 'pluronic f127',
        'f127 (eo106po70eo106)': 'pluronic f127',
    
        # Polyethyleneimine (PEI)
        'polyethyleneimine': 'poly(ethylene imine)',
        'polyethylenimine (pei)': 'poly(ethylene imine)',
        'poly(ethylenimine)': 'poly(ethylene imine)',
        'poly(ethylene imine)': 'poly(ethylene imine)',
        'bpei': 'poly(ethylene imine)',
        'pei': 'poly(ethylene imine)',
        'polyethylenimine (pei)': 'poly(ethylene imine)',
    
        # Poly(vinyl alcohol) (PVA)
        'pva': 'poly(vinyl alcohol)',
        'poly(vinyl alcohol)': 'poly(vinyl alcohol)',
        'polyvinyl alcohol (pva)': 'poly(vinyl alcohol)',
        'polyvinyl alcohol': 'poly(vinyl alcohol)',
    
        # Poly(N-isopropylacrylamide) (PNIPAM)
        'poly(n-isopropylacrylamide)': 'Poly(N-isopropylacrylamide)',
        'nipam': 'Poly(N-isopropylacrylamide)',
        'nipaam': 'Poly(N-isopropylacrylamide)',
        
        # Hyaluronic acid (HA)
        'hyaluronic acid': 'hyaluronic acid',
        'ha': 'hyaluronic acid',
    
        # Gelatin
        'gelatin': 'gelatin',
        'gelatin (type b)': 'gelatin',
    
        # Dextran
        'dextran': 'dextran',
    
        # CTAB
        'ctab': 'ctab',
        
        # DNA
        'dna': 'dna',
        'ssdna': 'dna',
        
        # Bovine Serum Albumin (BSA)
        'bsa': 'BSA',
        'bovine serum albumin': 'BSA',
        
        # Polyacrylamide
        'paam': 'poly(acryl amide)',
        'polyacrylamide': 'poly(acryl amide)',
    
        # Others
        'polyacrylic acid': 'poly(acrylic acid)',
        'poly(acrylic acid)': 'poly(acrylic acid)',
        'pss': 'poly(styrene sulfonate)',
        'pampps': 'poly(styrene sulfonate)',
        'pani': 'poly(aniline)',
        'pedot': 'poly(3,4-ethylenedioxythiophene)',
        'pmma': 'poly(methyl methacrylate)',
        'pan': 'poly(acrylonitrile)',
        'ppv': 'poly(p-phenylene vinylene)',
        'ps-hobt': 'poly(styrene sulfonate)',
        'pβ-cd': 'β-cyclodextrin',
    
        'polyaniline': 'poly(aniline)',
        'PPy': 'poly(aniline)',
    
        # Polystyrene derivatives
        'poly(styrenesulfate)': 'poly(styrene sulfonate)',
        'sodium polystyrene sulfonate': 'poly(styrene sulfonate)',
        'sulfonated polystyrene': 'poly(styrene sulfonate)',
        'poly(styrene sulfonate) and poly(diallyldimethylammonium chloride)': 'poly(styrene sulfonate)',
        'poly(styrene sulfonate, sodium salt)(PSS)/poly(diallyldimethylammonium) chloride (PDADMAC)': 'poly(styrene sulfonate)',
        'chloromethylated polystyrene': 'chloromethylated polystyrene',
    
        # Polyacrylonitrile
        'Polyacrylonitrile': 'Poly(acrylonitrile)',
    
        # Polydiallyldimethylammonium chloride (PDADMAC)
        'poly(diallyldimethylammonium chloride)': 'poly(diallyldimethylammonium chloride)',
        'Poly(diallyldimethylammonium chloride) (PDADMAC)': 'poly(diallyldimethylammonium chloride)',
    
        # Polyallylamine hydrochloride (PAH)
        'polyallylamine hydrochloride': 'poly(allylamine hydrochloride)',
        'poly(allylamine hydrochloride) (PAH)': 'poly(allylamine hydrochloride)',
        'Poly(allyl amine hydrochloride) (PAH)': 'poly(allylamine hydrochloride)',
        'poly(allyl amine hydrochloride)': 'poly(allylamine hydrochloride)',
    
        # Nafion
        'Nafion': 'Nafion',
        'nafion': 'Nafion',
    
        # Others
        'hyperbranched polyglycidol': 'Hyperbranched polyglycidol',
        'polypyrrole': 'Polypyrrole',
        'agarose': 'Agarose',
        'polyA': 'Polyadenine',
        'Pluronic P123': 'Pluronic P123',
        'PAMPS': 'Poly(2-acrylamido-2-methyl-1-propanesulfonic acid)',
        'Fmoc-Gly Alko-PEG resin': 'Fmoc-Gly Alko-PEG resin',
        'EDTA': 'Ethylenediaminetetraacetic acid',
        'CMC': 'Carboxymethyl cellulose',
        'Protamine': 'Protamine',
        'Polyoxometalates': 'Polyoxometalates',
        'D,L-lactide': 'D,L-lactide',
        'DSPE-mPEG 2K': 'DSPE-mPEG',
        'poly-L-lysine': 'Poly-L-lysine',
        'PLL-g-Dex': 'Poly-L-lysine-grafted dextran',
        'oleic acid': 'Oleic acid',
        'Sodium alginate': 'Sodium alginate',
        'carboxymethyl cellulose': 'Carboxymethyl cellulose',
        'APTES': '3-Aminopropyltriethoxysilane',
        'sodium lignosulfonate': 'Sodium lignosulfonate',
        'pS-PEG': 'Phosphorylated PEG',
        'Gum kondagogu': 'Gum kondagogu',
        'poly(vinyl chloride)': 'Poly(vinyl chloride)',
        'silk fibroin': 'Silk fibroin',
        'apoferritin': 'Apoferritin',
        'Hexadecyl trimethyl ammonium chloride': 'Hexadecyl trimethyl ammonium chloride',
        'polyethylenimine': 'Polyethylenimine',
        'Polyethyleneimine (PEI)': 'Polyethylenimine',
        'poly(ethyleneimine)': 'Polyethylenimine',
        'cotton fabric': 'Cotton fabric',
        'polyT20': 'Polythymine',
        'β-cyclodextrin': 'Beta-cyclodextrin',
        'SA': 'Stearic acid',
        'starch': 'Starch',
        'C 18-PEG': 'C18-PEG',
        'pVIII': 'pVIII',
        'β-casein': 'Beta-casein',
        'Poly (acrylic acid)': 'Poly(acrylic acid)',
        'poly(2-hydroxyethyl methacrylate) (PHEMA)': 'Poly(2-hydroxyethyl methacrylate)',
        'Heparin': 'Heparin',
        'poly(vinyl alcohol) (PVA)': 'Poly(vinyl alcohol)',
        'pepsin': 'Pepsin',
        'polymethacrylate': 'Polymethacrylate',
        'silica': 'Silica',
        'glutathione': 'Glutathione',
        'calcium alginate-pectin': 'Calcium alginate-pectin',
        'DSPE-PEG': 'DSPE-PEG',
        'Triblock copolymer poly(ethyleneoxide)–poly (propyleneoxide)–poly (ethyleneoxide) block copolymer ((EO) 20(PO) 70(EO) 20, Pluronic 123)': 'Pluronic 123',
        '1 H-3-Methylimidazolium acetate': 'Ionic liquid',
        'chitin': 'Chitin',
        'Triton-X': 'Triton-X',
        'Poly(ethylene oxide)–poly(propylene oxide)–poly(ethylene oxide)': 'Poly(ethylene oxide)-block-poly(propylene oxide)-block-poly(ethylene oxide)',
        'gum olibanum': 'Gum olibanum',
        'maltose-binding protein (MBP)': 'Maltose-binding protein',
        'Pericarpium Citri Reticulatae polysaccharide': 'Citrus polysaccharide',
        'Melamine (MA)': 'Melamine',
        'poly(acryloyl morpholine)': 'Poly(acryloyl morpholine)',
        'PEG-PAMA': 'PEG-PAMA',
        'poly(o-anisidine)': 'Poly(o-anisidine)',
        'poly(methyl methacrylate-co-glycidyl methacrylate)': 'Poly(methyl methacrylate-co-glycidyl methacrylate)',
        'N-methyl-2-pyrrolidinone': 'N-Methyl-2-pyrrolidone',
        'DETA': 'Diethylenetriamine',
        'P(BZMA-co-GMA)': 'Poly(benzyl methacrylate-co-glycidyl methacrylate)',
        'poly[(2-methacryloyloxyethyl)trimethyl ammonium chloride] (PMOTA)': 'Poly[(2-methacryloyloxyethyl)trimethyl ammonium chloride]',
        'LP': 'Lipopolysaccharide',
        'poly(N-vinylcaprolactam)': 'Poly(N-vinylcaprolactam)',
        'SIL-PEG-SIL': 'Silica-PEG-Silica',
        'β-CDP': 'Beta-cyclodextrin polymer',
        'oleylamine': 'Oleylamine',
        'gellan': 'Gellan gum',
        'cellulose': 'Cellulose',
        'triblock copolymer pluronic P123': 'Pluronic P123',
        'octyltrimethylammonium bromide': 'Octyltrimethylammonium bromide',
        'l-phenylalanine': 'L-Phenylalanine',
        'dodecyltrimethylammonium bromide': 'Dodecyltrimethylammonium bromide',
        'l-alanine': 'L-Alanine',
        'cholesterol': 'Cholesterol',
        'trisodium citrate': 'Trisodium citrate',
        'PDMS 90-b-PMOXA 10-COOH': 'PDMS-block-PMOXA',
        'sodium carboxymethyl cellulose': 'Sodium carboxymethyl cellulose',
        'poly(ethylene terephthalate)': 'Poly(ethylene terephthalate)',
        'PEG-b-poly(aspartate diethyltriamine)': 'PEG-block-poly(aspartate diethyltriamine)',
        'PDDA': 'Poly(diallyldimethylammonium chloride)',
        'acrylamide': 'Acrylamide',
        'sodium polyacrylate (PAA)': 'Sodium polyacrylate',
        'poly(vinylpyrrolidone) M_n 55,000': 'Polyvinylpyrrolidone',
        'oligo-N-isopropylacrylamide (ONIPAAm)': 'Oligo-N-isopropylacrylamide',
        'Tween 20': 'Tween 20',
        'Tween 80': 'Tween 80',
        'Carboxymethyl guar gum': 'Carboxymethyl guar gum',
        'polyamidoamine dendrimer': 'Polyamidoamine dendrimer',
        'PDMC': 'Poly(dimethyl diallyl ammonium chloride)',
        'Nylon': 'Nylon',
        'Phytagel': 'Phytagel',
        'Poly Bed 812': 'Poly Bed 812',
        'poly(N-isopropylacrylamide)-b-polyacrylamides': 'Poly(N-isopropylacrylamide)-block-polyacrylamides',
        'Pluronic F-127': 'Pluronic F-127',
        'o-phenylenediamine': 'O-Phenylenediamine',
        'ELP': 'Elastin-like polypeptide',
        # Poly(D,L-lactic-co-glycolic) acid (PLGA)
        'poly(D,L-lactic-co-glycolic) acid (PLGA)': 'Poly(D,L-lactic-co-glycolic acid) (PLGA)',
    
        # Poly(allyloxyethylcyanoacrylate) (P-AOECA)
        'Poly(allyloxyethylcyanoacrylate) (P-AOECA)': 'Poly(allyloxyethylcyanoacrylate)',
    
        # poly(N-isopropylacrylamide)-b-[polyacrylamides-co-poly(6-o-(triethylene glycol monoacrylate ether)-b-cyclodextrin)]
        'poly(N-isopropylacrylamide)-b-[polyacrylamides-co-poly(6-o-(triethylene glycol monoacrylate ether)-b-cyclodextrin)]': 'Poly(N-isopropylacrylamide)-block-[polyacrylamides-co-poly(cyclodextrin)]',
    
        # poly(MAA-co-EDMA)
        'poly(MAA-co-EDMA)': 'Poly(methacrylic acid-co-ethylene glycol dimethacrylate)',
    
        # Maltodextrin
        'Maltodextrin': 'Maltodextrin',
    
        # DOPA-PIMA-PEG
        'DOPA-PIMA-PEG': 'DOPA-PIMA-PEG',
    
        # poly(sodium-p-styrenesulfonate)
        'poly(sodium-p-styrenesulfonate)': 'Poly(sodium styrenesulfonate)',
    
        # Poly (N-vinyl-2-pyrrolidone) (PVP)
        'Poly (N-vinyl-2-pyrrolidone) (PVP)': 'Poly(N-vinyl-2-pyrrolidone)',
    
        # Poly (allyl amine hydrochloride) (PAH)
        'Poly (allyl amine hydrochloride) (PAH)': 'poly(allylamine hydrochloride)',
    
        # Triblock copolymer poly(ethyleneoxide)–poly (propyleneoxide)–poly (ethyleneoxide) block copolymer ((EO) 20(PO) 70(EO) 20, Pluronic 123
        'Triblock copolymer poly(ethyleneoxide)–poly (propyleneoxide)–poly (ethyleneoxide) block copolymer ((EO) 20(PO) 70(EO) 20, Pluronic 123': 'Pluronic P123',
    
        # PMNT
        'PMNT': 'PMNT',  # Specific polymer, no standardization needed unless more context is provided.
    
        # calcium alginate–pectin
        'calcium alginate–pectin': 'Calcium alginate-pectin',
    
        # poly(amidoamine)
        'poly(amidoamine)': 'Poly(amidoamine)',
    
        # poly(styrene/acrolein)
        'poly(styrene/acrolein)': 'Poly(styrene-co-acrolein)',
    
        # PAH
        'PAH': 'poly(allylamine hydrochloride)',
    
        # Poly[2-(3-thienyl)ethyloxy-4-butylsulfonate] sodium salt (PTEBS)
        'Poly[2-(3-thienyl)ethyloxy-4-butylsulfonate] sodium salt (PTEBS)': 'Poly(thienylethyloxy butylsulfonate)',
    
        # poly(3,4-ethylenedioxythiophene)
        'poly(3,4-ethylenedioxythiophene)': 'Poly(3,4-ethylenedioxythiophene)',
    
        # poly(allylamine hydrochloride)
        'poly(allylamine hydrochloride)': 'poly(allylamine hydrochloride)',
    
        # Chondroitin sulfate
        'Chondroitin sulfate': 'Chondroitin sulfate',
    
        # polyvinyl polypyrrolidone (PVPP)
        'polyvinyl polypyrrolidone (PVPP)': 'Polyvinylpolypyrrolidone',
    
        # Cationic cellulose nanofibrils
        'Cationic cellulose nanofibrils': 'Cationic cellulose nanofibrils',
    
        # PDI
        'PDI': 'Polydispersity index',  # Not a polymer, but a property of polymers.
    
        # Carboxymethyl cellulose (CMC)
        'Carboxymethyl cellulose (CMC)': 'Carboxymethyl cellulose',
    
        # polyadenine-containing diblock DNA
        'polyadenine-containing diblock DNA': 'Polyadenine-containing diblock DNA',
    
        # polyvinylpyrrolidone-25
        'polyvinylpyrrolidone-25': 'Polyvinylpyrrolidone',
    
        # ChEO 15
        'ChEO 15': 'ChEO 15',  # Specific polymer, no standardization needed unless more context is provided.
    
        # polytetrafluoroethylene
        'polytetrafluoroethylene': 'Poly(tetrafluoroethylene)',
    
        # polyethyleneimine-poly(ethylene glycol) (PEI-PEG)
        'polyethyleneimine-poly(ethylene glycol) (PEI-PEG)': 'Polyethyleneimine-poly(ethylene glycol)',
    
        # MDMO-PPV
        'MDMO-PPV': 'MDMO-PPV',  # Specific polymer, no standardization needed unless more context is provided.
    
        # Polyvinylidene fluoride (PVDF)
        'Polyvinylidene fluoride (PVDF)': 'Poly(vinylidene)fluoride',
    
        # glucose
        'glucose': 'Glucose',  # Not a polymer, but a monosaccharide.
    
        # polyvinylpoly-pyrolidone
        'polyvinylpoly-pyrolidone': 'polyvinylpoly-pyrolidone',
    }
    
    if check_nan(value):
        return np.nan

    polymers_map = {key.lower(): value for key, value in polymers_map.items()}
    try:
        return polymers_map[value.lower()]
    except:
        print(value)
        return np.nan


def unify_surfactant(value, chech_nan=check_nan):
    surfactant_map = {
        # Triton X-100
        'Triton X-100': 'Triton X-100',
        'TritonX-100 (TX-100)': 'Triton X-100',
        'triton X-100': 'Triton X-100',
        'Triton-X 100': 'Triton X-100',
        'Triton X-405': 'Triton X-405',
        'Triton CF-54': 'Triton CF-54',
    
        # CTAB and related compounds
        'CTAB': 'Cetyltrimethylammonium bromide',
        'Cetyltrimethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'cetyltrimethylammonium bromide': 'Cetyltrimethylammonium bromide',
        'Hexadecyl trimethyl ammonium bromide': 'Cetyltrimethylammonium bromide',
        'hexadecyltrimethylammonium bromide': 'Cetyltrimethylammonium bromide',
        'Cetyltrimethylammoniumbromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'Cetrimonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'Cetyltrimethyl ammonium bromide': 'Cetyltrimethylammonium bromide',
        'N-cetyltrimethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'Hexadecyltrimethylammonium bromide (HTAB)': 'Cetyltrimethylammonium bromide',
        'Cetyltrimethylammonium (CTAB)': 'Cetyltrimethylammonium bromide',
        'cetyl trimethyl ammonium bromide': 'Cetyltrimethylammonium bromide',
        'Cetylmethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'cetyltrimethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'hexadecyl trimethyl ammonium bromide': 'Cetyltrimethylammonium bromide',
        'hexadecyl trimethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'hexadecyl-trimethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'hexadecyltrimethylammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'Cetyl trimethyl ammonium bromide': 'Cetyltrimethylammonium bromide',
        'cetyltrimethylammoniumbromide': 'Cetyltrimethylammonium bromide',
    
        # Tween 20 and 80
        'Tween 20': 'Tween 20',
        'Tween-20': 'Tween 20',
        'polyoxyethylene sorbitolester (Tween), polyoxyethylene octylphenol (Triton X-100)': 'Tween 20',
        'Tween 80': 'Tween 80',
        'Tween-80': 'Tween 80',
    
        # SDS and related compounds
        'SDS': 'Sodium dodecyl sulfate',
        'sodium dodecyl sulfate': 'Sodium dodecyl sulfate',
        'Sodium dodecyl sulfate (SDS)': 'Sodium dodecyl sulfate',
        'Sodium n-dodecylsulphate (SDS) and Sodium bis(2-ethylhexyl)sulfosuccinate (AOT)': 'Sodium dodecyl sulfate',
        'Sodium dodecyl benzene sulfonate (SDBS)': 'Sodium dodecyl sulfate',
        'Sodium dodecylsulphate (SDS)': 'Sodium dodecyl sulfate',
        'Sodium dodecylbenzenesulfonate': 'Sodium dodecyl sulfate',
        'sodium dodecyl sulfate (SDS)': 'Sodium dodecyl sulfate',
        'sodium dodecylsulphate (SDS)': 'Sodium dodecyl sulfate',
        'SDBS': 'Sodium dodecyl sulfate',
        'SDS/DTAB': 'Sodium dodecyl sulfate / Dodecyltrimethylammonium bromide',
    
        # AOT and related compounds
        'AOT': 'Sodium bis(2-ethylhexyl)sulfosuccinate',
        'sodium bis(2-ethylhexyl)sulfosuccinate': 'Sodium bis(2-ethylhexyl)sulfosuccinate',
        'Aerosol OT': 'Sodium bis(2-ethylhexyl)sulfosuccinate',
        'dioctylsulfosuccinate sodium salt': 'Sodium bis(2-ethylhexyl)sulfosuccinate',
    
        # Pluronic F127 and P123
        'Pluronic F127': 'Pluronic F127',
        'Pluronic P123': 'Pluronic P123',
        'PEO20PPO70PEO20': 'Pluronic P123',
        'Triblock copolymer poly(ethyleneoxide)–poly (propyleneoxide)–poly (ethyleneoxide) block copolymer ((EO) 20(PO) 70(EO) 20, Pluronic 123': 'Pluronic P123',
    
        # Sodium citrate and related compounds
        'sodium citrate': 'Sodium citrate',
        'Sodium citrate': 'Sodium citrate',
        'trisodium citrate': 'Sodium citrate',
        'trisodium citrate dihydrate': 'Sodium citrate',
        'sodium citrate tribasic dehydrate': 'Sodium citrate',
        'Trisodium citrate': 'Sodium citrate',
        'sodium citrate dehydrate': 'Sodium citrate',
        'sodium tricitrate': 'Sodium citrate',
        'sodium citrate dihydrate': 'Sodium citrate',
        'citrate': 'Sodium citrate',  # Assuming generic "citrate" refers to sodium citrate
    
        # Ascorbic acid and related compounds
        'ascorbic acid': 'Ascorbic acid',
        'Ascorbic acid': 'Ascorbic acid',
        'L-ascorbic acid': 'Ascorbic acid',
        'L-Ascorbic acid': 'Ascorbic acid',
        'L-Ascorbic': 'Ascorbic acid',
        'ascorbic acid (AA)': 'Ascorbic acid',
        'L-Ascorbic acid (VC)': 'Ascorbic acid',
        'sodium ascorbate': 'Ascorbic acid',
        'Ascorbate': 'Ascorbic acid',
        'l-ascorbic acid': 'Ascorbic acid',
        'AA': 'Ascorbic acid',
    
        # Oleic acid and related compounds
        'oleic acid': 'Oleic acid',
        'Oleic acid': 'Oleic acid',
        'oleylamine': 'Oleylamine',
        'oleyl alcohol': 'Oleyl alcohol',
    
        # Other specific surfactants
        'didecyldimethylammonium bromide (DDAB)': 'Didecyldimethylammonium bromide',
        'Dodecyltrimethylammonium bromide (DTAB)': 'Dodecyltrimethylammonium bromide',
        'Tetradecyltrimethylammonium bromide (TTAB)': 'Tetradecyltrimethylammonium bromide',
        'Cetyltrimethylammonium chloride (CTAC)': 'Cetyltrimethylammonium chloride',
        'Cetyltrimethylammonium tosylate (CTATos)': 'Cetyltrimethylammonium tosylate',
        'Hexadecyl trimethyl ammonium chloride': 'Hexadecyl trimethyl ammonium chloride',
        'CTAC': 'Cetyltrimethylammonium chloride',
    
        # Non-surfactant additives
        '0.1% teepol': np.nan,
        '1-Butyl-3-methylimidazolium hexafluorophosphate (BMIPF6)': np.nan,
        '1-H-3-Methylimidazolium formate': np.nan,
        '2-methyl-2,4-pentanediol': np.nan,
        'C 12-LAS': np.nan,
        'C6H5Na3O7·2H2O': np.nan,
        'ChEO 15': np.nan,
        'Emasol 1130': np.nan,
        'N-cetyltrimethylammonium bromide': 'Cetyltrimethylammonium bromide',
        'PVP': np.nan,
        'Polyclar AT': np.nan,
        'Polyethylene glycol': np.nan,
        'Polyvinylpyrrolidone': np.nan,
        'Sandoclean PC': np.nan,
        'TBAB': np.nan,
        '[Cho][AOT]': 'Choline dioctylsulfosuccinate',
        'citrate': 'Sodium citrate',
        'cysteamine': np.nan,
        'ethylene glycol': np.nan,
        'sodium 2-mercaptoethanesulfonate (MPS)': np.nan,
        'sodium cholate': np.nan,
        'sodium n-tetradecanesulfonate': np.nan,
        'surfactant': np.nan,
        'triethylamine': np.nan,
    
        ###
        '1-octanol': np.nan,  # Not a surfactant
        'APTES': np.nan,  # Not a surfactant
        'BCL': np.nan,  # Not a surfactant
        'C12H25N(CH3)2–C12H24–N(CH3)2C12H25Br2': np.nan,  # Not a surfactant
        'C12H25N(COC(CH3)=CH2)-(C2H40)Is-H': np.nan,  # Not a surfactant
        'C18H37NH(C2H40)10-H': np.nan,  # Not a surfactant
        'C18H37NH(C2H40)15-H': np.nan,  # Not a surfactant
        'CHAPS': 'CHAPS',  # Zwitterionic surfactant
        'Cetrimonium bromide': 'Cetyltrimethylammonium bromide',
        'Cetyltrimethyl ammonium bromide (CTAB)': 'Cetyltrimethylammonium bromide',
        'Cetyltrimethylammonium bromide': 'Cetyltrimethylammonium bromide',
        'CnTAFB (n = 12, 14, 16)': np.nan,  # Not a surfactant
        'Ethylene glycol': np.nan,  # Not a surfactant
        'Hexadecyltrimethylammonium bromide': 'Cetyltrimethylammonium bromide',
        'Hydrogen peroxide (H2O2)': np.nan,  # Not a surfactant
        'KBr': np.nan,  # Not a surfactant
        'Lauric acid': np.nan,  # Not a surfactant
        'Lauryl maltoside (LM)': 'Lauryl maltoside',
        'N-decyl-heptaethyleneglycol mono-ether (C10E7)': np.nan,  # Not a surfactant
        'PEG 20000': np.nan,  # Not a surfactant
        'PEG-400': np.nan,  # Not a surfactant
        'Potassium bromide (KBr)': np.nan,  # Not a surfactant
        'Sodium dodecyl sulfate': 'Sodium dodecyl sulfate',
        'Span 80': 'Span 80',
        'Span 85': 'Span 85',
        'TOCL': np.nan,  # Not a surfactant
        'Tyrosine': np.nan,  # Not a surfactant
        'butanol': np.nan,  # Not a surfactant
        'choline dioctylsulfosuccinate ([Cho][AOT])': 'Choline dioctylsulfosuccinate',
        'citric acid': np.nan,  # Not a surfactant
        'dioleyl N-d-glucono-l-glutamate': np.nan,  # Not a surfactant
        'dioleyl glutamate ribitol amide': np.nan,  # Not a surfactant
        'ethanol': np.nan,  # Not a surfactant
        'lecithin': 'Lecithin',  # Surfactant
        'monoethanolamide (MEA)': 'Monoethanolamide',
        'nonanoic acid': np.nan,  # Not a surfactant
        'octyl glucoside': 'Octyl glucoside',
        'poly(N-Vinylpyrrolidone)': np.nan,  # Not a surfactant
        'poly(ethylene glycol) methyl ether M_n 5,000': np.nan,  # Not a surfactant
        'polyethyleneglycol': np.nan,  # Not a surfactant
        'polyoxyethylene sorbitan monolaurate (Tween 20)': 'Tween 20',
        'polyvinyl pyrrolidone (PVP)': np.nan,  # Not a surfactant
        'potassium bromide': np.nan,  # Not a surfactant
        'propanol-2': np.nan,  # Not a surfactant
        'sodium dodecyl benzene sulfonate (SDBS)': 'Sodium dodecyl sulfate',
        'sodium lignosulfonate': np.nan,  # Not a surfactant
        'sodium oleate': 'Sodium oleate',
        'trioctylphosphine': np.nan,  # Not a surfactant
    }

    if check_nan(value):
        return np.nan

    surfactant_map = {key.lower(): value for key, value in surfactant_map.items()}
    try:
        return surfactant_map[value.lower()]
    except:
        print(value)
        return np.nan

def map_activity(value):
    activity_map = {
        'peroxidase': 1,
        'glutathione peroxidase': 1,
        'thioredoxin peroxidase': 1,
        
        'oxidase': 2,
        
        'catalase': 3,
        'catalysis': 3,
        
        'laccase': 4,
        'unknown': np.nan,
        'superoxide dismutase': 5,
        'phosphatase': 6,
        'hydrolase': 7,
        'glutathione s-transferase': 8,
        'esterase': 9,
        
        'glutathione reductase': 10,
        'thioredoxin reductase': 10,
        'reductase': 10,
        
        'protease': 11,
        'antioxidant': 12,
        'rna splicing': 13,
        
        'dehydrogenase': 14,
        'aldehyde dehydrogenase': 14,
        
        'oxidation': 15,
        'lipase': 16,
        'phosphorylase': 17,
        'dnase': 18,
        'phosphodiesterase': 19,
        'rnase': 20,
        'lyase': 21,
        'multiple antioxidant enzymes': 22,
        'antimicrobial': 23,
        'synthetase': 24,
        'chitinase': 25,
        'oxidative stress marker': 26,
        'amidase': 27,
        'polymerase': 28,
        'transferase': 29,
        'ethylene forming enzyme': 30,
        'metathesis': 31,
        'atpase': 32,
        'deiodinase': 33,
        'glutathione': 34,
        'deaminase': 35,
        'cyclase': 36,
        'thioredoxin': 37,
        'glucanase': 38
    }

    if check_nan(value):
        return np.nan

    activity_map = {key.lower(): value for key, value in activity_map.items()}
    try:
        return activity_map[value.lower()]
    except:
        print(value)
        return np.nan

