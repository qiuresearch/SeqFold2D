#%%
from errno import ENETRESET
import logging
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
logger = logging.getLogger(__name__)

# homebrew
import misc

# 1) File names
def pdbinfo2filename(pdbinfo, modelnum=None, chainid=None) :
    ''' Convert PDB information to file name, minimum modelnum: 1 '''
    info_parts = pdbinfo.replace(':', '_').replace('+', '_').replace('|', '_').split('_')
    pdbid = info_parts[0]

    if len(info_parts) == 2: # only two parts: chainid > modelnum
        if chainid is None: 
            chainid = info_parts[1]
        elif modelnum is None and str.isnumeric(info_parts[1]):
            modelnum = int(info_parts[1])
    elif len(info_parts) > 2:
        # support multiple chain entries such as "PDBN:1:A+PDBN:1:B"
        # though assuming the same pdbid and model number
        if chainid is None:
            chainid = info_parts[2::3]
        if modelnum is None and str.isnumeric(info_parts[1]):
            modelnum = int(info_parts[1])

    model_str = ''
    if (modelnum != None) and (modelnum != '') :
        model_str = '_{}'.format(modelnum)

    chain_str = ''
    if isinstance(chainid, str) :
        chain_str = '_' + chainid
    elif isinstance(chainid, list) :
        chain_str = '_' + '-'.join(chainid)

    return pdbid + model_str + chain_str

def filename2pdbinfo(filename, sep1='_', sep2='-') :
    """ return pdbid, [[modelnum], and chainids] based on filename """
    file_parts = filename.split(sep1)
    pdbid = file_parts[0]; modelnum=None; chainids=[]
    if len(file_parts) == 2 :
        chainids = file_parts[1].split(sep2)
    elif len(file_parts) >= 3 :
        modelnum = int(file_parts[1])
        chainids = file_parts[2].split(sep2)
        if len(file_parts) > 3 :
            logger.warning("Only the first 3 parts in {} were used!!!".format(filename))

    return [pdbid, modelnum, chainids]

# 2) Chain IDs
ChainIdList = [
            'A','B','C','D','E','F','G','H','I','J','K','L','M',
            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            '0','1','2','3','4','5','6','7','8','9',
            'a','b','c','d','e','f','g','h','i','j','k','l','m',
            'n','o','p','q','r','s','t','u','v','w','x','y','z',
            ]  #,'!','"','#',
            #    '$','%','&',"'",'(',')','+',',','-','.','/',':',';',
            #    '<','=','>','?','@','[',']','^','_','`','{','|','}',
            #    '~']      
#
def get_unique_chainid(chainids, used_ids=[]) :
    """ Get unique chain ids to address conflicts or long chain ids in CIF files """
    # turn into single letters first by taking the last letter (arbitrary choice)
    if isinstance(chainids, str) :
        chainids = [chainids]
    chainids = [id[-1] for id in chainids]
    used_ids = set(used_ids)

    chainids_new = list(chainids)
    if len(chainids) + len(used_ids) > len(ChainIdList) :
        logger.warning('ERROR: the number of chains exceeds the number of allowable single letters')
        return []

    useable_ids = [id for id in ChainIdList if id not in used_ids]
    for i, id in enumerate(chainids) :
        if id in used_ids :
            chainids_new[i] = useable_ids[0]
            useable_ids = useable_ids[1:]
            used_ids = used_ids | {chainids_new[i]}
        else :
            if id in useable_ids :
                useable_ids.remove(id)
            used_ids = used_ids | {id}
    return chainids_new

# 3) Residue names
AA_ResName1to3Dict = {
    'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN',
    'E':'GLU', 'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS',
    'M':'MET', 'F':'PHE', 'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP',
    'Y':'TYR', 'V':'VAL', 'O':'PYL', 'Z':'GLX', 'B':'ASX', 'U':'SEC',
    'X':'UNK', 'J':'UNK',  # UNKNOWN
}
AA_ResName3to1Dict = {v: k for k, v in AA_ResName1to3Dict.items()}
try:
    from Bio.Data.SCOPData import protein_letters_3to1 as scop_3to1
    from Bio.Data.IUPACData import protein_letters_3to1_extended as iupac_3to1_ext
    AA_ResName3to1Dict.update(iupac_3to1_ext)
    AA_ResName3to1Dict.update(scop_3to1)
except:
    logger.debug(f'Unable to update AA_ResName3to1Dict by biophython dictionaries!')
    
AA_StdResNameList = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_ResNameList = AA_StdResNameList + list(AA_ResName3to1Dict.values())
AA_ResNameList = list(set(AA_ResNameList))
AA_ResNames = ''.join(AA_ResNameList)

AA_StdResName3List = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                   'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AA_ResName3List = AA_StdResName3List + list(AA_ResName3to1Dict.keys())
AA_ResName3List = list(set(AA_ResName3List))

# code_MSE= {'MSE':'M'}
# code_with_modified_residues = {
#     'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M', 'ILE':'I',
#     'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K', 'ARG':'R', 'SER':'S',
#     'THR':'T', 'TYR':'Y', 'HIS':'H', 'CYS':'C', 'ASN':'N', 'GLN':'Q',
#     'TRP':'W', 'GLY':'G',                                  'MSE':'M',
#     '2AS':'D', '3AH':'H', '5HP':'E', 'ACL':'R', 'AIB':'A', 'ALM':'A', 
#     'ALO':'T', 'ALY':'K', 'ARM':'R', 'ASA':'D', 'ASB':'D', 'ASK':'D', 
#     'ASL':'D', 'ASQ':'D', 'AYA':'A', 'BCS':'C', 'BHD':'D', 'BMT':'T', 
#     'BNN':'A', 'BUC':'C', 'BUG':'L', 'C5C':'C', 'C6C':'C', 'CCS':'C', 
#     'CEA':'C', 'CHG':'A', 'CLE':'L', 'CME':'C', 'CSD':'A', 'CSO':'C', 
#     'CSP':'C', 'CSS':'C', 'CSW':'C', 'CXM':'M', 'CY1':'C', 'CY3':'C', 
#     'CYG':'C', 'CYM':'C', 'CYQ':'C', 'DAH':'F', 'DAL':'A', 'DAR':'R', 
#     'DAS':'D', 'DCY':'C', 'DGL':'E', 'DGN':'Q', 'DHA':'A', 'DHI':'H', 
#     'DIL':'I', 'DIV':'V', 'DLE':'L', 'DLY':'K', 'DNP':'A', 'DPN':'F', 
#     'DPR':'P', 'DSN':'S', 'DSP':'D', 'DTH':'T', 'DTR':'W', 'DTY':'Y', 
#     'DVA':'V', 'EFC':'C', 'FLA':'A', 'FME':'M', 'GGL':'E', 'GLZ':'G', 
#     'GMA':'E', 'GSC':'G', 'HAC':'A', 'HAR':'R', 'HIC':'H', 'HIP':'H', 
#     'HMR':'R', 'HPQ':'F', 'HSD':'H', 'HSE':'H', 'HSP':'H', 'HTR':'W', 
#     'HYP':'P', 'IIL':'I', 'IYR':'Y', 'KCX':'K', 'LLY':'K', 'LTR':'W',
#     'LYM':'K', 'LYZ':'K', 'MAA':'A', 'MEN':'N', 'MHS':'H', 'MIS':'S',
#     'MLE':'L', 'MPQ':'G', 'MSA':'G', 'MVA':'V', 'NEM':'H', 'NEP':'H', 
#     'NLE':'L', 'NLN':'L', 'NLP':'L', 'NMC':'G', 'OAS':'S', 'OCS':'C', 
#     'OMT':'M', 'PAQ':'Y', 'PCA':'E', 'PEC':'C', 'PHI':'F', 'PHL':'F', 
#     'PR3':'C', 'PRR':'A', 'PTR':'Y', 'SAC':'S', 'SAR':'G', 'SCH':'C', 
#     'SCS':'C', 'SCY':'C', 'SEL':'S', 'SEP':'S', 'SET':'S', 'SHC':'C', 
#     'SHR':'K', 'SOC':'C', 'STY':'Y', 'SVA':'S', 'TIH':'A', 'TPL':'W', 
#     'TPO':'T', 'TPQ':'A', 'TRG':'K', 'TRO':'W', 'TYB':'Y', 'TYQ':'Y', 
#     'TYS':'Y', 'TYY':'Y', 'AGM':'R', 'GL3':'G', 'SMC':'C', 'CGU':'E',
#     'CSX':'C',
#     'SEC':'U', 'PYL':'O', 'ASX':'B', 'GLX':'Z', 'LLP':'X', 'UNK':'X',
#     ' DA':'a', ' DC':'c', ' DG':'g', ' DT':'t', ' DU':'u',
#     '  A':'a', '  C':'c', '  G':'g', '  T':'t', '  U':'u',
#     }

AA_DataDir = Path(__file__).parent / 'data'
try:
    AA_AtchleyFactor = pd.read_csv(AA_DataDir / 'aa_atchley.txt', sep='\t', index_col='aa').astype(float)
    AA_KideraFactor = pd.read_csv(AA_DataDir / 'aa_kidera.txt', sep='\t', index_col='aa').astype(float)
    AA_PropertyTable = pd.read_csv(AA_DataDir / 'aa_property_table.csv', index_col='aa').astype(float)
    AA_PropertyTable.drop(columns=['count'], inplace=True)
except:
    logger.debug(f'No amino acids property tables found!')
    
# AA_Atchley_Factor.index = AA_Atchley_Factor['aa']
# AA_Atchley_Factor.drop(['aa'], axis=1)

DNA_ResNames = 'ATGC'
DNA_ResNameList = list(DNA_ResNames)
DNA_ResName2List = ['DA', 'DT', 'DG', 'DC']
RNA_ResNames = 'AUGC'
RNA_ResNameList = list(RNA_ResNames)
NA_ResNames = 'ATUGC'
NA_ResNameList = DNA_ResName2List + RNA_ResNameList

def get_res2DNA_trans_dict(return_fn=False):
    if return_fn:
        # Res2DNA_Trans_Dict = misc.DefaultDict(lambda x: lambda: random.choice('ATGC'))
        Res2DNA_Trans_Dict = defaultdict(lambda: lambda: random.choice('ATGC'))
        Res2DNA_Trans_Dict.update(
            A=lambda: 'A',
            T=lambda: 'T',
            C=lambda: 'C',
            G=lambda: 'G',
            U=lambda: 'T',
            I=lambda: 'A',
            R=lambda: random.choice('GA'), # puRine
            Y=lambda: random.choice('CT'), # pYrimidine
            K=lambda: random.choice('GT'), # Ketone
            M=lambda: random.choice('AC'), # Amine
            S=lambda: random.choice('GC'), # Strong
            W=lambda: random.choice('AT'), # Weak
            B=lambda: random.choice('GCT'), # not A
            D=lambda: random.choice('AGT'), # not C
            H=lambda: random.choice('ACT'), # not G
            V=lambda: random.choice('AGC'), # not U
            X=lambda: random.choice('ATGC'),
            N=lambda: random.choice('ATGC'),
        )
    else:
        Res2DNA_Trans_Dict = defaultdict(lambda: random.choice('ATGC'))
        Res2DNA_Trans_Dict.update(
            A='A',
            T='T',
            C='C',
            G='G',            
            U='T',
            I='A',
            R=random.choice('GA'), # puRine
            Y=random.choice('CT'), # pYrimidine
            K=random.choice('GT'), # Ketone
            M=random.choice('AC'), # Amine
            S=random.choice('GC'), # Strong
            W=random.choice('AT'), # Weak
            B=random.choice('GCT'), # not A
            D=random.choice('AGT'), # not C
            H=random.choice('ACT'), # not G
            V=random.choice('AGC'), # not U
            X=random.choice('ATGC'),
            N=random.choice('ATGC'),
        )
    return Res2DNA_Trans_Dict


def get_res2RNA_trans_dict(return_fn=False):
    """ if return_fn, each value is a function that returns the translated RNA residue """
    if return_fn:
        # Res2RNA_Trans_Dict = misc.DefaultDict(lambda x: lambda: x)
        Res2RNA_Trans_Dict = defaultdict(lambda: lambda: random.choice('AUCG'))
        Res2RNA_Trans_Dict.update(
            A=lambda: 'A',
            U=lambda: 'U',
            C=lambda: 'C',
            G=lambda: 'G',            
            T=lambda: 'U',
            I=lambda: 'A',            
            R=lambda: random.choice('GA'), # puRine
            Y=lambda: random.choice('CU'), # pYrimidine
            K=lambda: random.choice('GU'), # Ketone
            M=lambda: random.choice('AC'), # Amine
            S=lambda: random.choice('GC'), # Strong
            W=lambda: random.choice('AU'), # Weak
            B=lambda: random.choice('GCU'), # not A
            D=lambda: random.choice('AGU'), # not C
            H=lambda: random.choice('ACU'), # not G
            V=lambda: random.choice('AGC'), # not U
            X=lambda: random.choice('AUGC'),
            N=lambda: random.choice('AUGC'),
        )
    else:
        Res2RNA_Trans_Dict = defaultdict(lambda: random.choice('AUCG'))
        Res2RNA_Trans_Dict = dict(
            A='A',
            U='U',
            C='C',
            G='G',               
            T='U',
            I='A',
            R=random.choice('GA'), # puRine
            Y=random.choice('CU'), # pYrimidine
            K=random.choice('GU'), # Ketone
            M=random.choice('AC'), # Amine
            S=random.choice('GC'), # Strong
            W=random.choice('AU'), # Weak
            B=random.choice('GCU'), # not A
            D=random.choice('AGU'), # not C
            H=random.choice('ACU'), # not G
            V=random.choice('AGC'), # not U
            X=random.choice('AUGC'),
            N=random.choice('AUGC'),
        )

    return Res2RNA_Trans_Dict

# NCO: Cobalt Hexammine ;
# TODO: FMN:
Ligand_ResNamesList = ['HOH', 'NCO', 'IRI', 'SPD', 'SPM', 'MG', 'NA', 'CS', 'PB', 'OS', 'TB',
                'CA', 'K', 'SR', 'BA', 'ZN', 'MN', 'CO', 'NI', 'BR', 'NH4', 'CL', 'PEG',
                'IOD', 'CD', 'F', 'GOL', 'TL', 'GAI', 'TAC',
                'EDO', 'ACT', 'TRS', 'SO4', 'MPD', 'FME', 'ACE', 'PG4', 'FMN', 'PAR', 'NMY',
                'ACY', 'PUT', 'OHX', 'PO4', 'ARG', 'LYS', 'ALA', 'EOH', 'IPA', 'AMZ', 'FMT']

# PDB atom lines for standard RNA residues (taken from DSSR by Xiang-Jun Lu)
RNA_AtomLines={
    'A': [
            "ATOM      1  P     A A   1       1.403   9.010   0.544" + " "*26 + "\n",
            "ATOM      2  O1P   A A   1       1.318  10.246   1.353" + " "*26 + "\n",
            "ATOM      3  O2P   A A   1       2.600   8.168   0.767" + " "*26 + "\n",
            "ATOM      4  O5'   A A   1       0.083   8.133   0.763" + " "*26 + "\n",
            "ATOM      5  C5'   A A   1      -1.187   8.665   0.338" + " "*26 + "\n",
            "ATOM      6  C4'   A A   1      -2.273   7.624   0.522" + " "*26 + "\n",
            "ATOM      7  O4'   A A   1      -2.138   6.620  -0.525" + " "*26 + "\n",
            "ATOM      8  C3'   A A   1      -2.208   6.817   1.819" + " "*26 + "\n",
            "ATOM      9  O3'   A A   1      -2.728   7.547   2.921" + " "*26 + "\n",
            "ATOM     10  C2'   A A   1      -3.044   5.610   1.392" + " "*26 + "\n",
            "ATOM     11  O2'   A A   1      -4.451   5.728   1.323" + " "*26 + "\n",
            "ATOM     12  C1'   A A   1      -2.479   5.346  -0.000" + " "*26 + "\n",
            "ATOM     13  N9    A A   1      -1.291   4.498  -0.000" + " "*26 + "\n",
            "ATOM     14  C8    A A   1       0.024   4.897   0.000" + " "*26 + "\n",
            "ATOM     15  N7    A A   1       0.877   3.902   0.000" + " "*26 + "\n",
            "ATOM     16  C5    A A   1       0.071   2.771   0.000" + " "*26 + "\n",
            "ATOM     17  C6    A A   1       0.369   1.398   0.000" + " "*26 + "\n",
            "ATOM     18  N6    A A   1       1.611   0.909   0.000" + " "*26 + "\n",
            "ATOM     19  N1    A A   1      -0.668   0.532  -0.000" + " "*26 + "\n",
            "ATOM     20  C2    A A   1      -1.912   1.023   0.000" + " "*26 + "\n",
            "ATOM     21  N3    A A   1      -2.320   2.290   0.000" + " "*26 + "\n",
            "ATOM     22  C4    A A   1      -1.267   3.124   0.000" + " "*26 + "\n",
    ],
    'C': [
            "ATOM      1  P     C A   1       1.424   9.045   0.545" + " "*26 + "\n",
            "ATOM      2  O1P   C A   1       1.345  10.281   1.354" + " "*26 + "\n",
            "ATOM      3  O2P   C A   1       2.617   8.197   0.768" + " "*26 + "\n",
            "ATOM      4  O5'   C A   1       0.099   8.175   0.763" + " "*26 + "\n",
            "ATOM      5  C5'   C A   1      -1.167   8.714   0.338" + " "*26 + "\n",
            "ATOM      6  C4'   C A   1      -2.259   7.678   0.523" + " "*26 + "\n",
            "ATOM      7  O4'   C A   1      -2.130   6.673  -0.525" + " "*26 + "\n",
            "ATOM      8  C3'   C A   1      -2.199   6.870   1.819" + " "*26 + "\n",
            "ATOM      9  O3'   C A   1      -2.715   7.603   2.922" + " "*26 + "\n",
            "ATOM     10  C2'   C A   1      -3.040   5.668   1.392" + " "*26 + "\n",
            "ATOM     11  O2'   C A   1      -4.447   5.793   1.324" + " "*26 + "\n",
            "ATOM     12  C1'   C A   1      -2.477   5.402  -0.000" + " "*26 + "\n",
            "ATOM     13  N1    C A   1      -1.285   4.542  -0.000" + " "*26 + "\n",
            "ATOM     14  C2    C A   1      -1.472   3.158   0.000" + " "*26 + "\n",
            "ATOM     15  O2    C A   1      -2.628   2.709   0.001" + " "*26 + "\n",
            "ATOM     16  N3    C A   1      -0.391   2.344  -0.000" + " "*26 + "\n",
            "ATOM     17  C4    C A   1       0.837   2.868  -0.000" + " "*26 + "\n",
            "ATOM     18  N4    C A   1       1.875   2.027   0.001" + " "*26 + "\n",
            "ATOM     19  C5    C A   1       1.056   4.275   0.000" + " "*26 + "\n",
            "ATOM     20  C6    C A   1      -0.023   5.068  -0.000" + " "*26 + "\n",
    ],
    'G': [
            "ATOM      1  P     G A   1       1.405   9.062   0.546" + " "*26 + "\n",
            "ATOM      2  O1P   G A   1       1.320  10.298   1.356" + " "*26 + "\n",
            "ATOM      3  O2P   G A   1       2.602   8.221   0.769" + " "*26 + "\n",
            "ATOM      4  O5'   G A   1       0.084   8.186   0.764" + " "*26 + "\n",
            "ATOM      5  C5'   G A   1      -1.185   8.718   0.339" + " "*26 + "\n",
            "ATOM      6  C4'   G A   1      -2.271   7.676   0.523" + " "*26 + "\n",
            "ATOM      7  O4'   G A   1      -2.136   6.672  -0.525" + " "*26 + "\n",
            "ATOM      8  C3'   G A   1      -2.206   6.869   1.819" + " "*26 + "\n",
            "ATOM      9  O3'   G A   1      -2.727   7.599   2.922" + " "*26 + "\n",
            "ATOM     10  C2'   G A   1      -3.042   5.663   1.392" + " "*26 + "\n",
            "ATOM     11  O2'   G A   1      -4.449   5.780   1.323" + " "*26 + "\n",
            "ATOM     12  C1'   G A   1      -2.477   5.399  -0.000" + " "*26 + "\n",
            "ATOM     13  N9    G A   1      -1.289   4.551  -0.000" + " "*26 + "\n",
            "ATOM     14  C8    G A   1       0.023   4.962   0.000" + " "*26 + "\n",
            "ATOM     15  N7    G A   1       0.870   3.969  -0.000" + " "*26 + "\n",
            "ATOM     16  C5    G A   1       0.071   2.833   0.000" + " "*26 + "\n",
            "ATOM     17  C6    G A   1       0.424   1.460   0.000" + " "*26 + "\n",
            "ATOM     18  O6    G A   1       1.554   0.955  -0.000" + " "*26 + "\n",
            "ATOM     19  N1    G A   1      -0.700   0.641  -0.000" + " "*26 + "\n",
            "ATOM     20  C2    G A   1      -1.999   1.087  -0.000" + " "*26 + "\n",
            "ATOM     21  N2    G A   1      -2.949   0.139  -0.001" + " "*26 + "\n",
            "ATOM     22  N3    G A   1      -2.342   2.364   0.001" + " "*26 + "\n",
            "ATOM     23  C4    G A   1      -1.265   3.177  -0.000" + " "*26 + "\n",
    ],
    'T': [
            "ATOM      1  P     T A   1       1.401   9.017   0.547" + " "*26 + "\n",
            "ATOM      2  O1P   T A   1       1.316  10.253   1.357" + " "*26 + "\n",
            "ATOM      3  O2P   T A   1       2.598   8.176   0.770" + " "*26 + "\n",
            "ATOM      4  O5'   T A   1       0.081   8.141   0.765" + " "*26 + "\n",
            "ATOM      5  C5'   T A   1      -1.189   8.673   0.340" + " "*26 + "\n",
            "ATOM      6  C4'   T A   1      -2.275   7.631   0.524" + " "*26 + "\n",
            "ATOM      7  O4'   T A   1      -2.140   6.627  -0.524" + " "*26 + "\n",
            "ATOM      8  C3'   T A   1      -2.210   6.824   1.820" + " "*26 + "\n",
            "ATOM      9  O3'   T A   1      -2.731   7.554   2.923" + " "*26 + "\n",
            "ATOM     10  C2'   T A   1      -3.046   5.618   1.392" + " "*26 + "\n",
            "ATOM     11  O2'   T A   1      -4.453   5.735   1.324" + " "*26 + "\n",
            "ATOM     12  C1'   T A   1      -2.481   5.354   0.000" + " "*26 + "\n",
            "ATOM     13  N1    T A   1      -1.284   4.500   0.000" + " "*26 + "\n",
            "ATOM     14  C2    T A   1      -1.462   3.135   0.000" + " "*26 + "\n",
            "ATOM     15  O2    T A   1      -2.562   2.608  -0.000" + " "*26 + "\n",
            "ATOM     16  N3    T A   1      -0.298   2.407  -0.000" + " "*26 + "\n",
            "ATOM     17  C4    T A   1       0.994   2.897  -0.000" + " "*26 + "\n",
            "ATOM     18  O4    T A   1       1.944   2.119  -0.000" + " "*26 + "\n",
            "ATOM     19  C5    T A   1       1.106   4.338   0.000" + " "*26 + "\n",
            "ATOM     20  C5M   T A   1       2.466   4.961   0.001" + " "*26 + "\n",
            "ATOM     21  C6    T A   1      -0.024   5.057  -0.000" + " "*26 + "\n",
    ],
    'U': [
            "ATOM      1  P     U A   1       1.401   9.017   0.548" + " "*26 + "\n",
            "ATOM      2  O1P   U A   1       1.316  10.253   1.358" + " "*26 + "\n",
            "ATOM      3  O2P   U A   1       2.598   8.175   0.772" + " "*26 + "\n",
            "ATOM      4  O5'   U A   1       0.080   8.140   0.766" + " "*26 + "\n",
            "ATOM      5  C5'   U A   1      -1.189   8.673   0.341" + " "*26 + "\n",
            "ATOM      6  C4'   U A   1      -2.275   7.631   0.524" + " "*26 + "\n",
            "ATOM      7  O4'   U A   1      -2.140   6.628  -0.524" + " "*26 + "\n",
            "ATOM      8  C3'   U A   1      -2.211   6.823   1.820" + " "*26 + "\n",
            "ATOM      9  O3'   U A   1      -2.731   7.553   2.923" + " "*26 + "\n",
            "ATOM     10  C2'   U A   1      -3.046   5.617   1.392" + " "*26 + "\n",
            "ATOM     11  O2'   U A   1      -4.454   5.734   1.323" + " "*26 + "\n",
            "ATOM     12  C1'   U A   1      -2.481   5.354   0.000" + " "*26 + "\n",
            "ATOM     13  N1    U A   1      -1.284   4.500   0.000" + " "*26 + "\n",
            "ATOM     14  C2    U A   1      -1.462   3.131  -0.000" + " "*26 + "\n",
            "ATOM     15  O2    U A   1      -2.563   2.608   0.000" + " "*26 + "\n",
            "ATOM     16  N3    U A   1      -0.302   2.397   0.000" + " "*26 + "\n",
            "ATOM     17  C4    U A   1       0.989   2.884  -0.000" + " "*26 + "\n",
            "ATOM     18  O4    U A   1       1.935   2.094  -0.001" + " "*26 + "\n",
            "ATOM     19  C5    U A   1       1.089   4.311   0.000" + " "*26 + "\n",
            "ATOM     20  C6    U A   1      -0.024   5.053  -0.000" + " "*26 + "\n",
    ],
}


# atomLines['A'] = 

# lib_dir = os.path.dirname(os.path.realpath(__file__))
# # print('library dir: ', lib_dir)
# #%% Load the RNA residue name mapping for modified residues
# resnames_map_file = 'RNAresname_map.txt'
# with open(os.path.join(lib_dir, resnames_map_file), 'r') as ofile:
#     map_lines = ofile.readlines()
# map_lines = [_f for _f in [strline[:-1].strip() for strline in map_lines[1:]] if _f]
# map_lines = [strline for strline in map_lines if strline[0] != '#']
# #%%
# RESNAMES_MOD_MAP = {}
# for _aline in map_lines:
#     tokens_line = _aline.split(' ')
#     # print "%s:%s" % tuple((tokens_line[0], tokens_line[-1]))
#     RESNAMES_MOD_MAP[tokens_line[0]]=tokens_line[-1].upper()
# del lib_dir, ofile, map_lines
# Unsupported: G3A (looks like a G + A), S9L (a linker only), 6U0 (looks like G), 5CY (quite big,probably a linker ususally)
#              3PO (from ATP?), GTG (looks like two G connected)
NA_ResNameModMap = {  "A":"A",   "U":"U",   "G":"G",   "C":"C",   "P":"U", "F86":"A",
                    "C4J":"C", "J48":"U", "O2Z":"A", "D4M":"U", "MUM":"U", "8AZ":"G", "GTA":"G", "ISH":"U", "4AC":"C", "LHH":"C", 
                    "5MC":"C", "B8H":"U", "LV2":"C",  "RY":"C", "A7C":"A", "8RJ":"U", "U5M":"U", "DSH":"A", "8NK":"G", "73W":"C", 
                    "M1Y":"U", "UVP":"U", "56B":"G", "G5J":"G", "75B":"U", "ISI":"U", "G4P":"G", "9QV":"U", "3DR":"C",  
                    # Above added by XQ. C4J should be x, and some U should be T or P
                    "PSU":"U", "3TD":"U", "FHU":"U", "P2U":"U",   
                    "T":"U",  "T3":"U",  "T5":"U",  "DT":"U",  "RT":"U", "DT3":"U", "DT5":"U", "THY":"U", # T changed to U for this line
                    "I":"G",  "+I":"G",   # I changed to G for this line
                    "S":"U",  "+T":"U",  "TS":"U",  "TT":"U", "0DT":"U", "0KZ":"U", "0R5":"U", 
                    "1FZ":"U", "218":"U", "23T":"U", "2AT":"U", "2BT":"U", "2DT":"U", "2GT":"U", "2JU":"U", "2L8":"U", "2NT":"U", 
                    "2OT":"U", "2ST":"U", "3ME":"U", "40T":"U", "4BD":"U", "5AT":"U", "5HT":"U", "5HU":"U", "5MU":"U", "5PY":"U", 
                    "64P":"U", "64T":"U", "6CT":"U", "6HT":"U", "ATD":"U", "ATL":"U", "ATM":"U", "AZT":"U", "BOE":"U", "C6T":"U", 
                    "CTG":"U", "D3T":"U", "DFT":"U", "DRT":"U", "EIT":"U", "F3H":"U", "F4H":"U", "F5H":"U", "F6H":"U", "FMU":"U", 
                    "GMU":"U", "HDP":"U", "HXB":"U", "HXZ":"U", "IPN":"U", "JDT":"U", "MDU":"U", "MFT":"U", "MMT":"U", "MTR":"U", 
                    # The five lines above have T changed to U. 
                    # All lines below are taken from X3DNA
                    "N6T":"U", "NMS":"U", "NMT":"U", "NTT":"U", "P2T":"U", "PBT":"U", "PLR":"U", "PST":"U", "PVX":"U", "PYI":"U", 
                    "QBT":"U", "S2M":"U", "SLD":"U", "SMT":"U", "SPT":"U", "T23":"U", "T2S":"U", "T2T":"U", "T32":"U", "T38":"U", 
                    "T39":"U", "T3P":"U", "T41":"U", "T48":"U", "T49":"U", "T4S":"U", "T64":"U", "T66":"U", "TA3":"U", "TAF":"U", 
                    "TCP":"U", "TDY":"U", "TED":"U", "TFE":"U", "TFF":"U", "TFT":"U", "THM":"U", "THP":"U", "THX":"U", "TLB":"U", 
                    "TLC":"U", "TM2":"U", "TMP":"U", "TP1":"U", "TPN":"U", "TSP":"U", "TTD":"U", "TTE":"U", "TTM":"U", "TTP":"U", 
                    "US2":"U", "US3":"U", "US4":"U", "XTF":"U", "XTH":"U", "XTL":"U", "XTR":"U", "ZDU":"U", "ZP4":"U", "ZTH":"U", 
                     "A3":"A",  "A5":"A",  "DA":"A",  "RA":"A", "ADE":"A", "DA3":"A", "DA5":"A", "RA3":"A", "RA5":"A",  "C3":"C", 
                     "C5":"C",  "DC":"C",  "RC":"C", "CYT":"C", "DC3":"C", "DC5":"C", "RC3":"C", "RC5":"C",  "G3":"G",  "G5":"G", 
                     "DG":"G",  "RG":"G", "DG3":"G", "DG5":"G", "GUA":"G", "RG3":"G", "RG5":"G",  "U3":"U",  "U5":"U",  "DU":"U", 
                     "RU":"U", "RU3":"U", "RU5":"U", "URA":"U", "URI":"U",   "E":"A",   "R":"A",   "Y":"A",  "+A":"A",  "0A":"A", 
                     "AS":"A",  "PU":"A", "02I":"A", "08T":"A", "0L3":"A", "0L4":"A", "0OH":"A", "0OJ":"A", "0UH":"A", "12A":"A", 
                    "1AP":"A", "1DP":"A", "1MA":"A", "1SY":"A", "2AD":"A", "2AR":"A", "2BA":"A", "2BU":"A", "2DA":"A", "2FE":"A", 
                    "2IA":"A", "2MA":"A", "2OA":"A", "2RW":"A", "31H":"A", "31M":"A", "365":"A", "3AD":"A", "3AT":"A", "3D1":"A", 
                    "3DA":"A", "3MA":"A", "40A":"A", "45A":"A", "4BW":"A", "4DU":"A", "54K":"A", "574":"A", "5AA":"A", "5AD":"A", 
                    "5JO":"A", "5UA":"A", "6AP":"A", "6HA":"A", "6HB":"A", "6IA":"A", "6MA":"A", "6MC":"A", "6MD":"A", "6MP":"A", 
                    "6MT":"A", "6MZ":"A", "7AT":"A", "7DA":"A", "84T":"A", "8AN":"A", "8BA":"A", "8XA":"A", "A23":"A", "A2F":"A", 
                    "A2L":"A", "A2M":"A", "A2P":"A", "A38":"A", "A3P":"A", "A40":"A", "A43":"A", "A44":"A", "A47":"A", "A5A":"A", 
                    "A5L":"A", "A5O":"A", "A66":"A", "A6A":"A", "A7E":"A", "A9Z":"A", "ABG":"A", "ABR":"A", "ABS":"A", "ACP":"A", 
                    "AD2":"A", "ADI":"A", "ADK":"A", "ADN":"A", "ADP":"A", "ADS":"A", "AET":"A", "AF2":"A", "AGS":"A", "AMD":"A", 
                    "AMO":"A", "AMP":"A", "ANP":"A", "ANZ":"A", "AP7":"A", "APC":"A", "APN":"A", "AT7":"A", "ATP":"A", "AVC":"A", 
                    "BA2":"A", "CMP":"A", "D5M":"A", "DAD":"A", "DDS":"A", "DJF":"A", "DTP":"A", "DZ4":"A", "DZM":"A", "EDA":"A", 
                    "EEM":"A", "F2A":"A", "F3A":"A", "F3N":"A", "F3O":"A", "FA2":"A", "FAX":"A", "FHA":"A", "FYA":"A", "GOM":"A", 
                    "GSU":"A", "ILA":"A", "LCA":"A", "LMS":"A", "LSS":"A", "MA6":"A", "MA7":"A", "MAD":"A", "MDV":"A", "MIA":"A", 
                    "MSP":"A", "N6G":"A", "N6M":"A", "N79":"A", "NEA":"A", "OIP":"A", "P5P":"A", "PPU":"A", "PPZ":"A", "PR5":"A", 
                    "PRN":"A", "PSD":"A", "PUY":"A", "QSI":"A", "QTP":"A", "RBD":"A", "RIA":"A", "RMP":"A", "S4A":"A", "SAH":"A", 
                    "SAM":"A", "SDE":"A", "SFG":"A", "SMP":"A", "SRA":"A", "SSA":"A", "T6A":"A", "TCY":"A", "TFO":"A", "TNV":"A", 
                    "TSB":"A", "V5A":"A", "VAA":"A", "XAD":"A", "XAL":"A", "XAR":"A", "XUA":"A", "YMP":"A", "ZAD":"A", "ZAN":"A", 
                     "+C":"C",  "0C":"C",  "CH":"C",  "IC":"C",  "LC":"C",  "PC":"C",  "SC":"C", "08Q":"C", "0DC":"C", "0G4":"C", 
                    "0G8":"C", "0KX":"C", "0L6":"C", "0R6":"C", "0R8":"C", "10C":"C", "1CC":"C", "1FC":"C", "1RT":"C", "1RZ":"C", 
                    "1SC":"C", "1W5":"C", "2GF":"C", "2TM":"C", "3TT":"C", "40C":"C", "47C":"C", "4OC":"C", "4SC":"C", "4U3":"C", 
                    "4Y3":"C", "55C":"C", "5BT":"C", "5CF":"C", "5CM":"C", "5FC":"C", "5HC":"C", "5HM":"C", "5IC":"C", "5NC":"C", 
                    "5OC":"C", "5PC":"C", "5SE":"C", "6CF":"C", "6FC":"C", "6HC":"C", "8XC":"C", "A5M":"C", "A6C":"C", "AG9":"C", 
                    "B7C":"C", "BLS":"C", "C25":"C", "C2L":"C", "C2S":"C", "C31":"C", "C34":"C", "C36":"C", "C37":"C", "C38":"C", 
                    "C42":"C", "C43":"C", "C45":"C", "C46":"C", "C49":"C", "C4S":"C", "C5L":"C", "C5P":"C", "C66":"C", "CAR":"C", 
                    "CB2":"C", "CBR":"C", "CBV":"C", "CCC":"C", "CDP":"C", "CFL":"C", "CFZ":"C", "CH1":"C", "CMR":"C", "CP1":"C", 
                    "CPN":"C", "CSG":"C", "CSL":"C", "CTP":"C", "CUD":"C", "CX2":"C", "D00":"C", "DCM":"C", "DCP":"C", "DCT":"C", 
                    "DCZ":"C", "DDY":"C", "DFC":"C", "DNR":"C", "DOC":"C", "EPE":"C", "FTD":"C", "GCK":"C", "GTF":"C", "I5C":"C", 
                    "IMC":"C", "L8P":"C", "LCC":"C", "LCH":"C", "LHC":"C", "LKC":"C", "M5M":"C", "MCY":"C", "MDJ":"C", "MDK":"C", 
                    "MDQ":"C", "ME6":"C", "N4S":"C", "N5C":"C", "N5M":"C", "NCU":"C", "O2C":"C", "OMC":"C", "RCP":"C", "RPC":"C", 
                    "RSP":"C", "RSQ":"C", "S4C":"C", "TPC":"C", "TX2":"C", "XCL":"C", "XCR":"C", "XCT":"C", "XCY":"C", "XJS":"C", 
                    "YYY":"C", "ZBC":"C", "ZCY":"C",   "X":"G",  "+G":"G",  "0G":"G",  "DI":"G",  "GS":"G",  "IG":"G",  "LG":"G", 
                     "YG":"G", "0AD":"G", "0DG":"G", "0L7":"G", "18M":"G", "1GC":"G", "1MG":"G", "1TW":"G", "1WA":"G", "23G":"G", 
                    "2BD":"G", "2BP":"G", "2EG":"G", "2FI":"G", "2JV":"G", "2LA":"G", "2LF":"G", "2MG":"G", "2ON":"G", "2PR":"G", 
                    "2SG":"G", "3ZO":"G", "40G":"G", "4DG":"G", "5CG":"G", "5GP":"G", "63G":"G", "63H":"G", "68Z":"G", "6GO":"G", 
                    "6GU":"G", "6HG":"G", "6OG":"G", "6PO":"G", "7DG":"G", "7GU":"G", "7MG":"G", "8AG":"G", "8DG":"G", "8FG":"G", 
                    "8MG":"G", "8OG":"G", "8XG":"G", "A1P":"G", "A6G":"G", "AGD":"G", "ANG":"G", "BGM":"G", "BGR":"G", "BRG":"G", 
                    "C2E":"G", "C6G":"G", "CG1":"G", "DCG":"G", "DDG":"G", "DFG":"G", "DG8":"G", "DGP":"G", "DGT":"G", "DX4":"G", 
                    "E1X":"G", "EDC":"G", "EDI":"G", "EFG":"G", "EHG":"G", "FAG":"G", "FDG":"G", "FHG":"G", "FMG":"G", "FOX":"G", 
                    "G1C":"G", "G1M":"G", "G25":"G", "G2C":"G", "G2L":"G", "G2M":"G", "G2P":"G", "G2S":"G", "G31":"G", "G32":"G", 
                    "G36":"G", "G38":"G", "G42":"G", "G46":"G", "G47":"G", "G48":"G", "G49":"G", "G7M":"G", "GAO":"G", "GBR":"G", 
                    "GCP":"G", "GDO":"G", "GDP":"G", "GF2":"G", "GFC":"G", "GFF":"G", "GFH":"G", "GFL":"G", "GFM":"G", "GGH":"G", 
                    "GH3":"G", "GMP":"G", "GMS":"G", "GN7":"G", "GNE":"G", "GNG":"G", "GNP":"G", "GPN":"G", "GRB":"G", "GRC":"G", 
                    "GSR":"G", "GSS":"G", "GTP":"G", "GUN":"G", "GX1":"G", "HGL":"G", "HN0":"G", "HN1":"G", "HPA":"G", "IGU":"G", 
                    "IMP":"G", "IQG":"G", "KAG":"G", "LCG":"G", "LGP":"G", "M1G":"G", "M2G":"G", "M7G":"G", "MG1":"G", "MGT":"G", 
                    "MRG":"G", "MTU":"G", "N2G":"G", "O2G":"G", "OGX":"G", "OMG":"G", "OXG":"G", "P9G":"G", "PG7":"G", "PGN":"G", 
                    "PGP":"G", "PPW":"G", "PQ0":"G", "PQ1":"G", "PRF":"G", "QUO":"G", "RDG":"G", "S4G":"G", "S6G":"G", "SDG":"G", 
                    "SOS":"G", "TEP":"G", "TGP":"G", "XAN":"G", "XG4":"G", "XGL":"G", "XGR":"G", "XGU":"G", "XPB":"G", "XUG":"G", 
                    "YYG":"G", "ZGU":"G",   "N":"U",   "Z":"U",  "+U":"U",  "0U":"U",  "IU":"U", "0KL":"U", "0L5":"U", "0R7":"U", 
                    "0U1":"U", "125":"U", "126":"U", "127":"U", "18Q":"U", "1RN":"U", "1TL":"U", "29G":"U", "29H":"U", "2AU":"U", 
                    "2MU":"U", "2QB":"U", "3AU":"U", "3AW":"U", "3AY":"U", "3KA":"U", "4PC":"U", "4SU":"U", "5BU":"U", "5FU":"U", 
                    "5GS":"U", "5IT":"U", "5IU":"U", "5SI":"U", "6FU":"U", "6GS":"U", "70U":"U", "8XU":"U", "9DG":"U", "A6U":"U", 
                    "BMN":"U", "BMR":"U", "BRO":"U", "BRU":"U", "CDW":"U", "CM0":"U", "CSM":"U", "D3N":"U", "DHU":"U", "DRP":"U", 
                    "DUP":"U", "DUR":"U", "DUT":"U", "DUZ":"U", "FFD":"U", "H2U":"U", "HEU":"U", "HMU":"U", "KIR":"U", "LHU":"U", 
                    "LLP":"U", "LTP":"U", "MAU":"U", "MEP":"U", "MHT":"U", "MNU":"U", "NF2":"U", "NPF":"U", "NYM":"U", "OHU":"U", 
                    "OMU":"U", "ONE":"U", "PDU":"U", "PYO":"U", "PYY":"U", "RCE":"U", "RUS":"U", "S4U":"U", "SSJ":"U", "SSU":"U", 
                    "SUR":"U", "T5O":"U", "T5S":"U", "TDR":"U", "TLN":"U", "TTI":"U", "U25":"U", "U2L":"U", "U2N":"U", "U31":"U", 
                    "U33":"U", "U34":"U", "U36":"U", "U37":"U", "U3H":"U", "U5P":"U", "U8U":"U", "UAR":"U", "UBB":"U", "UBD":"U", 
                    "UBI":"U", "UCL":"U", "UCP":"U", "UD5":"U", "UDP":"U", "UF2":"U", "UFP":"U", "UFR":"U", "UFT":"U", "ULF":"U", 
                    "UMP":"U", "UMS":"U", "UMX":"U", "UPC":"U", "UPE":"U", "UPG":"U", "UPS":"U", "UPV":"U", "UR3":"U", "URU":"U", 
                    "URX":"U", "US1":"U", "US5":"U", "USM":"U", "UTP":"U", "UVX":"U", "UZR":"U", "XAE":"U", "XBB":"U", "XGA":"U", 
                    "Y5P":"U", "YCO":"U", "ZBU":"U", "ZHP":"U"}

#%% convert residue names to single letter codes
def res2seqcode(resnames,  restype='auto', undef_code="?"):
    """Return the one-letter amino acid or nucleic acids code from the residue name.
    Unrecognized residues are returned as "?".
    input should be a list of single residues (either one, two, or three letters)
    """
    if isinstance(resnames, str): resnames = [resnames]

    if restype.upper() == 'AUTO' :
        aacode = [AA_ResName3to1Dict.get(_s, undef_code) for _s in resnames]
        nacode = [NA_ResNameModMap.get(_s, undef_code) for _s in resnames]
        # simple vote, may not be the best way for this.
        if aacode.count(undef_code) < nacode.count(undef_code):
            return aacode
        else:
            return nacode

    if restype.upper() == 'NA':
        return [NA_ResNameModMap.get(_s, undef_code) for _s in resnames]

    if restype.upper() == 'AA':
        return [AA_ResName3to1Dict.get(_s, undef_code) for _s in resnames]  
    
numCommonRes = 0
for _key in NA_ResNameModMap.keys():
    NA_ResNameModMap[_key] = NA_ResNameModMap[_key].upper()
    if _key in AA_ResName3to1Dict.keys() : numCommonRes += 1

logger.debug('{} out {} residue names are the same between AA and NA'.format(numCommonRes, len(NA_ResNameModMap)))

# One-hot encoding

# Turner energy parameters for RNA secondary structure
# copied from https://rna.urmc.rochester.edu/NNDB/turner04/wc-parameters.html
Turner_bpEnergy_Params = defaultdict(lambda: defaultdict(lambda: 0.))
Turner_bpEnergy_Params.update({
    'AU': 2.0,
    'GC': 3.0,
    'GU': 0.0,
})

_lambda_default_nnEnergy = lambda: [0., 0., 0., 0.]
Turner_nnEnergy_Params = defaultdict(lambda: defaultdict(_lambda_default_nnEnergy))
Turner_nnEnergy_Params.update({
    'AA': defaultdict(_lambda_default_nnEnergy, {
        'UU': [-0.93, 0.03, -6.82, 0.79],
    }),
    'AU': defaultdict(_lambda_default_nnEnergy, {
        'UA': [-1.10, 0.08, -9.38, 1.68],
        'UG': [-1.36, 0.24, -8.81, 2.10],
    }),
    'UA': defaultdict(_lambda_default_nnEnergy, {
        'AU': [-1.33, 0.09, -7.69, 2.02],
    }),
    'CU': defaultdict(_lambda_default_nnEnergy, {
        'GA': [-2.08, 0.06, -10.48, 1.24],
        'GG': [-2.11, 0.25, -12.11, 2.22],
    }),
    'CA': defaultdict(_lambda_default_nnEnergy, {
        'GU': [-2.11, 0.07, -10.44, 1.28],
    }),
    'GU': defaultdict(_lambda_default_nnEnergy, {
        'CA': [-2.24, 0.06, -11.40, 1.23],
        'CG': [-2.51, 0.25, -12.59, 2.18],
        'UG': [ 1.29, 0.56, -14.58, 4.92], # note b: GGUC-CUGG is favorable with total G of -4.12
    }),
    'GA': defaultdict(_lambda_default_nnEnergy, {
        'CU': [-2.35, 0.06, -12.44, 1.20],
        'UU': [-1.27, 0.28, -12.83, 2.44],
    }),
    'CG': defaultdict(_lambda_default_nnEnergy, {
        'GC': [-2.36, 0.09, -10.64, 1.65],
        'GU': [-1.41, 0.24, -5.61, 2.13],
    }),
    'GG': defaultdict(_lambda_default_nnEnergy, {
        'CC': [-3.26, 0.07, -13.39, 1.24],
        'CU': [-1.53, 0.27, -8.33, 2.33],
        'UU': [-0.50, 0.96, -13.47, 8.37], # note a: G=+0.47
    }),
    'GC': defaultdict(_lambda_default_nnEnergy, {
        'CG': [-3.42, 0.08, -14.88, 1.58],
    }),
    # GU pairs
    'AG': defaultdict(_lambda_default_nnEnergy, {
        'UU': [-0.55, 0.32, -3.21, 2.76],
    }),
    'GGUC': defaultdict(_lambda_default_nnEnergy, {
        'CUGG': [-4.12, 0.54, -30.80, 8.87],
    }),
    'UG': defaultdict(_lambda_default_nnEnergy, {
        'AU': [-1.00, 0.30, -6.99, 2.64],
        'GU': [ 0.30, 0.48, -9.26, 4.19],
    }),    
})
# include symmetrically equivalents
_Turner_nnEnergy_Params = defaultdict(lambda: defaultdict(_lambda_default_nnEnergy))
for _bp5to3, _energy_dict in Turner_nnEnergy_Params.items():
    for _bp3to5, _energies in _energy_dict.items():
        if _bp3to5[::-1] not in Turner_nnEnergy_Params:
            _Turner_nnEnergy_Params[_bp3to5[::-1]][_bp5to3[::-1]] = _energies
        else:
            Turner_nnEnergy_Params[_bp3to5[::-1]][_bp5to3[::-1]] = _energies
Turner_nnEnergy_Params.update(_Turner_nnEnergy_Params)

# Structures

