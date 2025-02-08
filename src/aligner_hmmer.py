from subprocess import run
from Bio import SeqIO

parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-i", "--inputfile", help="Specify the full path of the input file with cdr3b sequences to align")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file to save aligned cdr3b")
parser.add_argument("-r", "--remove", default=True, help="Remove auxiliary files after alignment is done: True/False")
args = parser.parse_args()

# open the file and parse sequences using BioPython
def _sequences_to_str_array(filename, fmt="fasta"):
    return np.asarray([list(str(record.seq)) for record in SeqIO.parse(filename, fmt)])

# run hmm command to bash: builds a profile hmm from the cdr3b data aligned
run(["hmmbuild"," --amino", "./outfile.hmm", "./tobuild_aligner.stockholm"])

input_filename =  agrs.inputfile
# aligns new cdr3b sequences usign the hmm profile created
run(["hmmalign","--amino", "--trim","-o", "./aligned_hmm.stockholm", "./outfile.hmm", input_filename]) 

# convert Stockholm format to fasta using BioPython
aligned_msa_stockholm = './aligned_hmm.stockholm'
aligned_msa_fasta = './tmp_file.fasta'
parsed = list(SeqIO.parse(aligned_msa_stockholm, "stockholm"))
with open(aligned_msa_fasta, "w") as file:
    SeqIO.write(parsed, file, "fasta")
    
msa_arr = _sequences_to_str_array(aligned_msa_fasta)

# make dictionary to upper case amino acids
translation = {ord(letter): ord(string.capwords(letter) ) for letter in string.ascii_lowercase}

# save aligned cdr3b to text like file
output_filename = 'msa_aligned_hmmalign_no_inserts.txt'
parsed = list(SeqIO.parse(aligned_msa_fasta, 'fasta'))
with open(removed_inserts_file, "w") as out_file:
    for i, record in enumerate(parsed):
        tmp_msa_seq= ''.join(msa_arr[i]).translate(translation) 
        if tmp_msa_seq.startswith('C'):
            out_file.write(tmp_msa_seq)
            out_file.write('\n')
            
remove = args.remove         
if remove:
    run(["rm", './aligned_hmm.stockholm'])
    run(["rm", './tmp_file.fasta'])

    
            
            
            
            
