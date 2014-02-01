#!/usr/bin/python
"""
generate
"""

from collections import defaultdict
import os
import helper



def working_stuff():

    from BCBio import GFF
    
        
    gff_type = ["gene", "mRNA", "CDS", "exon"]
    source_type = zip(["Coding_transcript"] * len(gff_type), gff_type)
    
    filter_type = dict(gff_source_type = source_type, gff_id = "I")
    
    gff_handle = open("/fml/ag-raetsch/share/databases/genomes/C_elegans/elegans_WS199/annotation/c_elegans.WS199.gff3")
    
    element = [e for e in GFF.parse(gff_handle, limit_info=filter_type)]


    return element


def create_taxonomy():
    
    from task_similarities import TreeNode
    root = TreeNode("root")
    chordata = TreeNode("chordata")
    protostomia = TreeNode("protostomia")
    root.add_child(chordata)
    root.add_child(protostomia)
    c_savignyi = TreeNode("c_savignyi")
    chordata.add_child(c_savignyi)
    vertebrata = TreeNode("vertebrata")
    chordata.add_child(vertebrata)
    actinopterygii = TreeNode("actinopterygii")
    vertebrata.add_child(actinopterygii)
    d_rerio = TreeNode("d_rerio")
    actinopterygii.add_child(d_rerio)
    g_aculeatus = TreeNode("g_aculeatus")
    actinopterygii.add_child(g_aculeatus)
    t_nigroviridis = TreeNode("t_nigroviridis")
    actinopterygii.add_child(t_nigroviridis)
    aves = TreeNode("aves")
    vertebrata.add_child(aves)
    g_gallus = TreeNode("g_gallus")
    aves.add_child(g_gallus)
    m_gallopavo = TreeNode("m_gallopavo")
    aves.add_child(m_gallopavo)
    mammals = TreeNode("mammals")
    vertebrata.add_child(mammals)
    b_taurus = TreeNode("b_taurus")
    mammals.add_child(b_taurus)
    h_sapiens = TreeNode("h_sapiens")
    mammals.add_child(h_sapiens)
    m_musculus = TreeNode("m_musculus")
    mammals.add_child(m_musculus)
    protostomia.children
    c_elegans = TreeNode("c_elegans")
    protostomia.add_child(c_elegans)
    d_melanogaster = TreeNode("d_melanogaster")
    protostomia.add_child(d_melanogaster)
    root.plot()


def get_positions_GFT(file_name, chromosome_names):
    '''
    the special feature of GTF files is the presence, of
    a special entry "start_codon", which can be used to create
    labeled data
    '''
    
    
    positions = defaultdict(list)
    
    f = file(file_name)
    
    for line in f:
        
        tokens = line.strip().split("\t")
        
        #print line, len(tokens), tokens
        
        # fetch only start codon from pos strand for chr set
        if tokens[2]=="start_codon" and tokens[6] == "+":
            
            if chromosome_names==None or tokens[0] in chromosome_names:
                positions[tokens[0]].append(int(tokens[3]))

    f.close()
    
    return positions



class GenomeHandler:
    
    
    def __init__(self, fasta_fn, chr_list):
        
        self.fasta_fn = fasta_fn
        self.chr_list = chr_list
        self.seqs = {}

        from Bio import SeqIO
        seq_io = SeqIO.parse(file(fasta_fn), "fasta")
        
        for record in seq_io:
            
            if record.id in chr_list:
                
                print "loading chromosome %s" % (record.id)
                
                self.seqs[record.id] = record
                        

    def get_codon(self, chr_name, pos):
        
        assert (chr_name in self.chr_list)
        
        return self.seqs[chr_name].seq[pos-1:pos-1 + 3]
        

    def get_window(self, chr_name, pos):
                       
        return self.seqs[chr_name].seq[pos-100:pos+100]
    
    
    def get_length(self, chr_name):
        
        return len(self.seqs[chr_name])



def get_chr_names(org_name):
    

    chr_names = None    
    
    if org_name == "b_taurus":        
        chr_names = [str(i) for i in range(1, 10)]         
 
    if org_name == "c_elegans":       
        chr_names = ["I", "II", "III", "IV", "V"]
 
    if org_name == "d_melanogaster": 
        chr_names = ['3RHet', '2R', '3R', '2RHet', '3LHet', '2LHet', '4', '3L', '2L']
        
    if org_name == "m_musculus":        
        chr_names = [str(i) for i in range(1, 5)]  
        
    if org_name == "h_sapiens":
        chr_names = [str(i) for i in range(1, 5)]     
        
    return chr_names



        
def create_seq_data(org_name, work_dir):
    '''
    the special feature of GTF files is the presence, of
    a special entry "start_codon", which can be used to create
    labeled data
    '''

    print "processing organism", org_name

    files = os.listdir(work_dir)
       
    fn_seq = work_dir + [fn for fn in files if fn.endswith(".fa")][0] 
    fn_pos = work_dir + [fn for fn in files if fn.endswith(".gtf")][0]
        
    chr_names = get_chr_names(org_name)

    max_mismatches = 2

    print "loading positions"

    # load positions
    positions = get_positions_GFT(fn_pos, chr_names)
    
    chr_names = positions.keys()
    
    print "done with positions" 
    
    genome = GenomeHandler(fn_seq, chr_names)

    
    pos_seqs = []
    neg_seqs = []
    
    
    for chr in chr_names:
        
        print "processing chromosome %s" % (chr) 

        # assemble positive list
        false_positions = set()
           
        for pos in positions[chr]:
        
            codon = genome.get_codon(chr, pos)
            
            if codon.count("ATG") != 1:
                false_positions.add(pos)
                print "ARRGH", codon
            else:
                if genome.get_window(chr, pos).count("N") < max_mismatches:
                    pos_seqs.append(genome.get_window(chr, pos).tostring().replace("N", "A"))
                else:
                    print "discarding candidate because of %i mismatches, current len=%i" % (genome.get_window(chr, pos).count("N"), len(pos_seqs)) 
            
        if len(pos_seqs) > 1000:
            break
        print "WARNING: number of incorrect positions: %i" % (len(false_positions))
    
    
    
    # generate negative list

    print chr_names

    for chr in chr_names:
        
        margin = 1000
        
        print "processing %s to generate negative examples" % (chr)
        
        for i in xrange(margin, genome.get_length(chr) - margin):
            
            if genome.get_codon(chr, i).count("ATG") == 1:
                    
                if not i in positions[chr]:

                    if genome.get_window(chr, i).count("N") < max_mismatches:
                        neg_seqs.append(genome.get_window(chr, i).tostring().replace("N", "A"))
                    
                    if len(neg_seqs) > len(pos_seqs)*5:
                        print "enough negative examples"
                        
                        return (neg_seqs, pos_seqs)
                    
                else:
                    print "hit pos example %s %i" % (chr, i) 
        
        
    return (neg_seqs, pos_seqs) 



def insert_into_database():
    
    base_dir = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/translation_start/"
    organisms = os.listdir(base_dir)
    
    data = defaultdict(dict)
    
    for org_name in organisms:
        
        work_dir = base_dir + org_name + "/"
        save_fn = work_dir + "seqs.pickle"
        
        data_raw = helper.load(save_fn)

        data_raw["neg"] = [s for s in data_raw["neg"] if len(s)!=0][0:6000]
        data_raw["pos"] = [s for s in data_raw["pos"] if len(s)!=0][0:60]

        labels = [-1]*len(data_raw["neg"]) + [1]*len(data_raw["pos"])
        examples = [e.upper() for e in (data_raw["neg"] + data_raw["pos"])]
        
        
    
        data[org_name]["LT"] = labels
        data[org_name]["XT"] = examples
        
        
    import data_processing
    
    data_processing.prepare_multi_datasets(data, 0.35, num_splits=7, description="start_codon tiny", feature_type="string", write_db=True, random=True)



def main():

    base_dir = "/fml/ag-raetsch/home/cwidmer/Documents/phd/projects/multitask/data/translation_start/"
    organisms = os.listdir(base_dir)    
        
        
    for org_name in organisms:
    
        work_dir = base_dir + org_name + "/"
            
        (neg, pos) = create_seq_data(org_name, work_dir)
        
        result = {}
        result["pos"] = pos
        result["neg"] = neg

        print "======================="
        print "%s pos=%i, neg=%i" % (org_name, len(pos), len(neg))

        save_fn = work_dir + "seqs.pickle"
        
        helper.save(save_fn, result)
        
    


def check_C_testset(mss_id):
    
    import pylab
    import expenv
    import numpy
    from helper import Options
    from method_hierarchy_svm_new import Method
    #from method_augmented_svm_new import Method
    
    
    #costs = 10000 #[float(c) for c in numpy.exp(numpy.linspace(numpy.log(10), numpy.log(20000), 6))]
    costs = [float(c) for c in numpy.exp(numpy.linspace(numpy.log(0.4), numpy.log(10), 6))] 
    
    print costs
    
    mss = expenv.MultiSplitSet.get(mss_id)
    
    train = mss.get_train_data(-1)
    test = mss.get_eval_data(-1)
    
    au_roc = []
    au_prc = []
    
    for cost in costs:
        #create mock param object by freezable struct
        param = Options()
        param.kernel = "WeightedDegreeStringKernel"
        param.wdk_degree = 10
        param.transform = cost
        param.base_similarity = 1.0
        param.taxonomy = mss.taxonomy
        param.id = 666
    
        #param.cost = cost
        param.cost = 10000
        param.freeze()
    
        # train
        mymethod = Method(param)
        mymethod.train(train)
    
        assessment = mymethod.evaluate(test)
        
        au_roc.append(assessment.auROC)
        au_prc.append(assessment.auPRC)
        
        print assessment
        assessment.destroySelf()

    pylab.title("auROC")
    pylab.semilogx(costs, au_roc, "-o")
    
    pylab.show()
    pylab.figure()
    pylab.title("auPRC")
    pylab.semilogx(costs, au_prc, "-o")
    pylab.show()
    
    return (costs, au_roc, au_prc)


if __name__ == "__main__":
    main()