import pandas as pd
import numpy as np
import bbi
from gtfparse import read_gtf
import bioframe
from akita_utils.dna_utils import scan_motif
from akita_utils.seq_gens import generate_spans_start_positions


def generate_signal_matrix(
    interval_df,
    chip_seq_file,
    columns=["chrom", "start", "end"],
    window_size=1000,
    window_type="extend",
    nbins=40,
):
    """
    Uses pybbi to measure signal over a set of input intervals.
    Returns a matrix [n_intervals x nbins] of the average of ChIP signal over
    each bin in the matrix.
    Parameters:
    -----------
    interval_df: pandas dataframe that has the list of intervals (usually,
                set of genes)
    chip_seq_file: filepath to ChIP-seq file of type bigWig or bigBed
    columns: the columns in interval_df that define the interval
    window_size: the distance of the
    window_type: 'extend': window_size defines padding length added to ends
                 'centered': window_size extends out from each side of middle
    nbins: number of bins the window is divided into.
    Returns:
    --------
    matrix: np.array of shape [n_intervals x nbins]
            rows correspond to intervals in interval_df and ChIP signal measured
            across column bins.
    """

    intervals = bioframe.from_any(interval_df[columns])
    intervals = intervals.rename(
        columns={columns[0]: "chrom", columns[1]: "start", columns[2]: "end"}
    )
    intervals = bioframe.sanitize_bedframe(intervals)

    if window_type == "extend":
        shapes = pd.Series(intervals["start"] - intervals["end"]).nunique()
        msg = """The size of intervals should be equal to perform stackup.
                 Try window_type: 'centered'"""
        assert shapes == 1, msg

        intervals = bioframe.expand(intervals, pad=window_size)

    if window_type == "centered":
        intervals = bioframe.expand(bioframe.expand(expanded, scale=0), pad=1000)

    with bbi.open(chip_seq_file) as f:
        matrix = f.stackup(
            intervals["chrom"], intervals["start"], intervals["end"], bins=nbins
        )

    return matrix



def bioframe_clean_autosomes(frame):
    """
    Takes a dataframe or bioframe and returns
    a sanitized bioframe with only the intervals on autosomes.
    """
    
    frame = frame[frame.chrom.str.match('^chr\d+$')]
    frame = bioframe.sanitize_bedframe(frame)
        
    return frame

def get_tss_gene_intervals(
    tss_df, 
    return_cols=["gene_id", "chrom", "start", "end", "strand"],
    chrom_keep='autosomes',
    ):
    """
    In: a .gtf file containing the chr, start, end
    corresponding to the TSS for the transcripts ready from a
    genomic .gtf format annotation file.
    Output: a dataframe in bioframe format with a single TSS
    per gene, with non-autosomes removed.
    """
    
    # rename column to chrom to match bedframe/bioframe format
    tss_df = tss_df.rename(columns={"seqname": "chrom"})

    # Removing pseudo chromosomes
    if chrom_keep == 'autosomes':
        tss_df = bioframe_clean_autosomes(tss_df)
    elif chrom_keep == 'chromosomes':
        tss_df = bioframe_clean_chromosomes(tss_df)

    # drop duplicate TSSes
    return tss_df[return_cols].drop_duplicates(["gene_id"])


def label_DE_status(
    df,
    significance_col="padj",
    significance_threshold=0.05,
    fold_change_col="log2FoldChange",
    fold_change_threshold=0,
    DE_status_column="DE_status",
):
    """
    Appends a column with differential expression status to a DataFrame with
    padj and log2FoldChange columns
    Returns
    --------
    df_out : pd.DataFrame
        DataFrame with DE_status_column
    """
    df_out = df.copy()
    df_out[DE_status_column] = "nonsig"
    sig_inds = df[significance_col] < significance_threshold
    down_sig_inds = (df[fold_change_col] < fold_change_threshold) * sig_inds
    up_sig_inds = (df[fold_change_col] > fold_change_threshold) * sig_inds
    df_out.loc[up_sig_inds, DE_status_column] = "up"
    df_out.loc[down_sig_inds, DE_status_column] = "down"

    return df_out


def generate_promoter_df(feature_dataframe,up_stream_bps = 10000):
    for row in feature_dataframe.itertuples():
        if row.strand == "+":
            feature_dataframe['start'].at[row.Index] = row.start - up_stream_bps
        else:
            feature_dataframe['end'].at[row.Index] = row.end + up_stream_bps
    return feature_dataframe

#-------------------------------not used yet --------------------------------
chip_dir = ('/project/fudenber_735/collaborations/karissa_2022/'+
            '2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14/')
bed_dict = {
    'H3K27Ac':chip_dir+'H3K27ac_EA92-97_peaks.xls.bed',
    'CTCF':   chip_dir+'CTCF_peaks_called_on_4reps_foundInatLeast2reps_noBlacklist.bed',
    'Nipbl':   chip_dir+'Nipbl_112.175.197.114.177.196_peaks.xls.bed',
    'Rad21':  chip_dir+'RAD21_peaks_called_on_6reps_foundInatLeast3reps_noBlacklist.bed'
}

# Load Chip-Seq files 
chip_folder = "/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14/"

ctcf_new = "CTCF_E14_RSC13new-22-37-60_average.bw"
rad21 = "RAD21_E14_RSC12new-21-36-59-74_average.bw"
nipbl = "NIPBL_E14_EA112-EA175_average.bw"
ring1b = "RING1B_E14_RSC24-39-62_average.bw"
promoter = "H3K27Ac_mESCs_EA92-EA94_average.bw"

#-------------------------------not used yet --------------------------------
dataset_folder = '/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/Published_datasets/'
chen_s1 = 'Enhancers_Chen2012_S1_remapped_mm10.bed'
whythe_super = 'Super-enhancers_mESCs_(OSN-MED1)_Wythe-Cell-2023_mm10-lifetover.bed'

enhancer_dict = {'enh_chen_s1' : dataset_folder+chen_s1,
                 'enh_wythe_super' : dataset_folder+whythe_super}

bed_dict = {**bed_dict, **enhancer_dict} # dict with all required data


proj = ("/project/fudenber_735/collaborations/karissa_2022/"+
        "20220812_EA18-1_RNAseq-Analysis_forGeoff/")

# Importing day 1 depletion in ESCs DEGS
day1_sigRes = 'EA18.1_ESC_1d-depletion_DESeq2/20220817_EA18-1_resSig_ESC_1d-depletion.csv'

# Sample count data for the non-significant results
normalized_counts = 'EA18.1_ESC_1d-depletion_DESeq2/20220817_EA18-1_ESC-1d_sf-normalized.csv'
vst_normalized_counts = 'EA18.1_ESC_1d-depletion_DESeq2/20220817_EA18-1_ESC-1d_sf-normalized_vst-transformed.csv'
feature_counts = '20220816_featureCounts.csv'

WT_samples = ['KHRNA1', 'KHRNA7', 'KHRNA13', 'KHRNA22', 'KHRNA23', 'KHRNA50']
day1_res_df = pd.read_csv(proj+day1_sigRes)


# import table of raw feature counts and calculate average
feat_counts_df = pd.read_csv(proj+feature_counts).rename(columns={'Unnamed: 0' : 'Geneid'})
feat_counts_df['avg'] = feat_counts_df[WT_samples].mean(axis='columns')
# print('raw feature counts shape: ', str(feat_counts_df.shape))

# import table of normalized feature counts and calculate average
vst_counts_df = pd.read_csv(proj+vst_normalized_counts).rename(columns={'Unnamed: 0' : 'Geneid'})
vst_counts_df['avg'] = vst_counts_df[WT_samples].mean(axis='columns')
# print('vst normalized feature counts shape: ', str(vst_counts_df.shape))

feat_counts_df = feat_counts_df.merge(vst_counts_df, on='Geneid', how='left', suffixes=('_counts', '_vst_counts'))
feat_counts_df['avg_vst_counts'].fillna(feat_counts_df['avg_counts'], inplace=True)

# print(feat_counts_df.shape)
# print(feat_counts_df['avg_vst_counts'].isna().sum())

# add average normalized counts value to results df
day1_res_df = day1_res_df.merge(feat_counts_df[['Geneid', 'avg_vst_counts', 'avg_counts']], on='Geneid', how='outer')
df = day1_res_df.copy()

tss_df = read_gtf("/project/fudenber_735/collaborations/karissa_2022/" + "old/RNAseq/STAR_Gencode_alignment/tss_annotions_gencode.vM23.primary_assembly.gtf")
tss_intervals = get_tss_gene_intervals(tss_df)
tss_intervals['tss'] = tss_intervals['start'].copy()

### create df with intervals & expression data
tss_df = tss_intervals.merge(df.copy(),  how='left',left_on='gene_id', right_on='Geneid')
tss_df = label_DE_status(tss_df)
tss_df = tss_df.query("avg_counts> 5").copy()
tss_df.to_csv(f'./data/tss_dataframe.tsv', sep='\t', index=False)


# experiental stuff down ****************************************************
enhancer_df = bioframe.read_table(enhancer_dict["enh_chen_s1"], schema='bed3', header=1)
enhancer_df = bioframe_clean_autosomes(enhancer_df)
nbins=1

enhancer_NIPBL_df = pd.DataFrame(generate_signal_matrix(enhancer_df,chip_folder+nipbl,nbins=nbins),columns=[f"enhancer_NIPBL_score_{i}" for i in range(nbins)])  
enhancer_H3K27Ac_df = pd.DataFrame(generate_signal_matrix(enhancer_df,chip_folder+promoter,nbins=nbins),columns=[f"enhancer_H3K27Ac_score_{i}" for i in range(nbins)])
enhancer_merged_df = pd.concat([enhancer_df.reset_index(drop=True),enhancer_NIPBL_df.reset_index(drop=True),enhancer_H3K27Ac_df.reset_index(drop=True)], axis=1) # 
enhancer_merged_df.to_csv("./data/enhancer_score_sample.csv")


promoter_df = generate_promoter_df(tss_df,up_stream_bps = 20000)
promoter_NIPBL_df = pd.DataFrame(generate_signal_matrix(promoter_df,chip_folder+nipbl,nbins=nbins),columns=[f"promoter_NIPBL_score_{i}" for i in range(nbins)])  
promoter_H3K27Ac_df = pd.DataFrame(generate_signal_matrix(promoter_df,chip_folder+promoter,nbins=nbins),columns=[f"promoter_H3K27Ac_score_{i}" for i in range(nbins)])
promoter_merged_df = pd.concat([promoter_df.reset_index(drop=True),promoter_NIPBL_df.reset_index(drop=True),promoter_H3K27Ac_df.reset_index(drop=True)], axis=1)
promoter_merged_df.to_csv("./data/promoter_score_sample.csv")





# print(merged_df) 
# merged_df['NIPBL_signal'] = 'non_significant'
# merged_df.loc[merged_df["NIPBL_score_0"] >= 4, 'NIPBL_signal'] = 'significant' 
# merged_df = merged_df.loc[merged_df['NIPBL_signal'] == 'significant' ].reset_index(drop=True)

# merged_df['H3K27Ac_signal'] = 'non_significant'
# merged_df.loc[merged_df["H3K27Ac_score_0"] >= 20, 'H3K27Ac_signal'] = 'significant'
# merged_df = merged_df.loc[merged_df['H3K27Ac_signal'] == 'significant' ].reset_index(drop=True)

 

# experiental stuff down ****************************************************
# nipbl_df = bioframe.read_table(bed_dict['Nipbl'], schema='bed3', header=1)
# nipbl_df = bioframe_clean_autosomes(nipbl_df)
# nipbl_df.to_csv(f"./data/NIPBL_data.csv")

# nipl_enhancer_df = bioframe.overlap(enhancer_df, nipbl_df, how='inner', suffixes=('_1','_2'))
# nipl_enhancer_df.to_csv(f"./data/NIPBL_enhancer.csv")


# # experiental stuff down ****************************************************
# H3K27Ac_df = bioframe.read_table(bed_dict['H3K27Ac'], schema='bed3', header=1)
# H3K27Ac_df = bioframe_clean_autosomes(H3K27Ac_df)
# H3K27Ac_df.to_csv(f"./data/H3K27Ac_data.csv")

# H3K27Ac_enhancer_df = bioframe.overlap(enhancer_df, H3K27Ac_df, how='inner', suffixes=('_1','_2'))
# H3K27Ac_enhancer_df.to_csv(f"./data/H3K27Ac_enhancer.csv")
