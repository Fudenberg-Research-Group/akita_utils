import pandas as pd
import numpy as np
import bbi
from gtfparse import read_gtf
import bioframe


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

#-------------------------------not used yet --------------------------------
chip_dir = ('/project/fudenber_735/collaborations/karissa_2022/'+
            '2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14/')
bed_dict = {
    'H3K27Ac':chip_dir+'H3K27ac_EA92-97_peaks.xls.bed',
    'CTCF':   chip_dir+'CTCF_peaks_called_on_4reps_foundInatLeast2reps_noBlacklist.bed',
    'Nipbl':   chip_dir+'Nipbl_112.175.197.114.177.196_peaks.xls.bed',
    'Rad21':  chip_dir+'RAD21_peaks_called_on_6reps_foundInatLeast3reps_noBlacklist.bed'
}
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
print('raw feature counts shape: ', str(feat_counts_df.shape))

# import table of normalized feature counts and calculate average
vst_counts_df = pd.read_csv(proj+vst_normalized_counts).rename(columns={'Unnamed: 0' : 'Geneid'})
vst_counts_df['avg'] = vst_counts_df[WT_samples].mean(axis='columns')
print('vst normalized feature counts shape: ', str(vst_counts_df.shape))

feat_counts_df = feat_counts_df.merge(vst_counts_df, on='Geneid', how='left', suffixes=('_counts', '_vst_counts'))
feat_counts_df['avg_vst_counts'].fillna(feat_counts_df['avg_counts'], inplace=True)

print(feat_counts_df.shape)
print(feat_counts_df['avg_vst_counts'].isna().sum())

# add average normalized counts value to results df
day1_res_df = day1_res_df.merge(feat_counts_df[['Geneid', 'avg_vst_counts', 'avg_counts']], on='Geneid', how='outer')
df = day1_res_df.copy()

tss_df = read_gtf("/project/fudenber_735/collaborations/karissa_2022/"+
      "old/RNAseq/STAR_Gencode_alignment/tss_annotions_gencode.vM23.primary_assembly.gtf")
tss_intervals = get_tss_gene_intervals(tss_df)
tss_intervals['tss'] = tss_intervals['start'].copy()

### create df with intervals & expression data
tss_df = tss_intervals.merge(df.copy(),  how='left',left_on='gene_id', right_on='Geneid')
tss_df = label_DE_status(tss_df)
tss_df = tss_df.query("avg_counts> 5").copy()

tss_df_up_expression = tss_df[True == (tss_df["DE_status"]=="up")]
tss_df_down_expression = tss_df[True == (tss_df["DE_status"]=="down")]

number_of_genes = 10 # will see if there is further modification needed
feature_dataframe = pd.concat([tss_df_up_expression.iloc[:number_of_genes,:], tss_df_down_expression.iloc[:number_of_genes,:]], ignore_index=True) 
feature_dataframe=feature_dataframe[["gene_id", "chrom", "start", "end", "strand","DE_status","SYMBOL"]]
feature_dataframe.to_csv(f'./data/feature_dataframe.tsv', sep='\t', index=False)


# feature_df = bioframe.read_table(enhancer_dict["enh_chen_s1"], schema='bed3', header=1)
# feature_df.to_csv(f"enhancer_data.tsv")