import json

# define file paths
chip_dir = "/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/ChIP-seq_in_WT-parental-E14/"
bed_files = {
    "H3K27Ac": f"{chip_dir}H3K27ac_EA92-97_peaks.xls.bed",
    "CTCF": f"{chip_dir}CTCF_peaks_called_on_4reps_foundInatLeast2reps_noBlacklist.bed",
    "Nipbl": f"{chip_dir}Nipbl_112.175.197.114.177.196_peaks.xls.bed",
    "Rad21": f"{chip_dir}RAD21_peaks_called_on_6reps_foundInatLeast3reps_noBlacklist.bed",
}

bw_files = {
    "CTCF": f"{chip_dir}CTCF_E14_RSC13new-22-37-60_average.bw",
    "Rad21": f"{chip_dir}RAD21_E14_RSC12new-21-36-59-74_average.bw",
    "Nipbl": f"{chip_dir}NIPBL_E14_EA112-EA175_average.bw",
    "Ring1b": f"{chip_dir}RING1B_E14_RSC24-39-62_average.bw",
    "H3K27Ac": f"{chip_dir}H3K27Ac_mESCs_EA92-EA94_average.bw",
}

dataset_dir = "/project/fudenber_735/collaborations/karissa_2022/2022_09_features_for_RNAseq/Published_datasets/"
enhancer_files = {
    "enh_chen_s1": f"{dataset_dir}Enhancers_Chen2012_S1_remapped_mm10.bed",
    "enh_wythe_super": f"{dataset_dir}Super-enhancers_mESCs_(OSN-MED1)_Wythe-Cell-2023_mm10-lifetover.bed",
}

proj_dir = "/project/fudenber_735/collaborations/karissa_2022/"
sub_folder_1 =  "20220812_EA18-1_RNAseq-Analysis_forGeoff/"

proj_files = {
    "day1_sigRes": f"{proj_dir}{sub_folder_1}EA18.1_ESC_1d-depletion_DESeq2/20220817_EA18-1_resSig_ESC_1d-depletion.csv",
    "normalized_counts" = f"{proj_dir}{sub_folder_1}EA18.1_ESC_1d-depletion_DESeq2/20220817_EA18-1_ESC-1d_sf-normalized.csv",
    "feature_counts": f"{proj_dir}{sub_folder_1}20220816_featureCounts.csv",
    "vst_normalized_counts": f"{proj_dir}{sub_folder_1}EA18.1_ESC_1d-depletion_DESeq2/20220817_EA18-1_ESC-1d_sf-normalized_vst-transformed.csv",
    "tss_annotations": f"{proj_dir}old/RNAseq/STAR_Gencode_alignment/tss_annotions_gencode.vM23.primary_assembly.gtf"
}


# combine all dictionaries
file_paths = {
    "bed_files": bed_files,
    "bw_files": bw_files,
    "enhancer_files": enhancer_files,
    "proj_files": proj_files
}

# save to a JSON file
with open("file_paths.json", "w") as f:
    json.dump(file_paths, f)