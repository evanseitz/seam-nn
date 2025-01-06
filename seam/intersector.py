import os, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm


py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)


#name = 'NXF1'
#name = 'CD40'
name = 'PIK3R3'
#name = 'IRF7'
#name = 'HLA-DRB1'
#name = 'HLA-C'

mode = 'quantity'
#mode = 'profile'

pct_threshold = 0.1 # counts threshold (e.g., 0.1 considers only counts above 10% of the total counts in the cluster)

dir_name = '/examples/outputs_local_clipnet_%s' % name
sequences = pd.read_csv(os.path.join(parent_dir + dir_name + '/heterozygous_pt01/%s_nfolds9' % mode, 'mave.csv'))['Sequence']

#clusters_dir = '/heterozygous_pt01/%s_nfolds9/clusters_umap_kmeans200' % mode
#clusters_dir = '/heterozygous_pt01/%s_nfolds9/clusters_umap_dbscan' % mode
clusters_dir = '/heterozygous_pt01/%s_nfolds9/clusters_hierarchical_cut2' % mode

save_dir = parent_dir + dir_name + clusters_dir
snp_df = pd.read_csv(os.path.join(parent_dir + dir_name, 'dbSNP_coords_%s.csv' % name))
mismatch_fname = 'mismatches_100pct'
mismatch_df = pd.read_csv(os.path.join(parent_dir + dir_name + clusters_dir, '%s.csv' % mismatch_fname))

if 1: # threshold mismatches based on a minimum cluster occupancy
    orig_length = len(mismatch_df)
    mismatch_df = mismatch_df.drop(mismatch_df[mismatch_df['Sum'] < 50].index)
    print('Thresholded %s clusters. Remaining clusters: %s' % (orig_length, len(mismatch_df)))

def snp_logic(allele_code, reference):
    disallowed = []
    allowed = []
    #allowed.append(reference)
    if allele_code == 'R':
        if reference == 'A':
            allowed.extend('RG')
        if reference == 'G':
            allowed.extend('RA')
        disallowed.extend('CTYKMSW')
    if allele_code == 'Y':
        if reference == 'C':
            allowed.extend('YT')
        if reference == 'T':
            allowed.extend('YC')
        disallowed.extend('AGRKMSW')
    if allele_code == 'K':
        if reference == 'G':
            allowed.extend('KT')
        if reference == 'T':
            allowed.extend('KG')
        disallowed.extend('ACRYMSW')
    if allele_code == 'M':
        if reference == 'A':
            allowed.extend('MC')
        if reference == 'C':
            allowed.extend('MA')
        disallowed.extend('GTRYKSW')
    if allele_code == 'S':
        if reference == 'C':
            allowed.extend('SG')
        if reference == 'G':
            allowed.extend('SC')
        disallowed.extend('ATRYKMW')
    if allele_code == 'W':
        if reference == 'A':
            allowed.extend('WT')
        if reference == 'T':
            allowed.extend('WA')
        disallowed.extend('CGRYKMS')
    if allele_code == 'B': # C/G/T (not A)
        disallowed.append('A')
        if reference == 'A':
            allowed.extend('CGT')
        if reference == 'C':
            allowed.extend('GT')
        if reference == 'G':
            allowed.extend('CT')
        if reference == 'T':
            allowed.extend('CG')
        allowed.extend('YKS')
        disallowed.extend('RMW')
    if allele_code == 'D': #A/G/T (not C)
        disallowed.append('C')
        if reference == 'A':
            allowed.extend('GT')
        if reference == 'C':
            allowed.extend('AGT')
        if reference == 'G':
            allowed.extend('AT')
        if reference == 'T':
            allowed.extend('AG')
        allowed.extend('RKW')
        disallowed.extend('YMS')
    if allele_code == 'H': #A/C/T (not G)
        disallowed.append('G')
        if reference == 'A':
            allowed.extend('CT')
        if reference == 'C':
            allowed.extend('AT')
        if reference == 'G':
            allowed.extend('ACT')
        if reference == 'T':
            allowed.extend('AC')
        allowed.extend('YMW')
        disallowed.extend('RKS')
    if allele_code == 'V': #A/C/G (not T)
        disallowed.append('T')
        if reference == 'A':
            allowed.extend('CG')
        if reference == 'C':
            allowed.extend('AG')
        if reference == 'G':
            allowed.extend('AC')
        if reference == 'T':
            allowed.extend('ACG')
        allowed.extend('RMS')
        disallowed.extend('YKW')
    if allele_code == 'N': #A/C/G/T
        allowed.extend('ACGTRYKMSW')
        disallowed = []

    return (allowed, disallowed)

print('test:', snp_logic('R','A'))


def fisher_exact_test(mismatch_df, snp_df):
    # perform Fisher exact test over positions (not considering nucleotide content)
    unique_values1, counts1 = np.unique(mismatch_df['Position'], return_counts=True)
    unique_values2, counts2 = np.unique(snp_df['seam_pos'], return_counts=True)
    possible_values = np.arange(0,500)
    union_values = np.union1d(unique_values1, unique_values2)
    contingency_table = np.zeros((2, 2), dtype=int)
    contingency_table[0,0] = len(np.intersect1d(unique_values1, unique_values2))
    contingency_table[1,1] = len(np.setxor1d(union_values, possible_values))
    contingency_table[1,0] = len(np.setdiff1d(unique_values1, unique_values2))
    contingency_table[0,1] = len(np.setdiff1d(unique_values2, unique_values1))
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    print('')
    print('Contingency table:')
    print(contingency_table)
    print("Odds Ratio:", odds_ratio)
    print("P-value:", p_value)
    print('')


fisher_exact_test(mismatch_df, snp_df)


def seam_snp_intersection(mismatch_df, snp_df, save_dir=None):
    mismatch_df['SNP_allele'] = ""
    mismatch_df['SNP_pos'] = False
    mismatch_df['SNP_base'] = False
    mismatch_df['SNP_conflict'] = False
    pos_matches_unique = 0
    for i in range(len(snp_df)):
        snp_position = snp_df['seam_pos'].iloc[i]
        snp_allele = snp_df['allele'].iloc[i]
        seam_positions = mismatch_df.loc[mismatch_df['Position'] == snp_position]
        if len(seam_positions) != 0:
            pos_matches_unique += 1
        for row in seam_positions.index:
            mismatch_df.loc[row,'SNP_allele'] = snp_allele
            mismatch_df.loc[row,'SNP_pos'] = True
            allowed, disallowed = snp_logic(snp_allele, seam_positions.loc[row]['Reference'])
            for char in allowed:
                seam_count = seam_positions.loc[row][char]
                if seam_count > (pct_threshold * seam_positions.loc[row]['Sum']):
                    mismatch_df.loc[row,'SNP_base'] = True
            for char in disallowed:
                seam_count = seam_positions.loc[row][char]
                if seam_count > (pct_threshold * seam_positions.loc[row]['Sum']):
                    if mismatch_df.loc[row,'SNP_base'] == True:
                        mismatch_df.loc[row, 'SNP_conflict'] = True
                    mismatch_df.loc[row,'SNP_base'] = False


    pos_matches = mismatch_df.loc[mismatch_df['SNP_pos'] == True]
    base_matches = mismatch_df.loc[(mismatch_df['SNP_base'] == True) & (mismatch_df['SNP_pos'] == True)]
    if save_dir is not None:
        mismatch_df.to_csv(os.path.join(save_dir, '%s_SNPs_thresh%spct.csv' % (mismatch_fname, int(pct_threshold*100))), index=False)
        #print(mismatch_df.to_string())
    return (pos_matches_unique, pos_matches, base_matches)

pos_matches_unique, pos_matches, base_matches = seam_snp_intersection(mismatch_df, snp_df, save_dir=save_dir)
observed_statistic = 100*(len(base_matches)/len(snp_df))

print('Percent of positions (with any nucleotide) where both a SEAM mismatch and SNP occur: %.2f%s' % (100*(pos_matches_unique/len(snp_df)), '%'))
print('Percent of matching SEAM/SNP positions with qualifying bases: %.2f%s' % (100*(len(base_matches)/len(pos_matches)), '%'))


if 1:
    FF = mismatch_df[(mismatch_df['SNP_pos'] == False) & (mismatch_df['SNP_base'] == False)].shape[0]
    TF = mismatch_df[(mismatch_df['SNP_pos'] == True) & (mismatch_df['SNP_base'] == False)].shape[0]
    TT = mismatch_df[(mismatch_df['SNP_pos'] == True) & (mismatch_df['SNP_base'] == True)].shape[0]
    TF_conflicts = mismatch_df[(mismatch_df['SNP_conflict'] == True)].shape[0]
    bars = plt.bar([0,1,2,3], [FF, TF, TF_conflicts, TT], color='lightgray', edgecolor='k', linewidth=3)
    plt.xticks(range(4), ['F/F', 'T/F', 'T/F Conflicts', 'T/T'])
    #bars[0].set_edgecolor('#8a2be2')
    #bars[1].set_edgecolor('magenta')
    #bars[2].set_edgecolor('magenta')
    plt.rcParams['savefig.dpi'] = 200
    plt.tight_layout()
    plt.show()
    print('T/F conflicts: %.3f%s' % ((TF_conflicts/TF)*100,'%'))
    print('')

#def mcnemar_test(mismatch_df):
#    from statsmodels.stats.contingency_tables import mcnemar
#    # perform McNemar test over matching SNP/SEAM positions, considering nucleotide content
#    '''contingency_table = pd.crosstab(mismatch_df['SNP_pos'], mismatch_df['SNP_base'])'''
#    a = mismatch_df[(mismatch_df['SNP_pos'] == True) & (mismatch_df['SNP_base'] == True)].shape[0]  # number of concordant pairs (both True)
#    b = mismatch_df[(mismatch_df['SNP_pos'] == True) & (mismatch_df['SNP_base'] == False)].shape[0]  # number of discordant pairs (A True, B False)
#    c = mismatch_df[(mismatch_df['SNP_pos'] == False) & (mismatch_df['SNP_base'] == True)].shape[0]  # number of discordant pairs (A False, B True)
#    d = mismatch_df[(mismatch_df['SNP_pos'] == False) & (mismatch_df['SNP_base'] == False)].shape[0]
#    print([[a, b], [c, d]])
#    result = mcnemar([[a, b], [c, d]], exact=True)
#    print("McNemar's test statistic:", result.statistic)
#    print("P-value:", result.pvalue)

#mcnemar_test(mismatch_df)


def binomial_test(mismatch_df):
    num_successes = mismatch_df[(mismatch_df['SNP_base'] == True)].shape[0]
    num_trials = len(mismatch_df)
    #probabilities = [0.6] * 4 + [0.3] * 6 + 1 # 4/10 inputs have 1/2 probability; 6/10 inputs have 2/10 probability
    #weighted_avg_prob = sum(probabilities) / len(probabilities)
    weighted_avg_prob = (0.3*6 + 0.6*4 + 1)/11.
    print('')
    #print('Weighted average probability:', weighted_avg_prob)
    p_value = stats.binomtest(num_successes, num_trials, p=weighted_avg_prob, alternative='less') #  probability of observing the given number of successes under H0 by chance
    print('Binomial test P-value:', p_value) # if sufficiently small, reject H0 and conclude that the observed number of successes is significantly less than expected by chance

binomial_test(mismatch_df)


if 0: # compute null distribution (scramble bases in SNP dataframe over many trials)
    snp_df_null = snp_df.copy(deep=True)
    allele_options = ['B','D','H','V']
    permuted_statistics = []
    num_trials = 10000
    print('')
    print('Computing null distribution...')
    for i in tqdm(range(num_trials), desc='Trials'):
        for j in range(len(snp_df_null)):
            if snp_df['allele'].iloc[j] == 'A':
                allele_options.extend('RMW')
            if snp_df['allele'].iloc[j] == 'C':
                allele_options.extend('YMS')
            if snp_df['allele'].iloc[j] == 'G':
                allele_options.extend('RKS')
            if snp_df['allele'].iloc[j] == 'T':
                allele_options.extend('YKW')
            snp_df_null.loc[j, 'allele'] = np.random.choice(allele_options, size=1)
        pos_matches_unique, pos_matches, base_matches = seam_snp_intersection(mismatch_df, snp_df_null, save_dir=None)
        permuted_statistics.append(100*(len(base_matches)/len(snp_df)))

    #norm_stat, norm_pval = stats.shapiro(permuted_statistics)
    norm_stat, norm_pval = stats.kstest(permuted_statistics, 'norm')

    if norm_pval >= 0.05: # z-test
        print('Fail to reject H0: not enough evidence to suggest sample does not come from normal distribution')
        mean_trials = np.mean(permuted_statistics)
        std_trials = np.std(permuted_statistics)
        print(mean_trials, std_trials)
        test_statistic = (observed_statistic - mean_trials) / (std_trials / np.sqrt(num_trials))
        p_value = 2 * (1 - stats.norm.cdf(np.abs(test_statistic))) # 2-tailed test for z-test
        print('Z-test P-value:', p_value)
    else:
        print('Reject H0. The data is not normally distributed.')
        U_statistic, p_value = stats.mannwhitneyu([observed_statistic], permuted_statistics, alternative='less') # greater: x is less than the distribution of y
        print('MWU P-value:', p_value)

    if 1: # permutation statistics by counts/proportion
        num_extreme = np.sum(np.array(permuted_statistics) <= observed_statistic)
        p_value = num_extreme / num_trials
        print('P-value (counting):', p_value)
        p_value = np.mean(np.array(permuted_statistics) <= observed_statistic)
        print('P-value (proportion):', p_value)

    if 1:
        plt.hist(permuted_statistics, bins=int(np.sqrt(num_trials)))
        plt.axvline(observed_statistic, color='k', label='observed')
        plt.legend()
        plt.tight_layout()
        plt.show()




        

